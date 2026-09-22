"""Export the actor as TorchScript so the native trainer runs the same graph.

Rather than reimplement the network in C++ and risk it drifting from the Python
one, the model is defined once here and scripted. The C++ trainer loads the
scripted module, runs it, and optimises its parameters directly -- there is only
ever one definition of the policy.

The scripted wrapper is plain tensors in, plain tensors out: TorchScript cannot
take the dataclasses the eager path uses.
"""

from __future__ import annotations

import pathlib

import torch
from torch import Tensor, nn

from .actor import NEG_INF, Actor
from .critic import CriticState, TeamCritic


class ScriptableActor(nn.Module):
    """One timestep of the actor, as tensors."""

    def __init__(self, actor: Actor):
        super().__init__()
        self.spatial = actor.spatial
        self.scalar = actor.scalar
        self.message = actor.message
        self.fuse = actor.fuse
        # A single timestep through cuDNN's RNN path costs a weight compaction
        # per call and is the wrong primitive for seq_len 1, so the gate maths
        # is done directly on the same parameters. test_model.py checks this
        # against nn.GRU.
        self.weight_ih = actor.gru.weight_ih_l0
        self.weight_hh = actor.gru.weight_hh_l0
        self.bias_ih = actor.gru.bias_ih_l0
        self.bias_hh = actor.gru.bias_hh_l0
        self.action_head = actor.action_head
        self.value_head = actor.value_head
        self.transmit_head = actor.transmit_head
        self.symbol_head = actor.symbol_head
        self.n_symbols = int(actor.n_symbols)
        self.symbol_vocab = int(actor.symbol_vocab)
        self.hidden_size = int(actor.hidden)
        self.sonar = bool(actor.enable_sonar)
        self.neg_inf = float(NEG_INF)   # TorchScript cannot close over globals

    def forward(self, grid: Tensor, scalars: Tensor, messages: Tensor,
                message_mask: Tensor, action_mask: Tensor, hidden: Tensor
                ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        features = self.fuse(torch.cat(
            [self.spatial(grid), self.scalar(scalars),
             self.message(messages, message_mask)], dim=-1))

        # one GRU step, matching torch.nn.GRU's gate order (r, z, n)
        gates_input = torch.addmm(self.bias_ih, features, self.weight_ih.t())
        gates_hidden = torch.addmm(self.bias_hh, hidden, self.weight_hh.t())
        size = self.hidden_size
        reset = torch.sigmoid(gates_input[:, :size] + gates_hidden[:, :size])
        update = torch.sigmoid(gates_input[:, size:2 * size] +
                               gates_hidden[:, size:2 * size])
        candidate = torch.tanh(gates_input[:, 2 * size:] +
                               reset * gates_hidden[:, 2 * size:])
        state = (1.0 - update) * candidate + update * hidden
        new_hidden = state

        logits = self.action_head(state)
        # a boxed-in dragon has nothing legal; leave its row alone rather than
        # emit a row of -inf that would sample NaNs
        usable = action_mask | torch.logical_not(action_mask.any(-1, keepdim=True))
        logits = logits.masked_fill(torch.logical_not(usable), self.neg_inf)

        symbols = self.symbol_head(state).view(
            state.size(0), self.n_symbols, self.symbol_vocab)
        return (logits, self.transmit_head(state).squeeze(-1), symbols, new_hidden,
                self.value_head(state).squeeze(-1))


class ScriptableCritic(nn.Module):
    """The team critic as tensors, for the native trainer.

    Spec section 4's baseline: full state in, one value per player out. The
    actor never sees any of this -- it is loaded by the learner only.
    """

    def __init__(self, critic: TeamCritic):
        super().__init__()
        self.map_body = critic.map_body
        self.map_head = critic.map_head
        self.dragon_encoder = critic.dragon_encoder
        self.set_head = critic.set_head
        self.value = critic.value
        self.pairs_a = [a for a, _ in critic.team_channel_pairs]
        self.pairs_b = [b for _, b in critic.team_channel_pairs]
        self.zero_sum = bool(critic.zero_sum)
        # torch.finfo is not scriptable; any value below every real activation
        # does the same job for a masked max.
        self.very_negative = -1.0e30

    def _side(self, map_grid: Tensor, dragons: Tensor, team: Tensor,
              mask: Tensor) -> Tensor:
        own = torch.nn.functional.one_hot(team.clamp(0, 1), 2).to(dragons.dtype)
        encoded = self.dragon_encoder(torch.cat([dragons, own], dim=-1))
        weight = mask.unsqueeze(-1).to(encoded.dtype)
        mean = (encoded * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
        peak = encoded.masked_fill(torch.logical_not(mask.unsqueeze(-1)),
                                   self.very_negative)
        peak = peak.max(dim=1).values
        peak = torch.where(mask.any(dim=1, keepdim=True), peak, torch.zeros_like(peak))
        pooled = self.set_head(torch.cat([mean, peak], dim=-1))
        spatial = self.map_head(self.map_body(map_grid))
        return self.value(torch.cat([spatial, pooled], dim=-1)).squeeze(-1)

    def forward(self, map_grid: Tensor, dragons: Tensor, team: Tensor,
                mask: Tensor) -> Tensor:
        """[B, 2] -- the value for team A and for team B."""
        a = self._side(map_grid, dragons, team, mask)
        # the other side's view: swap the team-specific map channels and flip
        # the own/enemy flags, so both values come from the same weights
        flipped = map_grid.clone()
        for i in range(len(self.pairs_a)):
            first, second = self.pairs_a[i], self.pairs_b[i]
            swap = flipped[:, first].clone()
            flipped[:, first] = flipped[:, second]
            flipped[:, second] = swap
        b = self._side(flipped, dragons, 1 - team, mask)
        if self.zero_sum:
            half = (a - b) * 0.5
            return torch.stack([half, -half], dim=-1)
        return torch.stack([a, b], dim=-1)


def export_critic(critic: TeamCritic, path: str | pathlib.Path) -> pathlib.Path:
    module = ScriptableCritic(critic).eval()
    scripted = torch.jit.script(module)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(path))
    return path


def export(actor: Actor, path: str | pathlib.Path) -> pathlib.Path:
    """Script the actor and write it where the native trainer can load it."""
    module = ScriptableActor(actor).eval()
    scripted = torch.jit.script(module)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(path))
    return path
