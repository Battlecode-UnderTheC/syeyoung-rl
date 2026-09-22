"""The decentralised actor: one GRU state per dragon, nothing global.

Its only inputs are the dragon's own 7x7 window, its own status scalars, the
messages that were delivered to it, its own legal-action mask and its own
recurrent state. There is deliberately no argument through which global state
could arrive.

The action is a joint of conditional heads:

    action  : one of the 47 move/split actions, masked to the legal ones
    transmit: whether to fire the sonar at all
    symbols : the payload, only when transmit is on

so the joint log-prob and entropy are sums over the *active* heads -- when
transmit is off the payload contributes exactly nothing to either, and PPO must
recompute them the same way it stored them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Bernoulli, Categorical

from ..env.observation import N_CHANNELS, N_SCALARS
from .encoder import MessageEncoder, ScalarEncoder, SpatialEncoder, bytes_to_symbols

NEG_INF = -1e9


def _gumbel_argmax(logits: Tensor) -> Tensor:
    """Exact categorical sampling without building a distribution object."""
    noise = torch.rand_like(logits).clamp_(min=torch.finfo(logits.dtype).tiny)
    return (logits - torch.log(-torch.log(noise))).argmax(dim=-1)


@dataclass
class ActorObs:
    """A batch of per-dragon observations. Time-major entries are [B, T, ...]."""
    grid: Tensor              # [B, C, H, W] or [B, T, C, H, W]
    scalars: Tensor           # [B, D]       or [B, T, D]
    messages: Tensor          # [B, K, S]    or [B, T, K, S] symbol ids
    message_mask: Tensor      # [B, K]       or [B, T, K]
    action_mask: Tensor       # [B, A]       or [B, T, A]

    def flatten_time(self) -> tuple["ActorObs", int, int]:
        b, t = self.grid.shape[0], self.grid.shape[1]
        return ActorObs(
            self.grid.reshape(b * t, *self.grid.shape[2:]),
            self.scalars.reshape(b * t, *self.scalars.shape[2:]),
            self.messages.reshape(b * t, *self.messages.shape[2:]),
            self.message_mask.reshape(b * t, *self.message_mask.shape[2:]),
            self.action_mask.reshape(b * t, *self.action_mask.shape[2:]),
        ), b, t


@dataclass
class JointAction:
    action: Tensor            # [B] long
    transmit: Tensor          # [B] bool
    symbols: Tensor           # [B, S] long, meaningless where transmit is False


@dataclass
class PolicyHeads:
    """Logits for the conditional heads, with the action head already masked.

    `sonar` off means the communication heads are not part of the action at all
    -- they contribute nothing to the joint log-prob or entropy, which is what
    the early curriculum stages want.
    """
    action_logits: Tensor     # [B, A]
    transmit_logit: Tensor    # [B]
    symbol_logits: Tensor     # [B, S, V]
    sonar: bool = True

    @property
    def action_dist(self) -> Categorical:
        return Categorical(logits=self.action_logits)

    @property
    def transmit_dist(self) -> Bernoulli:
        return Bernoulli(logits=self.transmit_logit)

    @property
    def symbol_dist(self) -> Categorical:
        return Categorical(logits=self.symbol_logits)

    def sample(self, deterministic: bool = False) -> JointAction:
        if deterministic:
            action = self.action_logits.argmax(-1)
            transmit = self.transmit_logit > 0
            symbols = self.symbol_logits.argmax(-1)
        else:
            action = self.action_dist.sample()
            transmit = self.transmit_dist.sample().bool()
            symbols = self.symbol_dist.sample()
        if not self.sonar:
            transmit = torch.zeros_like(transmit)
            symbols = torch.zeros_like(symbols)
        return JointAction(action, transmit, symbols)

    def sample_with_log_prob(self, deterministic: bool = False
                             ) -> tuple[JointAction, Tensor]:
        """Sample and score in one pass, without building distribution objects.

        `torch.distributions` costs more than the network itself at batch 1, and
        a rollout never needs entropy -- PPO recomputes it during the update.
        The result is identical to `sample()` followed by `log_prob()`.
        """
        action_logp = torch.log_softmax(self.action_logits, dim=-1)
        if deterministic:
            action = self.action_logits.argmax(dim=-1)
        else:
            action = _gumbel_argmax(self.action_logits)
        logp = action_logp.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        if not self.sonar:
            return JointAction(action, torch.zeros_like(action, dtype=torch.bool),
                               torch.zeros_like(self.symbol_logits[..., 0],
                                                dtype=torch.long)), logp

        if deterministic:
            transmit = self.transmit_logit > 0
        else:
            transmit = torch.rand_like(self.transmit_logit) < torch.sigmoid(
                self.transmit_logit)
        logp = logp + torch.where(transmit, F.logsigmoid(self.transmit_logit),
                                  F.logsigmoid(-self.transmit_logit))

        symbol_logp = torch.log_softmax(self.symbol_logits, dim=-1)
        if deterministic:
            symbols = self.symbol_logits.argmax(dim=-1)
        else:
            symbols = _gumbel_argmax(self.symbol_logits)
        payload = symbol_logp.gather(-1, symbols.unsqueeze(-1)).squeeze(-1).sum(-1)
        logp = logp + payload * transmit.to(logp.dtype)
        return JointAction(action, transmit, symbols), logp

    def log_prob(self, joint: JointAction) -> Tensor:
        """Sum over the heads that were actually active."""
        logp = self.action_dist.log_prob(joint.action)
        if not self.sonar:
            return logp
        logp = logp + self.transmit_dist.log_prob(joint.transmit.to(logp.dtype))
        payload = self.symbol_dist.log_prob(joint.symbols).sum(-1)
        return logp + payload * joint.transmit.to(logp.dtype)

    def entropy(self) -> Tensor:
        """Conditional entropy: the payload counts in proportion to transmitting."""
        if not self.sonar:
            return self.action_dist.entropy()
        transmit = self.transmit_dist
        p_transmit = torch.sigmoid(self.transmit_logit)
        return (self.action_dist.entropy() + transmit.entropy()
                + p_transmit * self.symbol_dist.entropy().sum(-1))


class Actor(nn.Module):
    def __init__(self, n_actions: int, in_channels: int = N_CHANNELS,
                 n_scalars: int = N_SCALARS,
                 grid: int = 7, hidden: int = 256, symbol_vocab: int = 16,
                 n_symbols: int = 4, spatial_dim: int = 256, scalar_dim: int = 64,
                 message_dim: int = 64, enable_sonar: bool = True):
        super().__init__()
        self.n_actions = n_actions
        self.enable_sonar = enable_sonar
        self.hidden = hidden
        self.symbol_vocab = symbol_vocab
        self.n_symbols = n_symbols

        self.spatial = SpatialEncoder(in_channels, grid, spatial_dim)
        self.scalar = ScalarEncoder(n_scalars, scalar_dim)
        self.message = MessageEncoder(symbol_vocab, n_symbols, out_dim=message_dim)
        self.fuse = nn.Sequential(
            nn.Linear(spatial_dim + scalar_dim + message_dim, hidden), nn.SiLU())

        # one module used for both single steps and sequences, so the weights a
        # rollout acted with are exactly the weights the update differentiates
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)

        self.action_head = nn.Linear(hidden, n_actions)
        self.transmit_head = nn.Linear(hidden, 1)
        self.symbol_head = nn.Linear(hidden, n_symbols * symbol_vocab)
        # A value head on the shared trunk. This is the IPPO option from the
        # spec's ablation list: it sees only what the actor sees. The centralised
        # team critic in critic.py remains the CTDE alternative.
        self.value_head = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.value_head.weight, 1.0)
        nn.init.zeros_(self.value_head.bias)
        for head, gain in ((self.action_head, 0.01), (self.transmit_head, 0.01),
                           (self.symbol_head, 0.01)):
            nn.init.orthogonal_(head.weight, gain)
            nn.init.zeros_(head.bias)

    # -- state ------------------------------------------------------------

    def initial_state(self, batch: int, device=None, dtype=None) -> Tensor:
        return torch.zeros(batch, self.hidden, device=device,
                           dtype=dtype or torch.float32)

    # -- encoding ---------------------------------------------------------

    def encode(self, obs: ActorObs) -> Tensor:
        parts = (self.spatial(obs.grid), self.scalar(obs.scalars),
                 self.message(obs.messages, obs.message_mask))
        return self.fuse(torch.cat(parts, dim=-1))

    def _heads(self, features: Tensor, action_mask: Tensor) -> PolicyHeads:
        logits = self.action_head(features)
        if action_mask is not None:
            # a boxed-in dragon has nothing legal; leave its row untouched rather
            # than produce a row of -inf that would sample NaNs
            usable = action_mask | ~action_mask.any(-1, keepdim=True)
            logits = logits.masked_fill(~usable, NEG_INF)
        symbols = self.symbol_head(features).view(*features.shape[:-1],
                                                  self.n_symbols, self.symbol_vocab)
        return PolicyHeads(logits, self.transmit_head(features).squeeze(-1), symbols,
                           sonar=self.enable_sonar)

    # -- one step (rollout) ----------------------------------------------

    def step(self, obs: ActorObs, hidden: Tensor) -> tuple[PolicyHeads, Tensor]:
        """One timestep for a batch of dragons. `hidden` is [B, H]."""
        features = self.encode(obs).unsqueeze(1)                   # [B, 1, H]
        out, new_hidden = self.gru(features, hidden.unsqueeze(0))
        return self._heads(out.squeeze(1), obs.action_mask), new_hidden.squeeze(0)

    def value(self, obs: ActorObs, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """State value from the same trunk, for the IPPO setup."""
        features = self.encode(obs).unsqueeze(1)
        out, new_hidden = self.gru(features, hidden.unsqueeze(0))
        return self.value_head(out.squeeze(1)).squeeze(-1), new_hidden.squeeze(0)

    @torch.no_grad()
    def act(self, obs: ActorObs, hidden: Tensor, deterministic: bool = False,
            with_entropy: bool = False
            ) -> tuple[JointAction, Tensor, Tensor | None, Tensor]:
        """Sample an action. Returns (joint, log_prob, entropy or None, hidden).

        Entropy is off by default: it is a diagnostic during a rollout and the
        update recomputes it anyway, and at batch 1 it costs more than the GRU.
        """
        heads, new_hidden = self.step(obs, hidden)
        joint, log_prob = heads.sample_with_log_prob(deterministic)
        entropy = heads.entropy() if with_entropy else None
        return joint, log_prob, entropy, new_hidden

    # -- sequences (training) --------------------------------------------

    def unroll(self, obs: ActorObs, hidden: Tensor) -> tuple[PolicyHeads, Tensor]:
        """A contiguous per-dragon sequence: obs entries are [B, T, ...]."""
        flat, b, t = obs.flatten_time()
        features = self.encode(flat).view(b, t, self.hidden)
        out, new_hidden = self.gru(features, hidden.unsqueeze(0))
        heads = self._heads(out, obs.action_mask)
        return heads, new_hidden.squeeze(0)

    def evaluate(self, obs: ActorObs, hidden: Tensor, joint: JointAction
                 ) -> tuple[Tensor, Tensor, Tensor]:
        """Recompute log-prob and entropy for stored actions, with the same
        conditional head structure the rollout used.

        `joint` must carry the same leading dims as the observations, so for a
        sequence its fields are [B, T] (and [B, T, S] for the symbols).
        """
        heads, new_hidden = self.unroll(obs, hidden)
        return heads.log_prob(joint), heads.entropy(), new_hidden

    # -- helpers ----------------------------------------------------------

    def symbols_from_messages(self, messages: Tensor) -> Tensor:
        return bytes_to_symbols(messages, self.symbol_vocab)

    def payload(self, joint: JointAction) -> Tensor:
        """Symbols packed into the uint32 the engine carries, one symbol per byte."""
        shift = torch.arange(self.n_symbols, device=joint.symbols.device) * 8
        return (joint.symbols.long() << shift).sum(-1)
