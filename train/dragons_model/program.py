"""The programs the C++ trainer runs, as pure functions of fixed shapes.

    act         the policy for every env's acting dragon: its slot's state in,
                an action sampled, the critic's value, the slot's state out
    value       the critic's value alone, changing nothing (bootstraps)
    gather      copies of some slots' states (the start of a training chunk)
    train_step  one PPO step on a minibatch of chunks, replayed from their
                exact start states

`export.py` compiles them ahead of time; C++ calls them through PJRT and
never sees inside. Everything C++ must agree on is in `Config` and is written
next to the programs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import optax

from . import features as F
from . import model as M

MAP_DTYPE = jnp.float16     # slot maps are stored in half precision
LOSS_SCALE = 1024.0         # static loss scale for the float16 backward pass


@dataclass(frozen=True)
class Config:
    num_envs: int = 8192          # rows of `act`
    pool: int = 98304             # dragon slots (h and map) on the rollout device
    inbox_cap: int = 16
    gather: int = 1024            # rows of `gather`
    chunk: int = 32               # turns per training chunk
    minibatch: int = 512          # chunks per grad_step (per learner device)
    learners: int = 4             # devices whose gradients one apply averages
    seed: int = 0

    def layout(self) -> dict:
        return F.layout(self.inbox_cap, True)

    def items(self) -> dict:
        d = asdict(self)
        d["record_bytes"] = self.layout()["bytes"]
        d["hidden"] = M.HIDDEN
        d["map_channels"] = M.MAP_CHANNELS
        d["map_cells"] = M.CELLS
        d["bins"] = M.BINS
        d["actions"] = F.NUM_ACTIONS
        return d


def init_params(key) -> dict:
    ka, kc = jax.random.split(key)
    return {"actor": M.init_actor(ka), "critic": M.init_critic(kc)}


def _read(pool_h, pool_map, pool_bins, slots, fresh):
    """Slot states as a turn reads them: zero for a dragon's first turn."""
    keep = (~fresh)[:, None]
    return (jnp.where(keep, pool_h[slots], 0.0),
            jnp.where(keep[:, :, None], pool_map[slots], jnp.zeros((), MAP_DTYPE)),
            jnp.where(keep[:, :, None], pool_bins[slots], 0.0))


def make_act(cfg: Config):
    L = cfg.layout()

    def act(params, records, slots, fresh, pool_h, pool_map, pool_bins, key):
        """records u8[N, R]; slots i32[N]; fresh bool[N] (a dragon's first turn: zero state);
        pool_h f32[S, 512]; pool_map f16[S, 1024, 16]; pool_bins f32[S, 64, 16]; key u32[2].
        -> action i32[N], log_prob f32[N], value f32[N], pool_h, pool_map, pool_bins, key"""
        x = F.decode(records, L)
        h, m, bins = _read(pool_h, pool_map, pool_bins, slots, fresh)
        logits, h2, m2, bins2 = M.actor_step(params["actor"], x, h, m, bins)
        log_probs = M.masked_log_probs(logits, x.mask)
        key, sub = jax.random.split(key)
        action = jax.random.categorical(sub, log_probs).astype(jnp.int32)
        log_prob = jnp.take_along_axis(log_probs, action[:, None], 1)[:, 0]
        value = M.critic_value(params["critic"], F.decode_state(records, L), h2)
        return (action, log_prob, value, pool_h.at[slots].set(h2), pool_map.at[slots].set(m2),
                pool_bins.at[slots].set(bins2), key)

    return act


def make_value(cfg: Config):
    L = cfg.layout()

    def value(params, records, slots, fresh, pool_h, pool_map, pool_bins):
        """The critic's value for every env's acting dragon, changing nothing (rollout-end bootstrap)."""
        x = F.decode(records, L)
        h, m, bins = _read(pool_h, pool_map, pool_bins, slots, fresh)
        _, h2, _, _ = M.actor_step(params["actor"], x, h, m, bins)
        return M.critic_value(params["critic"], F.decode_state(records, L), h2)

    return value


def make_gather(cfg: Config):
    def gather(pool_h, pool_map, pool_bins, slots, fresh):
        """The states of `slots` (i32[G]) as `act` would read them."""
        return _read(pool_h, pool_map, pool_bins, slots, fresh)

    return gather


METRICS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "actor_grad_norm",
           "critic_grad_norm", "samples")


def make_train_step(cfg: Config):
    L = cfg.layout()
    adam = optax.scale_by_adam()

    def loss_fn(params, batch, hyper):
        records, valid, actions, old_log_prob, advantage, returns, h0, m0, bins0 = batch
        clip, ent_coef, vf_coef = hyper["clip"], hyper["ent_coef"], hyper["vf_coef"]

        @jax.checkpoint
        def step(carry, t):
            h, m, bins = carry
            rec, ok, a, old_lp, adv, ret = t
            x = F.decode(rec, L)
            logits, h2, m2, bins2 = M.actor_step(params["actor"], x, h, m, bins)
            lp_all = M.masked_log_probs(logits, x.mask)
            lp = jnp.take_along_axis(lp_all, a[:, None], 1)[:, 0]
            ratio = jnp.exp(lp - old_lp)
            surrogate = jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv)
            value = M.critic_value(params["critic"], F.decode_state(rec, L), h2)
            okf = ok.astype(jnp.float32)
            terms = jnp.stack([
                -(surrogate * okf).sum(),
                ((value - ret) ** 2 * okf).sum(),
                (M.entropy(lp_all, x.mask) * okf).sum(),
                (((ratio - 1) - (lp - old_lp)) * okf).sum(),                       # approx KL (k3)
                ((jnp.abs(ratio - 1) > clip) * okf).sum(),
            ])
            # padded steps leave the state as it was
            h2 = jnp.where(ok[:, None], h2, h)
            m2 = jnp.where(ok[:, None, None], m2, m)
            bins2 = jnp.where(ok[:, None, None], bins2, bins)
            return (h2, m2, bins2), terms

        _, terms = jax.lax.scan(step, (h0, m0, bins0),
                                     (records, valid, actions, old_log_prob, advantage, returns))
        sums = terms.sum(0)
        n = jnp.maximum(valid.sum().astype(jnp.float32), 1.0)
        means = sums / n
        loss = means[0] + vf_coef * means[1] - ent_coef * means[2]
        return loss, (means, n)

    def grad_step(params, records, valid, actions, old_log_prob, advantage, returns, h0, m0, bins0,
                  clip, ent_coef, vf_coef):
        """The gradient of one minibatch, for data-parallel learners.
        -> grads (params' structure), metrics f32[6]: the 5 loss terms and the sample count

        The loss is scaled by LOSS_SCALE before differentiating (float16 matmuls would
        otherwise flush small activation gradients to zero) and the gradients unscaled."""
        hyper = {"clip": clip, "ent_coef": ent_coef, "vf_coef": vf_coef}
        batch = (records, valid, actions, old_log_prob, advantage, returns, h0, m0, bins0)
        scaled = lambda p: (lambda l, aux: (l * LOSS_SCALE, aux))(*loss_fn(p, batch, hyper))
        (_, (means, n)), grads = jax.value_and_grad(scaled, has_aux=True)(params)
        grads = jax.tree_util.tree_map(lambda g: g / LOSS_SCALE, grads)
        return grads, jnp.concatenate([means, n[None]])

    def apply(params, opt_state, grads, lr, max_grad_norm):
        """Average the learners' gradients (`grads`: a tuple of K gradient trees), clip the actor's
        and the critic's separately, and take an Adam step. -> params, opt_state, grad norms f32[2]"""
        mean = jax.tree_util.tree_map(lambda *g: sum(g) / len(g), *grads)
        # actor and critic clipped separately, so one cannot starve the other
        clipped, norms = {}, []
        for group in ("actor", "critic"):
            g = mean[group]
            norm = optax.tree.norm(g)
            scale = jnp.minimum(1.0, max_grad_norm / (norm + 1e-6))
            clipped[group] = jax.tree_util.tree_map(lambda v: v * scale, g)
            norms.append(norm)
        # a step whose gradient overflowed (inf / nan) is skipped: parameters and Adam unchanged
        finite = jnp.isfinite(norms[0]) & jnp.isfinite(norms[1])
        updates, new_opt = adam.update(clipped, opt_state, params)
        new_params = jax.tree_util.tree_map(lambda p, u: p - lr * u, params, updates)
        keep = lambda new, old: jax.tree_util.tree_map(lambda a, b: jnp.where(finite, a, b), new, old)
        norms = jnp.where(finite, jnp.stack(norms), -1.0)       # -1 in the log: a skipped step
        return keep(new_params, params), keep(new_opt, opt_state), norms

    def train_step(params, opt_state, records, valid, actions, old_log_prob, advantage, returns, h0, m0, bins0,
                   lr, clip, ent_coef, vf_coef, max_grad_norm):
        """grad_step then apply on one device (tests use it). -> params, opt_state, metrics f32[8]"""
        grads, m = grad_step(params, records, valid, actions, old_log_prob, advantage, returns, h0, m0, bins0,
                             clip, ent_coef, vf_coef)
        params, opt_state, norms = apply(params, opt_state, (grads,), lr, max_grad_norm)
        return params, opt_state, jnp.concatenate([m[:5], norms, m[5:]])

    train_step.grad_step = grad_step
    train_step.apply = apply
    return train_step, adam


def shapes(cfg: Config):
    """Abstract inputs of each program, for export."""
    L = cfg.layout()
    R, N, S, G, T, B = L["bytes"], cfg.num_envs, cfg.pool, cfg.gather, cfg.chunk, cfg.minibatch
    sd = jax.ShapeDtypeStruct
    params = jax.eval_shape(init_params, jax.random.PRNGKey(0))
    opt = jax.eval_shape(optax.scale_by_adam().init, params)
    h = sd((S, M.HIDDEN), jnp.float32)
    m = sd((S, M.CELLS, M.MAP_CHANNELS), MAP_DTYPE)
    bins = sd((S, M.BINS, M.MAP_CHANNELS), jnp.float32)
    scalar = sd((), jnp.float32)
    return {
        "act": (params, sd((N, R), jnp.uint8), sd((N,), jnp.int32), sd((N,), jnp.bool_), h, m, bins,
                sd((2,), jnp.uint32)),
        "value": (params, sd((N, R), jnp.uint8), sd((N,), jnp.int32), sd((N,), jnp.bool_), h, m, bins),
        "gather": (h, m, bins, sd((G,), jnp.int32), sd((G,), jnp.bool_)),
        "grad_step": (params, sd((T, B, R), jnp.uint8), sd((T, B), jnp.bool_), sd((T, B), jnp.int32),
                      sd((T, B), jnp.float32), sd((T, B), jnp.float32), sd((T, B), jnp.float32),
                      sd((B, M.HIDDEN), jnp.float32), sd((B, M.CELLS, M.MAP_CHANNELS), MAP_DTYPE),
                      sd((B, M.BINS, M.MAP_CHANNELS), jnp.float32), scalar, scalar, scalar),
        "apply": (params, opt, tuple(params for _ in range(cfg.learners)), scalar, scalar),
    }
