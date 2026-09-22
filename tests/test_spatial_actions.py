"""Action codec and actor acceptance tests (spec sections 6, 13 and 16)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from spatial_helpers import NativeMatch, _dragons, two_dragons
from dragons_env.mapgen import MapSpec, edge_between, snake, west
from snake_rl.env.actions import Command, CommandCodec
from snake_rl.env.schema import (PF, TOKEN_STOP, TOP_MOVE, TOP_SPLIT, N_PREFIX_FEATURES,
                                 N_TOKENS)
from snake_rl.model.actor import (SpatialActor, SpatialActorConfig, SpatialInputs,
                                  StoredCommand)
from snake_rl.model.policy import SpatialPolicy

pytestmark = pytest.mark.skipif(_dragons is None, reason="native sim not built")
codec = CommandCodec(8)


def first_turn(spec: MapSpec, dragon_id: int = 0) -> tuple[NativeMatch, object]:
    m = NativeMatch(spec)
    m.game.begin_round()
    return m, m.observe(dragon_id)


# -- split semantics ----------------------------------------------------------------

def test_a_length_40_parent_detaching_36_emits_split_36_and_keeps_four():
    w = h = 51
    spec = MapSpec(w, h, [(0, snake(45, 25, 40, "E", w, h)), (1, snake(5, 5, 3, "E", w, h))],
                   pearl_gap=(0, 0))
    m, obs = first_turn(spec)
    sizes, feats = codec.split_table(obs)
    assert sizes.min() == 2 and sizes.max() == 38 and len(sizes) == 37
    k = int(np.flatnonzero(sizes == 36)[0])
    cmd = Command(TOP_SPLIT, split_n=36, split_index=k)
    assert cmd.text == "SPLIT 36"
    assert codec.retained(40, 36) == 4
    m.act(0, cmd.text)
    assert m.dragon(0)["length"] == 4 and m.dragon(2)["length"] == 36


def test_a_command_is_either_a_move_or_a_split_never_both():
    assert Command(TOP_MOVE, dirs="NE").text == "MOVE NE"
    assert Command(TOP_SPLIT, split_n=3).text == "SPLIT 3"
    assert Command(TOP_MOVE, dirs="N", sonar=5).text == "MOVE N\nSONAR 4"


def test_split_is_masked_only_by_its_actual_constraints():
    m, obs = first_turn(two_dragons(la=3, lb=3))
    assert not obs.split_legal and codec.split_table(obs)[0].size == 0
    m, obs = first_turn(two_dragons(la=4, lb=3))
    assert obs.split_legal and codec.split_table(obs)[0].tolist() == [2]
    w = h = 21
    spec = MapSpec(w, h, [(0, snake(5, 10, 6, "E", w, h)), (1, snake(5, 2, 3, "E", w, h))],
                   pearl_gap=(0, 0), unit_limit=1)
    m, obs = first_turn(spec)
    assert not obs.split_legal, "at the unit limit there is no split"


# -- sprint affordability ---------------------------------------------------------

@pytest.mark.parametrize("length,dirs,ok", [
    (4, "EEE", True), (4, "EEEE", False), (3, "EE", True), (3, "EEE", False), (2, "EE", False),
])
def test_paid_steps_are_masked_exactly_when_the_engine_would_refuse_them(length, dirs, ok):
    m, obs = first_turn(two_dragons(la=length, lb=3))
    if ok:
        trace = codec.trace_move(obs, m.states[0], dirs)
        assert codec.dirs_of(trace) == dirs
        m.act(0, "MOVE " + dirs)
        assert m.dragon(0)["alive"]
    else:
        with pytest.raises(ValueError):
            codec.trace_move(obs, m.states[0], dirs)
        m.act(0, "MOVE " + dirs)
        assert not m.dragon(0)["alive"] and m.game.deaths()[-1][2] == "A"


def test_food_on_an_unaffordable_destination_cannot_pay_for_entry():
    w = h = 21
    spec = MapSpec(w, h, [(0, snake(5, 10, 3, "E", w, h)), (1, snake(5, 2, 3, "E", w, h))],
                   pearl_gap=(0, 0), tile_gaps={(8, 10): (1, 1)})
    m, obs = first_turn(spec)
    assert obs.facts[3, 10, 8] == 1.0, "the pearl on the third tile is visible"
    with pytest.raises(ValueError):
        codec.trace_move(obs, m.states[0], "EEE")
    m.act(0, "MOVE EEE")
    assert not m.dragon(0)["alive"]


def test_a_pearl_eaten_part_way_pays_for_another_step():
    w = h = 21
    spec = MapSpec(w, h, [(0, snake(5, 10, 3, "E", w, h)), (1, snake(5, 2, 3, "E", w, h))],
                   pearl_gap=(0, 0), tile_gaps={(7, 10): (1, 1)})
    m, obs = first_turn(spec)
    trace = codec.trace_move(obs, m.states[0], "EEE")
    assert trace.active[:3].all()
    assert trace.prefix[2][PF.LENGTH_KNOWN] == 1.0 and trace.prefix[2][PF.CAN_PAY] == 1.0
    assert trace.prefix[2][PF.PEARLS_EATEN] == pytest.approx(1 / 8)
    m.act(0, "MOVE EEE")
    assert m.dragon(0)["alive"] and m.dragon(0)["length"] == 2


def test_prefix_simulation_never_reads_an_unseen_portal_exit():
    w = h = 21
    portals = [(west(8, 10, w), 4), (west(15, 3, w), 4)]
    spec = MapSpec(w, h, [(0, snake(7, 10, 6, "E", w, h)), (1, snake(2, 18, 3, "E", w, h))],
                   portals=portals, pearl_gap=(0, 0))
    m, obs = first_turn(spec)
    trace = codec.trace_move(obs, m.states[0], "EN")
    assert trace.prefix[0][PF.AHEAD_PORTAL_E] == 1.0
    assert trace.prefix[1][PF.POSITION_KNOWN] == 0.0 and trace.prefix[1][PF.RESOURCE_UNCERTAIN] == 1.0
    assert trace.prefix[1][PF.AHEAD_UNKNOWN_N:PF.AHEAD_UNKNOWN_N + 4].sum() == 4.0
    assert trace.masks[1].all(), "an uncertain continuation stays available"


# -- fatal moves stay selectable ----------------------------------------------------

def test_head_collisions_stay_selectable_and_kill_both():
    w = h = 21
    spec = MapSpec(w, h, [(0, snake(8, 10, 3, "E", w, h)), (1, snake(9, 10, 9, "W", w, h))],
                   pearl_gap=(0, 0))
    m, obs = first_turn(spec)
    trace = codec.trace_move(obs, m.states[0], "E")
    assert trace.masks[0][1], "moving into the enemy head is a real action"
    assert trace.prefix[0][PF.OCC_HEAD_E] == 1.0
    m.act(0, "MOVE E")
    assert {d[0] for d in m.game.deaths()} == {0, 1} and all(d[2] == "H" for d in m.game.deaths())


def test_a_known_fatal_step_forces_stop_afterwards_but_is_itself_allowed():
    w = h = 21
    spec = MapSpec(w, h, [(0, snake(5, 10, 6, "E", w, h)), (1, snake(5, 2, 3, "E", w, h))],
                   kelp=[edge_between(5, 10, "E", w, h)], pearl_gap=(0, 0))
    m, obs = first_turn(spec)
    trace = codec.trace_move(obs, m.states[0], "E")
    assert trace.prefix[0][PF.AHEAD_KELP_E] == 1.0 and trace.masks[0][1]
    assert not trace.masks[1][:4].any() and trace.masks[1][TOKEN_STOP]
    assert not trace.active[1], "the forced STOP is not a sampled token"
    with pytest.raises(ValueError):
        codec.trace_move(obs, m.states[0], "EE")


def test_reversing_into_the_neck_is_a_known_self_collision():
    m, obs = first_turn(two_dragons(la=5, lb=3))
    trace = codec.trace_move(obs, m.states[0], "W")
    assert trace.prefix[0][PF.OCC_SELF_W] == 1.0 and trace.masks[0][3]


# -- actor: sampling and evaluation share one structure ---------------------------

@pytest.fixture(scope="module")
def actor() -> SpatialActor:
    torch.manual_seed(0)
    return SpatialActor(SpatialActorConfig(sonar=True)).eval()


def _batch(n_boards: int = 3):
    items, states = [], []
    for i in range(n_boards):
        w = h = 17
        spec = MapSpec(w, h, [(0, snake(4 + i, 8, 5 + 2 * i, "E", w, h)), (1, snake(12, 8 + i, 4, "W", w, h))],
                       pearl_gap=(3, 3))
        m = NativeMatch(spec)
        m.play_round()
        m.game.begin_round()
        items.append(m.observe(0))
        states.append(m.states[0])
    return items, states


def test_sampled_log_probability_equals_teacher_forced_evaluation(actor):
    items, states = _batch(4)
    x = SpatialInputs.from_prepared(items, codec)
    M, h = actor.initial_state(4, 17, 17)
    torch.manual_seed(1)
    with torch.no_grad():
        ctx, M2, h2 = actor.encode_and_update(x, M, h)
        for _ in range(6):
            cmds, lp, stored = actor.decode_sample(ctx, x, items, states, codec)
            lp2, ent, sent = actor.decode_evaluate(ctx, x, stored)
            assert torch.allclose(lp, lp2, atol=1e-6)
            assert torch.isfinite(ent).all() and torch.isfinite(sent).all()
            for c in cmds:
                assert c.text == c.text.upper() and len(c.text.encode()) < 10_000
                if c.top == TOP_MOVE:
                    assert 1 <= len(c.dirs) <= 8 and set(c.dirs) <= set("NESW")


def test_inactive_tokens_padding_and_forced_stops_contribute_nothing(actor):
    items, states = _batch(3)
    x = SpatialInputs.from_prepared(items, codec)
    M, h = actor.initial_state(3, 17, 17)
    torch.manual_seed(2)
    with torch.no_grad():
        ctx, _, _ = actor.encode_and_update(x, M, h)
        cmds, lp, stored = actor.decode_sample(ctx, x, items, states, codec)
        altered = StoredCommand(stored.top, stored.tokens.clone(), stored.active, stored.masks.clone(),
                                stored.prefix.clone(), stored.split_index, stored.sonar)
        inactive = ~stored.active
        # the command itself is the tokens up to and including its terminating
        # STOP; everything after it is padding and may hold anything
        after_stop = torch.cumsum((stored.tokens == TOKEN_STOP).long(), 1) - (stored.tokens == TOKEN_STOP).long() > 0
        padding = inactive & (after_stop | (stored.top == TOP_SPLIT)[:, None])
        altered.tokens[padding] = (altered.tokens[padding] + 1) % N_TOKENS
        altered.prefix[inactive] = torch.randn_like(altered.prefix[inactive])
        altered.masks[inactive] = True
        lp2, _, _ = actor.decode_evaluate(ctx, x, altered)
        assert torch.allclose(lp, lp2, atol=1e-6)
        # a split row ignores its (empty) move payload entirely
        split_rows = stored.top == TOP_SPLIT
        if split_rows.any():
            assert not stored.active[split_rows].any()


def test_permuting_the_batch_cannot_change_one_actors_result(actor):
    items, states = _batch(3)
    x = SpatialInputs.from_prepared(items, codec)
    M, h = actor.initial_state(3, 17, 17)
    M = M + 0.3 * torch.randn_like(M)
    h = torch.randn_like(h)
    with torch.no_grad():
        ctx, M2, h2 = actor.encode_and_update(x, M, h)
        order = torch.tensor([2, 0, 1])
        ctx_p, M2_p, h2_p = actor.encode_and_update(x.select(order), M[order], h[order])
    assert torch.allclose(ctx.h[order], ctx_p.h, atol=1e-5)
    assert torch.allclose(M2[order], M2_p, atol=1e-5)


def test_the_actor_has_no_argument_for_global_state_or_critic_values():
    import inspect
    fields = set(SpatialInputs.__dataclass_fields__)
    assert not {"map_grid", "value", "reward", "truth", "critic"} & fields
    assert set(inspect.signature(SpatialActor.encode_and_update).parameters) == {"self", "x", "M_old", "h_old"}


def test_encode_and_update_is_pure_with_respect_to_caller_state(actor):
    items, states = _batch(2)
    x = SpatialInputs.from_prepared(items, codec)
    M, h = actor.initial_state(2, 17, 17)
    M0, h0 = M.clone(), h.clone()
    with torch.no_grad():
        ctx1, M1, h1 = actor.encode_and_update(x, M, h)
        ctx2, M2, h2 = actor.encode_and_update(x, M, h)
    assert torch.equal(M, M0) and torch.equal(h, h0), "inputs are untouched"
    assert torch.allclose(M1, M2) and torch.allclose(h1, h2), "a second call does not advance memory twice"


def test_unobserved_learned_cells_are_unchanged_by_a_write(actor):
    items, _ = _batch(1)
    x = SpatialInputs.from_prepared(items, codec)
    M, h = actor.initial_state(1, 17, 17)
    M = torch.randn_like(M)
    with torch.no_grad():
        _, M2, _ = actor.encode_and_update(x, M, h)
    changed = (M2 != M).any(dim=1)[0]
    visible = x.facts[0, 1] > 0
    assert torch.equal(changed | visible, visible), "only observed cells may change"
    assert changed.sum() == 49


def test_gradients_reach_writer_reader_gru_and_decoders():
    torch.manual_seed(3)
    actor = SpatialActor(SpatialActorConfig(sonar=True))
    items, states = _batch(3)
    x = SpatialInputs.from_prepared(items, codec)
    M, h = actor.initial_state(3, 17, 17)
    with torch.no_grad():
        ctx, _, _ = actor.encode_and_update(x, M, h)
        cmds, lp, stored = actor.decode_sample(ctx, x, items, states, codec)
        # make sure both branches are represented
        stored.top[0] = TOP_MOVE
        stored.top[1] = TOP_SPLIT
    ctx, M2, h2 = actor.encode_and_update(x, M, h)
    lp, ent, sent = actor.decode_evaluate(ctx, x, stored)
    (-lp.mean() - 0.01 * ent.mean() - 0.01 * sent.mean() + M2.pow(2).mean()).backward()
    for prefix in ("writer", "reader.gru", "reader.fine_in", "gru", "dec_cell", "tok_head",
                   "split_mlp", "top_head", "global_reads", "sonar_head"):
        params = [(n, p) for n, p in actor.named_parameters() if n.startswith(prefix)]
        assert params, prefix
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for _, p in params), f"no finite gradient reached {prefix}"


def test_save_and_load_reproduce_inference(tmp_path, actor):
    items, _ = _batch(2)
    x = SpatialInputs.from_prepared(items, codec)
    M, h = actor.initial_state(2, 17, 17)
    with torch.no_grad():
        ctx, M2, _ = actor.encode_and_update(x, M, h)
    torch.save(actor.state_dict(), tmp_path / "a.pt")
    clone = SpatialActor(SpatialActorConfig(sonar=True)).eval()
    clone.load_state_dict(torch.load(tmp_path / "a.pt"))
    with torch.no_grad():
        ctx2, M3, _ = clone.encode_and_update(x, M, h)
    assert torch.allclose(ctx.h, ctx2.h, atol=1e-6) and torch.allclose(M2, M3, atol=1e-6)


# -- lifecycle through the policy runtime ---------------------------------------

def test_policy_keeps_parent_state_across_a_split_and_starts_the_child_cold(actor):
    from dragons_env.protocol import parse_spawn, parse_turn
    w = h = 21
    spec = MapSpec(w, h, [(0, snake(8, 10, 8, "E", w, h)), (1, snake(15, 3, 3, "E", w, h))],
                   pearl_gap=(0, 0))
    m = NativeMatch(spec)
    policy = SpatialPolicy(actor, codec, match_id=7)
    for i in range(2):
        policy.on_spawn(i, parse_spawn(m.game.spawn_block(i)))
    for _ in range(2):
        m.play_round()
    m.game.begin_round()
    block = parse_turn(m.game.turn_block(0, m.ids), w, h)
    policy.decide(0, block)                       # whatever it chose, state advanced once
    M_before, h_before = policy.memory[0]
    st = policy.states[0]
    # force a split through the runtime's private state, exactly as the engine would see it
    st.pending = None
    st.commit_command(block, "split", split_n=3)
    m.game.take_turn(0, _dragons.Action.split(3))
    m._spawn()
    policy.on_spawn(2, parse_spawn(m.game.spawn_block(2)))
    assert torch.equal(policy.memory[0][0], M_before) and torch.equal(policy.memory[0][1], h_before)
    assert policy.memory[2][0].abs().sum() == 0 and policy.memory[2][1].abs().sum() == 0
    assert policy.states[2].facts.seen.sum() == 0 and policy.states[2].body is None
    assert policy.states[2].key.dragon_id == 2 and policy.states[2].key.match_id == 7
    policy.on_death(2, 3, "H")
    assert 2 not in policy.states and 2 not in policy.memory and 0 in policy.states


# -- export ---------------------------------------------------------------------

def test_scripted_encoder_and_decoder_match_the_eager_actor(tmp_path, actor):
    """The exported graphs are the same functions as the eager model; the
    runtime's prefix bookkeeping stays outside them."""
    from snake_rl.model.export import export_spatial
    items, states = _batch(2)
    x = SpatialInputs.from_prepared(items, codec)
    M, h = actor.initial_state(2, 17, 17)
    with torch.no_grad():
        ctx, M2, _ = actor.encode_and_update(x, M, h)
        cmds, lp, stored = actor.decode_sample(ctx, x, items, states, codec)
    out = export_spatial(actor, tmp_path)
    enc = torch.jit.load(str(out["encoder"]))
    dec = torch.jit.load(str(out["decoder"]))
    with torch.no_grad():
        h_s, M_s, _ = enc(x.local, x.facts, x.board_valid, x.visible_xy, x.visible_prev,
                          x.visible_age, x.status, x.history, x.history_mask, x.entities,
                          x.entity_mask, x.portal_pairs, x.pair_mask, x.portal_axis,
                          x.portal_unpaired, x.unpaired_mask, x.inbox, x.inbox_mask, x.head_xy, M, h)
        assert torch.allclose(h_s, ctx.h, atol=1e-5) and torch.allclose(M_s, M2, atol=1e-5)
        top, tok, dec_new, split = dec(ctx.h, x.split_legal, actor.move_init(ctx.h),
                                       torch.full((2,), 5, dtype=torch.long), stored.prefix[:, 0],
                                       stored.masks[:, 0] | ~stored.masks[:, 0].any(-1, keepdim=True),
                                       x.split_feats, x.split_mask)
        expect_top = actor.top_logits(ctx.h, x.split_legal)
        expect_tok, _ = actor.move_step(actor.move_init(ctx.h), ctx.h, torch.full((2,), 5, dtype=torch.long),
                                        stored.prefix[:, 0], stored.masks[:, 0] | ~stored.masks[:, 0].any(-1, keepdim=True))
        assert torch.allclose(top, expect_top, atol=1e-5) and torch.allclose(tok, expect_tok, atol=1e-5)
        assert torch.allclose(split, actor.split_logits(ctx.h, x.split_feats, x.split_mask), atol=1e-5)
    import json
    meta = json.loads(out["metadata"].read_text())
    assert meta["schema_version"] == "dragons-spatial-v1" and meta["max_move_steps"] == 8
