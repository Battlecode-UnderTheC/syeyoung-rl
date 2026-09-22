"""The versioned schema for `dragons-spatial-v1`.

Every tensor the spatial-memory actor consumes is described here by name, so
there are no anonymous magic offsets anywhere else. The registry is plain data
and hashes to a string that checkpoints, exported models and the C++ side all
carry; a mismatch is an explicit error, never a silent reshape.

Sections of SPATIAL_MEMORY_RL_IMPLEMENTATION_SPEC.md this pins down: 4.3 (the
42 materialised factual channels), 4.5 (the status vector and the per-command
history features), 5.3 (the entity table) and 6.2 (the decoder prefix features).
"""

from __future__ import annotations

import hashlib
import json
from enum import IntEnum

SCHEMA_VERSION = "dragons-spatial-v1"
ACTION_CODEC_VERSION = "dragons-cmd-v1"

MAX_BOARD = 64
MAX_ROUNDS = 500
LAST_ROUND = MAX_ROUNDS - 1
VISION = 7
VISION_RADIUS = 3
TILES = VISION * VISION
HISTORY_TURNS = 32
MIN_SIZE = 2

DIRS = ("N", "E", "S", "W")
DIR_INDEX = {d: i for i, d in enumerate(DIRS)}
# (dx, dy) of one step; y grows downwards
DIR_OFFSET = ((0, -1), (1, 0), (0, 1), (-1, 0))
OPPOSITE = (2, 3, 0, 1)


class Occupant(IntEnum):
    """What the last observation of a tile showed standing on it."""
    EMPTY = 0
    SELF_HEAD = 1
    SELF_BODY = 2
    ALLY_HEAD = 3
    ALLY_BODY = 4
    ENEMY_HEAD = 5
    ENEMY_BODY = 6


N_OCCUPANT_KINDS = len(Occupant)


class EdgeType(IntEnum):
    EMPTY = 0
    KELP = 1
    PORTAL = 2


# -- 4.3 materialised factual feature map ------------------------------------

FACT_CHANNELS: tuple[str, ...] = (
    "seen_ever",                 # 0
    "visible_now",               # 1
    "age_since_seen",            # 2   clipped age / 500
    "pearl_last_seen",           # 3
    "countdown_soon",            # 4   1/(1+countdown) for a known active countdown
    "never_spawns",              # 5
    "occ_empty", "occ_self_head", "occ_self_body", "occ_ally_head",   # 6..9
    "occ_ally_body", "occ_enemy_head", "occ_enemy_body",             # 10..12
    "occ_facing_n", "occ_facing_e", "occ_facing_s", "occ_facing_w",  # 13..16
    "edge_known_n", "edge_known_e", "edge_known_s", "edge_known_w",  # 17..20
    "kelp_n", "kelp_e", "kelp_s", "kelp_w",                          # 21..24
    "portal_n", "portal_e", "portal_s", "portal_w",                  # 25..28
    "visit_log",                 # 29  log1p(visits)/log(501), clipped to 1
    "age_since_visit",           # 30
    "visited_ever",              # 31
    "prev_exists",               # 32
    "prev_pearl",                # 33
    "prev_countdown_soon",       # 34
    "seen_interval",             # 35  rounds between the two last observations / 500
    "occ_id_earlier",            # 36  recorded non-self occupant id < own id
    "occ_id_later",              # 37  recorded non-self occupant id > own id
    "own_body_known",            # 38
    "own_body_index",            # 39  index / max(length-1, 1)
    "own_tail_known",            # 40
    "own_head",                  # 41
)
N_FACT_CHANNELS = len(FACT_CHANNELS)
assert N_FACT_CHANNELS == 42


class F(IntEnum):
    """Channel indices into the factual map, by name."""
    SEEN = 0
    VISIBLE = 1
    AGE_SEEN = 2
    PEARL = 3
    COUNTDOWN = 4
    NEVER = 5
    OCC = 6                      # 7 kinds, Occupant order
    FACING = 13                  # 4
    EDGE_KNOWN = 17              # 4
    KELP = 21                    # 4
    PORTAL = 25                  # 4
    VISIT_LOG = 29
    AGE_VISIT = 30
    VISITED = 31
    PREV_EXISTS = 32
    PREV_PEARL = 33
    PREV_COUNTDOWN = 34
    INTERVAL = 35
    ID_EARLIER = 36
    ID_LATER = 37
    BODY_KNOWN = 38
    BODY_INDEX = 39
    TAIL_KNOWN = 40
    HEAD = 41


# -- 4.5 status vector -------------------------------------------------------

STATUS_FIELDS: tuple[str, ...] = (
    "length_linear",             # L / area
    "length_log",                # log1p(L) / log1p(area)
    "round",                     # round / 499
    "rounds_remaining",          # (499 - round) / 500
    "width", "height",           # / 64
    "unit_fraction",             # unit count / unit limit
    "heading_n", "heading_e", "heading_s", "heading_w",
    "split_legal",
    "first_own_turn",
    "own_turns",                 # completed own turns / 500
    "known_body_fraction",
    "tail_known",
    "head_sin_x", "head_cos_x", "head_sin_y", "head_cos_y",
    "prev_none", "prev_move", "prev_split",
    "prev_split_fraction",       # detached n / L before
    "prev_retained_fraction",    # (L - n) / L before
    "prev_steps",                # directions in the previous MOVE / K
    "prev_dir_n", "prev_dir_e", "prev_dir_s", "prev_dir_w",   # counts / K
    "prev_last_n", "prev_last_e", "prev_last_s", "prev_last_w",  # final token
    "prev_intermediates_known",
    "inbox_count",               # delivered words / 8 (raw count only; words stay in the adapter)
)
N_STATUS = len(STATUS_FIELDS)
S = IntEnum("S", {name.upper(): i for i, name in enumerate(STATUS_FIELDS)})


# -- 4.5 per-command history features ----------------------------------------

HISTORY_FIELDS: tuple[str, ...] = (
    "is_move", "is_split",
    "old_sin_x", "old_cos_x", "old_sin_y", "old_cos_y",
    "new_sin_x", "new_cos_x", "new_sin_y", "new_cos_y",
    "old_heading_n", "old_heading_e", "old_heading_s", "old_heading_w",
    "new_heading_n", "new_heading_e", "new_heading_s", "new_heading_w",
    "round",                     # / 500
    "rounds_ago",                # (current - round) / 500
    "old_length", "new_length",  # / area
    "length_delta",              # (new - old) / K
    "steps",                     # / K
    "dir_n", "dir_e", "dir_s", "dir_w",   # counts / K
    "split_fraction",            # n / L_old
    "intermediates_known",
)
N_HISTORY_FEATURES = len(HISTORY_FIELDS)
HF = IntEnum("HF", {name.upper(): i for i, name in enumerate(HISTORY_FIELDS)})


# -- 5.3 entity table --------------------------------------------------------

ENTITY_FIELDS: tuple[str, ...] = (
    "count",                     # visible segments / 49 (a lower bound on length)
    "count_log",                 # log1p(count) / log(50)
    "head_present",
    "head_facing_n", "head_facing_e", "head_facing_s", "head_facing_w",
    "head_dx", "head_dy",        # window-relative, / 3; zero without a head
    "mean_dx", "mean_dy",        # / 3
    "min_dist",                  # nearest segment, Chebyshev / 3
    "ally", "enemy",
    "id_earlier", "id_later",
)
N_ENTITY_FEATURES = len(ENTITY_FIELDS)
EF = IntEnum("EF", {name.upper(): i for i, name in enumerate(ENTITY_FIELDS)})
MAX_ENTITIES = TILES   # at most one occupied tile per visible segment


# -- 6.2 decoder prefix features ---------------------------------------------

PREFIX_FIELDS: tuple[str, ...] = (
    "step",                      # tokens emitted so far / K
    "position_known",
    "alive_known",               # no fatal collision is known along the prefix
    "fatal_known",
    "length_known",
    "length",                    # known length / area, 0 if unknown
    "can_pay",                   # known length > 2
    "resource_uncertain",
    "prev_n", "prev_e", "prev_s", "prev_w",  # last direction token, if any
    "ahead_empty_n", "ahead_empty_e", "ahead_empty_s", "ahead_empty_w",
    "ahead_kelp_n", "ahead_kelp_e", "ahead_kelp_s", "ahead_kelp_w",
    "ahead_portal_n", "ahead_portal_e", "ahead_portal_s", "ahead_portal_w",
    "ahead_unknown_n", "ahead_unknown_e", "ahead_unknown_s", "ahead_unknown_w",
    "occ_self_n", "occ_self_e", "occ_self_s", "occ_self_w",
    "occ_other_n", "occ_other_e", "occ_other_s", "occ_other_w",
    "occ_head_n", "occ_head_e", "occ_head_s", "occ_head_w",
    "occ_unknown_n", "occ_unknown_e", "occ_unknown_s", "occ_unknown_w",
    "pearl_n", "pearl_e", "pearl_s", "pearl_w",
    "pearls_eaten",              # known pearls along the prefix / K
)
N_PREFIX_FEATURES = len(PREFIX_FIELDS)
PF = IntEnum("PF", {name.upper(): i for i, name in enumerate(PREFIX_FIELDS)})

# move decoder tokens
TOKEN_N, TOKEN_E, TOKEN_S, TOKEN_W, TOKEN_STOP = 0, 1, 2, 3, 4
N_TOKENS = 5
TOP_MOVE, TOP_SPLIT = 0, 1

# sonar experiment (6.5)
SONAR_SILENT = 0
SONAR_CODES = 16
N_SONAR_CHOICES = SONAR_CODES + 1


def registry() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "action_codec_version": ACTION_CODEC_VERSION,
        "max_board": MAX_BOARD,
        "max_rounds": MAX_ROUNDS,
        "vision": VISION,
        "history_turns": HISTORY_TURNS,
        "occupant_kinds": [k.name for k in Occupant],
        "fact_channels": list(FACT_CHANNELS),
        "status_fields": list(STATUS_FIELDS),
        "history_fields": list(HISTORY_FIELDS),
        "entity_fields": list(ENTITY_FIELDS),
        "prefix_fields": list(PREFIX_FIELDS),
        "move_tokens": ["N", "E", "S", "W", "STOP"],
        "sonar_choices": N_SONAR_CHOICES,
    }


def schema_hash() -> str:
    text = json.dumps(registry(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def check_schema(version: str, digest: str | None = None) -> None:
    """Raise rather than let an old checkpoint reshape itself into the new schema."""
    if version != SCHEMA_VERSION:
        raise ValueError(f"schema mismatch: checkpoint is {version!r}, "
                         f"runtime is {SCHEMA_VERSION!r}")
    if digest is not None and digest != schema_hash():
        raise ValueError(f"schema hash mismatch: checkpoint {digest}, "
                         f"runtime {schema_hash()}")
