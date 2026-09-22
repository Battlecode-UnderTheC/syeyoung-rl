"""Turn a `TurnBlock` into tensors.

Everything here comes from what the dragon can actually see: its 7x7 window,
its own length/heading/team unit count, the round number and the messages that
reached it. Nothing global.

The window is optionally rotated into the dragon's frame so that "forward" is
always up; the directional channel groups are rolled to match. With relative
actions this makes the policy heading invariant.
"""

from __future__ import annotations

import numpy as np

from dragons_env.protocol import DIR_INDEX, KELP, MIN_SIZE, TurnBlock, VISION

# channel layout
C_PEARL = 0
C_PEARL_SOON = 1
C_PEARL_NEVER = 2
C_OWN_BODY = 3
C_OWN_HEAD = 4
C_ALLY_HEAD = 5
C_ALLY_BODY = 6
C_ENEMY_HEAD = 7
C_ENEMY_BODY = 8
C_FACING = 9        # 4, in N E S W order before rotation
C_KELP = 13         # 4
C_PORTAL = 17       # 4
N_CHANNELS = 21

DIRECTIONAL = (C_FACING, C_KELP, C_PORTAL)
# Section 3 of the spec lists "own length, own heading/speed/sprint resource/
# split cooldown, own legal status and game time" as the scalar vector. Heading
# was missing: with ego rotation on it is implicit in the frame, and with it off
# the network had no idea which way it faced. This game has no sprint resource
# and no split cooldown, so those slots are the affordable sprint length and
# whether a split is legal.
N_SCALARS = 12

class ObservationEncoder:
    def __init__(self, max_messages: int = 8, ego_rotate: bool = True,
                 max_sprint: int = 3):
        self.max_messages = max_messages
        self.ego_rotate = ego_rotate
        self.max_sprint = max_sprint
        self.n_channels = N_CHANNELS
        self.n_scalars = N_SCALARS

    def grid(self, block: TurnBlock, dragon_id: int, team: str) -> np.ndarray:
        out = np.zeros((N_CHANNELS, VISION, VISION), dtype=np.float32)
        flat = out.reshape(N_CHANNELS, -1)

        timers = block.pearl_time.astype(np.float32)
        flat[C_PEARL] = block.pearl
        never = timers < 0
        flat[C_PEARL_NEVER] = never
        soon = np.where(never, 0.0, 1.0 / (1.0 + np.maximum(timers, 0.0)))
        flat[C_PEARL_SOON] = soon

        for part in block.parts:
            i = block.window_index(part.x, part.y)
            if i is None:
                continue
            if part.dragon_id == dragon_id:
                flat[C_OWN_HEAD if part.is_head else C_OWN_BODY, i] = 1.0
            elif part.team == team:
                flat[C_ALLY_HEAD if part.is_head else C_ALLY_BODY, i] = 1.0
            else:
                flat[C_ENEMY_HEAD if part.is_head else C_ENEMY_BODY, i] = 1.0
            flat[C_FACING + DIR_INDEX[part.facing], i] = 1.0

        flat[C_KELP:C_KELP + 4] = block.kelp.T
        flat[C_PORTAL:C_PORTAL + 4] = block.portal.T

        if self.ego_rotate:
            k = DIR_INDEX[block.facing]
            if k:
                out = np.rot90(out, k=k, axes=(1, 2)).copy()
                for base in DIRECTIONAL:
                    out[base:base + 4] = np.roll(out[base:base + 4], -k, axis=0)
        return out

    def scalars(self, block: TurnBlock, unit_limit: int) -> np.ndarray:
        max_sprint = min(self.max_sprint, max(block.length - MIN_SIZE + 1, 1))
        return np.array([
            block.length / 20.0,
            np.log1p(block.length) / 4.0,
            block.unit_count / 8.0,
            block.round / 500.0,
            max_sprint / max(self.max_sprint, 1),
            1.0 if block.length - MIN_SIZE >= MIN_SIZE and block.unit_count < unit_limit else 0.0,
            len(block.messages) / 4.0,
            1.0 if block.unit_count >= unit_limit else 0.0,
            *(1.0 if block.facing == d else 0.0 for d in ("N", "E", "S", "W")),
        ], dtype=np.float32)

    def messages(self, block: TurnBlock) -> tuple[np.ndarray, np.ndarray]:
        """Delivered messages as bytes, padded to `max_messages`."""
        out = np.zeros((self.max_messages, 4), dtype=np.uint8)
        mask = np.zeros(self.max_messages, dtype=bool)
        for i, value in enumerate(block.messages[: self.max_messages]):
            out[i] = np.frombuffer(int(value).to_bytes(4, "little"), dtype=np.uint8)
            mask[i] = True
        return out, mask
