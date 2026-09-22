import helper as unswbc
from helper import Direction, EdgeType
import random

ct: unswbc.Controller
game: unswbc.Game

# Seed so we get the same random generator every time.
random.seed(0)

def execute_turn() -> None:
    """Step onto the first open neighbouring tile."""
    here = ct.get_position()
    here_tile = ct.get_tile(here)

    directions = Direction.get_direction_list()
    random.shuffle(directions)

    for direction in directions:
        
        edge = here_tile.get_edge(direction).get_edge_type()
        if edge == EdgeType.KELP:
            continue

        ahead = ct.get_tile(here.add_dir(direction))
        if ahead.get_dragon() is not None:
            continue

        ct.output_log("Moving in", direction.value)
        ct.make_move(direction)
        return

    ct.make_move(Direction.NORTH)

def main() -> None:
    global ct, game
    ct, game = unswbc.init()

    while unswbc.update(ct, game):
        execute_turn()
        unswbc.end_turn()

if __name__ == "__main__":
    main()
