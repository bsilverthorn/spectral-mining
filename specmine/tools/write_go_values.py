import specmine

logger = specmine.get_logger(__name__)

def main():
    game_state = specmine.go.GameState([], specmine.go.BoardState())

    specmine.go.estimate_value(game_state, 16)

if __name__ == "__main__":
    specmine.script(main)

