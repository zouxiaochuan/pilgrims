from kaggle_environments.envs.kore_fleets.helpers import Board
from kaggle_environments.envs.kore_fleets import kore_fleets


if __name__ == '__main__':
    board = kore_fleets.populate_board()
    print(board.observation)