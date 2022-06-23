
from kaggle_environments.envs.kore_fleets.helpers import Board
import numpy as np


def get_board_kore(board: Board):
    size = board.configuration.size
    size2 = size * size
    kore = np.zeros(size2, dtype='float32')
    for i, (poit, cell) in enumerate(sorted(
            board.cells.items(), 
            key=lambda item: (item[0].y, item[0].x))):
        kore[i] = cell.kore
        pass

    return kore
    pass


def add_mirror_xy(xy, size=21):
    xy[:, 0] += 20
    xy[:, 1] -= 20

    return xy
    pass


def stack_value(v1, v2):
    '''
    Parameters:
        v1:
        v2:

    Returns:
        [n, 2]
    '''
    if isinstance(v1, np.ndarray):
        v2_ = np.zeros_like(v1)
        v2_.fill(v2)
        v2 = v2_
        pass
    else:
        v1_ = np.zeros_like(v2)
        v1_.fill(v1)
        v1 = v1_
        pass

    return np.stack([v1, v2], axis=1)
