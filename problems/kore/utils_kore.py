
from kaggle_environments.envs.kore_fleets.helpers import Board
import numpy as np
import torch


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


def stack_value(v1, v2, device='cpu'):
    '''
    Parameters:
        v1:
        v2:

    Returns:
        [n, 2]
    '''
    if isinstance(v1, torch.Tensor):
        v2_ = torch.zeros_like(v1)
        v2_.fill_(v2)
        v2 = v2_
        pass
    else:
        v1_ = torch.zeros_like(v2)
        v1_.fill_(v1)
        v1 = v1_
        pass

    return torch.stack([v1, v2], dim=1)
