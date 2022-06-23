import numpy as np
import torch


num_points = 21 * 21 - 1

point_index_to_xy = []

xpos_o = -10
ypos_o = -10
for i in range(21):
    for j in range(21):
        if i == 10 and j == 10:
            continue

        point_index_to_xy.append((xpos_o + i, ypos_o + j))
        pass
    pass
pass

point_index_to_xy = torch.IntTensor(point_index_to_xy)

reverse_dict = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
