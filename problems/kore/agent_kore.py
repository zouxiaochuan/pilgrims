import kaggle_environments
import torch
import torch.nn.functional as F
from kaggle_environments.envs.kore_fleets.helpers import (
    Board, ShipyardAction, ShipyardActionType)
import numpy as np
from . import utils_kore
import math
from .env_kore import KoreEnv
from . import const_kore
from agent import DecisionResult, Agent
from typing import Dict, List
from .model_kore import KoreNetworks


class ShipyardActionEx(object):
    def __init__(self, raw_action: ShipyardAction, latent_variables: Dict):
        self.raw_action = raw_action
        self.latent_variables = latent_variables
        pass
    pass


class KoreAgent(Agent):

    def __init__(self, config):
        self.size = config['map_size']
        self.size2 = self.size * self.size
        self.num_channels_fleet = 1 + 1 + 4  # kore, ship_count, direction
        self.num_channels_shipyard = 1  # ship_count
        self.num_channels = 1 + self.num_channels_fleet * 2 + \
            self.num_channels_shipyard * 2
        self.num_feat_player = 4  # kore, total kore, total ships, total shipyards
        self.num_feat = self.num_feat_player * 3  # first player, second, the difference
        config['agent']['num_input_map_channels'] = self.num_channels
        config['agent']['num_input_vec_channels'] = self.num_feat
        config['agent']['size2'] = self.size2
        self.point_temperature = config['agent']['point_temperature']

        self.num_candidate_points = config['agent']['num_candidate_points']
        self.num_launch_points = self.size2 - 1
        self.model = KoreNetworks(config['agent'])
        self.need_grad = False
        env = KoreEnv(config)
        self.spawn_cost = env.spawn_cost

        pass

    def train(self):
        self.need_grad = True
        pass

    def eval(self):
        self.need_grad = False
        pass

    def vectorize_env(self, obs):
        board: Board = obs

        map = torch.zeros((self.num_channels, self.size2))
        vec = torch.zeros(self.num_feat)

        map[0, :] = torch.from_numpy(utils_kore.get_board_kore(board))
        current_player_id = board.current_player_id
        my_shipyard_index = []
        my_shipyard_max_spawn = []
        my_shipyard_ship_count = []
        my_shipyard_id = []

        for player_id, player in board.players.items():
            player_absid = abs(current_player_id - player_id)
            player_feat_start = player_absid * self.num_feat_player

            vec[player_feat_start + 0] = player.kore

        for fleet_id, fleet in board.fleets.items():
            player_absid = abs(current_player_id - fleet.player_id)
            fleet_channel = player_absid * self.num_channels_fleet + 1
            player_feat_start = player_absid * self.num_feat_player

            fleet_index = fleet.position.to_index(self.size)
            map[fleet_channel, fleet_index] = fleet.kore
            map[fleet_channel + 1, fleet_index] = fleet.ship_count
            map[fleet_channel + 2 + fleet.direction.to_index(), fleet_index] = 1

            vec[player_feat_start + 1] += fleet.kore
            vec[player_feat_start + 2] += fleet.ship_count
            pass

        for shipyard_id, shipyard in board.shipyards.items():
            player_absid = abs(current_player_id - shipyard.player_id)
            shipyard_channel = player_absid * self.num_channels_shipyard + (1 + self.num_channels_fleet * 2)
            player_feat_start = player_absid * self.num_feat_player
            shipyard_index = shipyard.position.to_index(self.size)
            map[shipyard_channel, shipyard_index] = shipyard.ship_count
            vec[player_feat_start + 2] += shipyard.ship_count
            vec[player_feat_start + 3] += 1

            if player_absid == 0:
                my_shipyard_index.append(shipyard_index)
                my_shipyard_max_spawn.append(shipyard.max_spawn)
                my_shipyard_ship_count.append(shipyard.ship_count)
                my_shipyard_id.append(shipyard_id)
                pass
            pass

        kore_map = map[0, :].reshape(self.size, self.size)
        map = torch.transpose(map, 0, 1)

        return {
            'map': map, 'vec': vec,
            'my_shipyard': {
                'index': torch.LongTensor(my_shipyard_index),
                'max_spawn': torch.FloatTensor(my_shipyard_max_spawn),
                'ship_count': torch.FloatTensor(my_shipyard_ship_count),
                'id': my_shipyard_id
            },
            'kore_map': kore_map
        }
        pass

    def decide(self, obs_list):
        if self.need_grad:
            result = self.decide_(obs_list)
        else:
            with torch.no_grad():
                result = self.decide_(obs_list)
                pass
            pass
        return result

    def decide_(self, obs_list):
        obs_vec_list = [self.vectorize_env(obs) for obs in obs_list]
        obs_vec_batch = self.collate_obs_vec(obs_vec_list)

        # max_plan_length = np.floor(2 * np.log(obs_vec['my_shipyard_ship_count'])) + 1

        embs = self.model.encode_obs(obs_vec_batch)
        # map_emb: [B, S, C], my_shipyard.index: [B, SY]
        batch_size = embs['map_emb'].shape[0]
        max_shipyard = obs_vec_batch['my_shipyard']['mask'].shape[1]
        batch_index = torch.tile(
            torch.arange(batch_size), [max_shipyard, 1]).T

        shipyard_embs = embs['map_emb'][batch_index, obs_vec_batch['my_shipyard']['index']]
        # shipyard_embs: [B, SY, C]

        v = self.model.forward_v(embs)

        score_fist_layer = self.model.forward_first_layer(shipyard_embs)
        score_spawn_branch = score_fist_layer[..., 0]
        score_launch_branch = score_fist_layer[..., 1]
        score_convert_branch = score_fist_layer[..., 2]
        # [SY, 3]

        # if spawn
        action_spawn, score_spawn, mask_spawn = self.decide_spawn(score_spawn_branch, obs_vec_batch)

        # if launch, what's the plan
        action_launch, score_launch, mask_launch = self.decide_launch(
            score_launch_branch, shipyard_embs, obs_vec_batch)

        # if create new shipyard, what's the plan
        action_convert, score_convert, mask_convert = self.decide_convert(
            score_convert_branch, shipyard_embs, obs_vec_batch)

        actions = [
            [
                action_shipyard[0] + action_shipyard[1] + action_shipyard[2]
                for action_shipyard in zip(
                    *action_map)
            ] for action_map in zip(
                action_spawn, action_launch, action_convert)
        ]

        score, score_mask = self.concat_sparse(
            (score_spawn, score_launch, score_convert),
            (mask_spawn, mask_launch, mask_convert)
        )

        dresult = DecisionResult(
            actions, score, score_mask, obs_vec_batch['my_shipyard']['id']
        )

        return dresult, v.flatten()

    def concat_sparse(self, datas, masks):
        '''
        Parameters:
            datas: tuple of tensors with shape: [B, SY, N, P]
            masks: same shape of datas
        '''
        max_len = max([data.shape[-1] for data in datas])

        datas_new = []
        masks_new = []
        for i, (data, mask) in enumerate(zip(datas, masks)):
            size_new = data.shape[:-1] + (max_len,)
            data_new = torch.zeros(
                size_new, dtype=data.dtype,
                device=data.device)
            mask_new = torch.zeros(
                size_new, dtype=mask.dtype,
                device=mask.device
            )
            data_new[..., : data.shape[-1]] = data
            mask_new[..., : mask.shape[-1]] = mask
            datas_new.append(data_new)
            masks_new.append(mask_new)
            pass

        return torch.cat(datas_new, dim=-2), torch.cat(masks_new, dim=-2)
        pass

    def collate_seq(self, feat_list, return_mask=False):
        '''
        Parameters:
            feat_list: list of tensors with different shapes
            return_mask: if return mask
        '''

        batch_size = len(feat_list)
        feat_max_len = np.max([feat.shape[0] for feat in feat_list])
        # feat_dim = feat_list[0].shape[1]
        feat = torch.zeros(
            # (batch_size, feat_max_len, feat_dim),
            (batch_size, feat_max_len),
            dtype=feat_list[0].dtype)

        if return_mask:
            mask = torch.zeros((batch_size, feat_max_len), dtype=torch.float32)
            pass

        for i, ifeat in enumerate(feat_list):
            size = ifeat.shape[0]
            feat[i, :size] = ifeat
            if return_mask:
                mask[i, :size] = 1
                pass
            pass
        if return_mask:
            return feat, mask
        else:
            return feat
            pass
        pass

    def collate_seq_values(self, values):
        result = dict()
        mask = None
        sizes = None

        for key, value in values[0].items():
            if isinstance(value, list):
                result[key] = [v[key] for v in values]
            elif isinstance(value, torch.Tensor):
                v_list = [v[key] for v in values]
                if mask is None:
                    v, mask = self.collate_seq(v_list, return_mask=True)
                    sizes = torch.sum(mask, dim=1).int()
                    result['mask'] = mask
                    result['sizes'] = sizes
                    pass
                else:
                    v = self.collate_seq(v_list)
                    pass
                result[key] = v
                pass
            else:
                raise RuntimeError('unknown type')
            pass
        return result

    def collate_obs_vec(self, obs_vec_list):
        result = dict()
        for key, value in obs_vec_list[0].items():
            value_list = [obs_vec[key] for obs_vec in obs_vec_list]
            if isinstance(value, dict):
                # length is variable
                result[key] = self.collate_seq_values(value_list)
                pass
            else:
                result[key] = torch.stack(value_list)
                pass
            pass

        return result

    def decide_spawn(self, score_spawn_branch, obs_vec_batch):
        '''
        Parameters:
            score_spawn_branch: [B, SY]

        Returns:
            action_spawn: [B, SY, 1]
            score_spawn: [B, SY, 1, 1]
            mask_spawn: [B, SY, 1, 1]
        '''
        max_spawn = obs_vec_batch['my_shipyard']['max_spawn']
        shipyard_mask = obs_vec_batch['my_shipyard']['mask']
        # shipyard_mask: [B, SY]
        kore_own = obs_vec_batch['vec'][:, 0]
        max_spawn2 = kore_own / self.spawn_cost
        action_spawn = [
            [
                [
                    ShipyardActionEx(
                        ShipyardAction(
                            ShipyardActionType.SPAWN,
                            int(min(max_spawn[b, i], max_spawn2[b])),
                            None),
                        latent_variables={})
                ] if shipyard_mask[b, i] > 0 else [None] for i in range(len(shipyard_mask[b]))
            ]
            for b in range(len(shipyard_mask))
        ]

        score_spawn = score_spawn_branch.unsqueeze(-1)
        mask_spawn = shipyard_mask.unsqueeze(-1).unsqueeze(-1)

        return action_spawn, score_spawn.unsqueeze(-1), mask_spawn

    def point_index2shift(self, point_index):
        return const_kore.point_index_to_xy[point_index]
        pass

    def map_index2xy(self, index, size=21):
        x = index % size
        y = torch.div(index, size, rounding_mode='trunc')
        # y = - y
        xy = torch.stack([x, y], axis=-1)

        return xy

    def decide_launch(self, score_launch_branch, shipyard_embs, obs_vec_batch):
        '''
        Parameters:
            score_launch_branch: [B, SY]
        '''
        B, SY = score_launch_branch.shape
        batch_mask = torch.zeros((B, SY))
        ship_counts = obs_vec_batch['my_shipyard']['ship_count']

        batch_mask[ship_counts >= 21] = 1
        shipyard_mask = obs_vec_batch['my_shipyard']['mask']
        shipyard_action_mask = shipyard_mask * batch_mask

        if torch.sum(shipyard_action_mask) > 0:
            index_score = self.model.forward_launch_point(shipyard_embs)
            index_proba = torch.softmax(index_score / self.point_temperature, dim=-1)
            # index_proba: [B, SY, P(num_launch_points)]
            sampled_index = torch.multinomial(
                index_proba.reshape(-1, self.num_launch_points), num_samples=self.num_candidate_points, replacement=False)
            sampled_index = sampled_index.reshape(B, SY, self.num_candidate_points)

            # sampled_index: [B, SY, num_candidate_points]
            sampled_score = torch.gather(index_score, -1, sampled_index)
            # sample probability

            shift_xy = self.point_index2shift(sampled_index)
            # shift_xy: [B, SY, num_candidate_points, 2]

            shipyard_xy = self.map_index2xy(obs_vec_batch['my_shipyard']['index'])
            # shipyard_xy: [B, SY, 2]

            batch_plans = self.generate_launch_plan(
                shift_xy, shipyard_xy, shipyard_action_mask, shipyard_mask,
                obs_vec_batch['kore_map'])

            # batch_plans: list of shape [B, num_shipyards, num_candidates]
        else:
            batch_plans = [
                [
                    []
                    for _ in range(len(shipyard_mask[ib]))
                ] for ib in range(B)
            ]
            sampled_score = torch.zeros((B, SY, 0))

            pass

        action_launch = [
            [
                [
                    ShipyardActionEx(
                        ShipyardAction(ShipyardActionType.LAUNCH, int(ship_counts[ibatch, ishipyard]), flight_plan),
                        latent_variables={}
                    ) if flight_plan is not None else None
                    for flight_plan in shipyard_plan
                ] if shipyard_action_mask[ibatch, ishipyard] > 0 else
                [
                    None for _ in shipyard_plan
                ]
                for ishipyard, shipyard_plan in enumerate(plans)
            ] for ibatch, plans in enumerate(batch_plans)
        ]

        if sampled_score.shape[-1] == 0:
            score_launch = sampled_score.unsqueeze(-1)
            return action_launch, score_launch, torch.zeros_like(score_launch)
        else:
            score_launch = torch.cat(
                (
                    sampled_score[..., None],
                    torch.tile(
                        score_launch_branch[..., None, None],
                        [1, 1, sampled_score.shape[-1], 1])
                ),
                dim=-1)
            score_mask = torch.zeros_like(score_launch)
            score_mask[shipyard_action_mask.bool(), :, :] = 1

            return action_launch, score_launch, score_mask

    def decide_convert(self, score_convert_branch, shipyard_embs, obs_vec_batch):
        B, SY = score_convert_branch.shape
        batch_mask = torch.zeros((B, SY))
        ship_counts = obs_vec_batch['my_shipyard']['ship_count']

        batch_mask[ship_counts >= 50] = 1
        shipyard_mask = obs_vec_batch['my_shipyard']['mask']
        shipyard_action_mask = shipyard_mask * batch_mask

        if torch.sum(shipyard_action_mask) > 0:
            index_score = self.model.forward_convert_point(shipyard_embs)
            index_proba = torch.softmax(index_score / self.point_temperature, dim=-1)
            # index_proba: [B, SY, P(num_convert_points)]

            sampled_index = torch.multinomial(
                index_proba.reshape(-1, self.num_launch_points), num_samples=self.num_candidate_points, replacement=False)
            sampled_index = sampled_index.reshape(B, SY, self.num_candidate_points)

            # sampled_index: [B, SY, num_candidate_points]
            sampled_score = torch.gather(index_score, -1, sampled_index)
            # sample probability

            shift_xy = self.point_index2shift(sampled_index)
            # shift_xy: [B, SY, num_candidate_points, 2]

            shipyard_xy = self.map_index2xy(obs_vec_batch['my_shipyard']['index'])
            # shipyard_xy: [B, SY, 2]

            batch_plans = self.generate_launch_plan(
                shift_xy, shipyard_xy, shipyard_action_mask, shipyard_mask,
                obs_vec_batch['kore_map'],
                need_back=False, remove_last_number=False)
            pass
        else:
            batch_plans = [
                [
                    []
                    for _ in range(len(shipyard_mask[ib]))
                ] for ib in range(B)
            ]
            sampled_score = torch.zeros((B, SY, 0))
            pass

        action_convert = [
            [
                [
                    ShipyardActionEx(
                        ShipyardAction(
                            ShipyardActionType.LAUNCH, int(ship_counts[ibatch, ishipyard]),
                            flight_plan + 'C'
                        ),
                        latent_variables={}
                    ) if flight_plan is not None else None
                    for flight_plan in shipyard_plan
                ] if shipyard_action_mask[ibatch, ishipyard] > 0 else
                [
                    None for _ in shipyard_plan
                ] for ishipyard, shipyard_plan in enumerate(plans)
            ] for ibatch, plans in enumerate(batch_plans)
        ]

        if sampled_score.shape[-1] == 0:
            score_convert = sampled_score.unsqueeze(-1)
            return action_convert, score_convert, torch.zeros_like(score_convert)
        else:
            score_convert = torch.cat(
                (
                    sampled_score[..., None],
                    torch.tile(
                        score_convert_branch[..., None, None],
                        [1, 1, sampled_score.shape[-1], 1])
                ),
                dim=-1)
            score_mask = torch.zeros_like(score_convert)
            score_mask[shipyard_action_mask.bool(), :, :] = 1

            return action_convert, score_convert, score_mask

    def generate_launch_plan(
            self, shift_xy, shipyard_xy, shipyard_action_mask, shipyard_mask, kore_map,
            need_back=True, remove_last_number=True):
        '''

        Parameters:
            shift_xy: [B, SY, P, 2]
            shipyard_xy: [B, SY, 2]
            shipyard_mask: [B, SY]
            kore_map: [B, size, size]
            need_back: bool, if the fleet need back to shipyard
            remove_last_number: bool, if the plan should remove the last number
        Returns:
            plans: list of shape [B, num_shipyards, num_candidates]
        '''
        batch_plans = []
        if need_back:
            nearest_shipyard, nearest_shipyard_shift = self.find_nearest_back_shipyard(
                shift_xy, shipyard_xy, shipyard_mask)
            # nearest_shipyard: [B, SY, P], nearest_shipyard_shift: [B, SY, P, 2]
            pass

        for ibatch in range(shift_xy.shape[0]):
            kore_map_i = kore_map[ibatch]
            plans = []

            for ishipyard in range(shift_xy.shape[1]):
                shipyard_plan = []
                if shipyard_action_mask[ibatch, ishipyard] > 0:
                    shipyard_xy_i = shipyard_xy[ibatch, ishipyard]
                    for ishift in range(shift_xy.shape[2]):
                        shift_xy_i = shift_xy[ibatch, ishipyard, ishift]
                        plan_forward = self.find_plan(kore_map_i, shipyard_xy_i, shift_xy_i)

                        if need_back:
                            nearest_shipyard_i = nearest_shipyard[ibatch, ishipyard, ishift]
                            nearest_shipyard_xy_i = shipyard_xy[ibatch, nearest_shipyard_i]
                            nearest_shipyard_shift_i = nearest_shipyard_shift[ibatch, ishipyard, ishift]
                            plan_backward = self.find_plan(kore_map_i, nearest_shipyard_xy_i, nearest_shipyard_shift_i)
                            plan_backward = self.reverse_plan(plan_backward)
                            pass
                        else:
                            plan_backward = ''
                            pass

                        plan = plan_forward + plan_backward

                        if len(plan) == 0:
                            raise RuntimeError('plan is empty')

                        if remove_last_number:
                            if str.isdigit(plan[-1]):
                                plan = plan[:-1]
                                pass
                            pass

                        shipyard_plan.append(plan)
                        pass
                    pass
                else:
                    shipyard_plan = [None for _ in range(shift_xy.shape[2])]
                    pass

                plans.append(shipyard_plan)
                pass
            batch_plans.append(plans)
            pass
        return batch_plans
        pass

    def reverse_plan(self, plan):
        result = []
        for i in range(len(plan) - 1, -1, -1):
            ch = plan[i]
            if str.isdigit(ch):
                continue

            d = const_kore.reverse_dict[ch]
            n = ''
            if i < (len(plan) - 1):
                ch_post = plan[i + 1]
                if str.isdigit(ch_post):
                    n = ch_post
                    pass
                pass
            result.append(d)
            result.append(n)
            pass
        return ''.join(result)

    def calc_pair_diff(self, x1, x2, size=21):
        '''
        Parameters:
            x1: [B, n]
            x2: [B, m]
        '''
        pair_diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
        # pair_diff: [B, n, m]

        size_half = size // 2

        mask_overflow = pair_diff > size_half
        mask_underflow = pair_diff < -size_half

        pair_diff[mask_overflow] -= size
        pair_diff[mask_underflow] += size

        return pair_diff

    def find_nearest_back_shipyard(self, shift_xy, shipyard_xy: torch.Tensor, shipyard_mask, size=21):
        '''
        Parameters:
            shift_xy: [B, SY, P, 2]
            shipyard_index_xy: [B, SY, 2]
            shipyard_mask: [B, SY]
        Returns:
            nearest_idx: [B, SY, P], nearest shipyard idx
            nearest_shift: [B, SY, P, 2], shift from nearest shipyard to launch point
        '''
        point_xy = shipyard_xy.unsqueeze(-2) + shift_xy
        point_xy %= size
        B = point_xy.shape[0]
        SY = point_xy.shape[1]

        point_xy = torch.reshape(point_xy, (B, -1, 2))

        xdiff = self.calc_pair_diff(point_xy[:, :, 0], shipyard_xy[:, :, 0])
        ydiff = self.calc_pair_diff(point_xy[:, :, 1], shipyard_xy[:, :, 1])
        # xdiff, ydiff: [B, SY*P, SY]

        dist = torch.abs(xdiff) + torch.abs(ydiff)
        dist = dist.float()

        dist += (1 - shipyard_mask[:, None, :]) * 999

        nearest_idx = torch.argmin(dist, axis=-1, keepdim=True)
        # nearest_idx: [B, SY*P, 1]

        nearest_shift = torch.concat(
            (
                torch.gather(xdiff, -1, nearest_idx),
                torch.gather(ydiff, -1, nearest_idx),
            ),
            axis=-1
        )
        # nearest_shift: [B, SY*P, 2]

        nearest_idx = torch.reshape(nearest_idx, (B, SY, -1))
        nearest_shift = torch.reshape(nearest_shift, (B, SY, -1, 2))
        return nearest_idx, nearest_shift

    def shift2plan(self, direction: str, shift) -> str:
        '''

        Returns:
            plan: str
        '''

        shift = abs(shift)

        if shift == 0:
            plan = ''
        elif shift == 1:
            plan = direction
        else:
            plan = direction + str(shift - 1)
            pass

        return plan

    def find_plan(self, kore_map, start_xy, shift_xy):
        end_xy = start_xy + shift_xy

        xstep = math.copysign(1, shift_xy[0])
        ystep = math.copysign(1, shift_xy[1])

        slice1_a = utils_kore.stack_value(
            np.arange(start_xy[0] + xstep, end_xy[0] + xstep, xstep),
            start_xy[1])
        slice2_a = utils_kore.stack_value(
            end_xy[0], np.arange(start_xy[1] + ystep, end_xy[1] + ystep, ystep)
        )
        slice1_b = utils_kore.stack_value(
            start_xy[0], np.arange(start_xy[1] + ystep, end_xy[1] + ystep, ystep)
        )
        slice2_b = utils_kore.stack_value(
            np.arange(start_xy[0] + xstep, end_xy[0] + xstep, xstep),
            end_xy[1]
        )

        x_direction = 'E' if shift_xy[0] > 0 else 'W'
        y_direction = 'S' if shift_xy[1] > 0 else 'N'

        plan_a = self.shift2plan(x_direction, int(shift_xy[0])) + self.shift2plan(y_direction, int(shift_xy[1]))
        plan_b = self.shift2plan(y_direction, int(shift_xy[1])) + self.shift2plan(x_direction, int(shift_xy[0]))

        kore_a = self.get_kore(kore_map, slice1_a, slice2_a)
        kore_b = self.get_kore(kore_map, slice1_b, slice2_b)

        if kore_a > kore_b:
            return plan_a
        else:
            return plan_b
        pass

    def get_kore(self, kore_map, slice1, slice2, size=21):
        slice1 %= size
        slice2 %= size

        # y is the first dimention and x is the second
        return torch.sum(kore_map[slice1[:, 1], slice1[:, 0]]) + \
            torch.sum(kore_map[slice2[:, 1], slice2[:, 0]])

    def calculate_loss(self, obs_list, rewards):
        dresult, v = self.decide(obs_list)
        loss_v = F.binary_cross_entropy_with_logits(v, rewards)

        correct_scores = self.correct_action(dresult, obs_list)
        correct_scores = correct_scores.unsqueeze(-1)
        correct_scores = torch.broadcast_to(correct_scores, dresult.scores.shape)
        loss_actions = F.binary_cross_entropy_with_logits(
            dresult.scores, torch.sigmoid(correct_scores))
        loss_actions = torch.mean(torch.masked_select(loss_actions, dresult.mask>0))

        return loss_v + loss_actions

    def correct_action(self, dresult: DecisionResult, obs_list: List):
        batch_size = len(obs_list)
        score_indices = []
        next_boards = []
        for ib in range(batch_size):
            board: Board = obs_list[ib]

            for isy, action_list in enumerate(dresult.actions[ib]):
                for ia, action in enumerate(action_list):
                    if dresult.mask[ib, isy, ia, 0] > 0:
                        board_next = KoreEnv._next_board(
                            board, {dresult.ids[ib][isy]: action.raw_action})
                        next_boards.append(board_next)
                        score_indices.append((ib, isy, ia))
                        pass
                    pass
                pass
            pass

        next_v = self.evaluate_obs(next_boards)

        next_scores = torch.zeros_like(dresult.scores[..., 0])
        score_indices = torch.tensor(score_indices, device=next_v.device)
        next_scores[score_indices[:, 0], score_indices[:, 1], score_indices[:, 2]] = next_v

        return next_scores

    def evaluate_obs(self, obs_list):
        obs_vec_list = [self.vectorize_env(obs) for obs in obs_list]
        obs_vec_batch = self.collate_obs_vec(obs_vec_list)

        # max_plan_length = np.floor(2 * np.log(obs_vec['my_shipyard_ship_count'])) + 1

        embs = self.model.encode_obs(obs_vec_batch)
        # map_emb: [B, S, C], my_shipyard.index: [B, SY]

        v = self.model.forward_v(embs)
        return v.flatten()

    def parameter_size(self):
        return self.model.parameter_size()
        pass

    def set_parameter(self, start, end, parameter):
        self.model.set_parameter(start, end, parameter)
        pass

    def copy_parameter(self, start, end):
        return self.model.copy_parameter(start, end)

    def parameters(self):
        return self.model.parameters()
    pass
