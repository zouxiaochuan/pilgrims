from distutils.spawn import spawn
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
        self.config = config
        self.size = config['map_size']
        self.size2 = self.size * self.size
        self.size_half = self.size // 2
        self.num_channels_fleet = 1 + 1 + 4  # kore, ship_count, direction
        self.num_channels_shipyard = 1  # ship_count
        self.num_channels = 1 + self.num_channels_fleet * 2 + \
            self.num_channels_shipyard * 2
        self.num_feat_player = 5  # kore, total kore, total ships, total shipyards, ship in shipyards
        self.num_feat = self.num_feat_player * 3 + 1 # first player, second, the difference
        config['agent']['num_input_map_channels'] = self.num_channels
        config['agent']['num_input_vec_channels'] = self.num_feat
        config['agent']['size2'] = self.size2
        config['agent']['map_size'] = self.size
        self.point_temperature = config['agent']['point_temperature']
        self.first_layer_temperature = config['agent']['first_layer_temperature']

        self.num_candidate_points = config['agent']['num_candidate_points']
        self.num_launch_points = self.size2
        self.model = KoreNetworks(config['agent'])
        self.need_grad = False
        env = KoreEnv(config)
        self.spawn_cost = env.spawn_cost
        self.device = 'cpu'
        self.mask4_list = []

        pass

    def train(self):
        self.need_grad = True
        pass

    def eval(self):
        self.need_grad = False
        pass

    def vectorize_env(self, obs):
        board: Board = obs

        map = torch.zeros((self.num_channels, self.size2), device=self.device)
        vec = torch.zeros(self.num_feat, device=self.device)

        map[0, :] = torch.from_numpy(utils_kore.get_board_kore(board))
        
        max_map_kore = torch.max(map[0, :])
        max_fleet_kore = 0
        max_ship = 0

        current_player_id = board.current_player_id
        my_shipyard_index = []
        my_shipyard_max_spawn = []
        my_shipyard_ship_count = []
        my_shipyard_id = []
        player_kore = torch.zeros(2, device=self.device)

        for player_id, player in board.players.items():
            player_absid = abs(current_player_id - player_id)
            player_feat_start = player_absid * self.num_feat_player

            vec[player_feat_start + 0] = player.kore
            player_kore[player_absid] = player.kore

        for fleet_id, fleet in board.fleets.items():
            player_absid = abs(current_player_id - fleet.player_id)
            fleet_channel = player_absid * self.num_channels_fleet + 1
            player_feat_start = player_absid * self.num_feat_player

            fleet_index = fleet.position.to_index(self.size)
            map[fleet_channel, fleet_index] = fleet.kore
            max_fleet_kore = max(max_fleet_kore, fleet.kore)
            map[fleet_channel + 1, fleet_index] = fleet.ship_count
            max_ship = max(max_ship, fleet.ship_count)
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
            max_ship = max(max_ship, shipyard.ship_count)
            vec[player_feat_start + 2] += shipyard.ship_count
            vec[player_feat_start + 3] += 1
            vec[player_feat_start + 4] += shipyard.ship_count

            if player_absid == 0:
                my_shipyard_index.append(shipyard_index)
                my_shipyard_max_spawn.append(shipyard.max_spawn)
                my_shipyard_ship_count.append(shipyard.ship_count)
                my_shipyard_id.append(shipyard_id)
                pass
            pass
        
        for i in range(self.num_feat_player):
            vec[2*self.num_feat_player + i] = torch.log(vec[i] + 1)
            pass

        if vec[2] <= 70:
            vec[3*self.num_feat_player + 0] = 1
            pass

        # normalize map
        map[0, :] /= max_map_kore
        if max_fleet_kore > 0:
            map[1, :] /= max_fleet_kore
            map[1 + self.num_channels_fleet, :] /= max_fleet_kore
            pass
        if max_ship > 0:
            map[2, :] /= max_ship
            map[2 + self.num_channels_fleet, :] /= max_ship
            channle_shipyard_start = 1 + self.num_channels_fleet * 2
            map[channle_shipyard_start: channle_shipyard_start+1, :] /= max_ship
            pass

        # normalize vec
        vec_sum = vec[:self.num_feat_player] + vec[self.num_feat_player:2*self.num_feat_player]
        vec_sum[vec_sum==0] = 1
        vec[:self.num_feat_player] /= vec_sum
        vec[self.num_feat_player:2*self.num_feat_player] /= vec_sum

        kore_map = map[0, :].reshape(self.size, self.size)
        map = torch.transpose(map, 0, 1)

        return {
            'map': map, 'vec': vec,
            'my_shipyard': {
                'index': torch.LongTensor(my_shipyard_index).to(self.device),
                'max_spawn': torch.FloatTensor(my_shipyard_max_spawn).to(self.device),
                'ship_count': torch.FloatTensor(my_shipyard_ship_count).to(self.device),
                'id': my_shipyard_id
            },
            'kore_map': kore_map,
            'player_kore': player_kore
        }
        pass

    def vectorize_act(self, actions: List[ShipyardActionEx]):
        v = torch.zeros((len(actions), 2), dtype=torch.long, device=self.device)
        for i, action in enumerate(actions):
            v[i, 0] = action.latent_variables['first_layer']
            point_index = action.latent_variables.get('point_index', 0)
            v[i, 1] = (self.size2)*(v[i, 0]-1) + point_index
            pass

        return v
    
    def collate_act_vec(self, action_vecs):
        max_len = max([v.shape[0] for v in action_vecs])
        batch = torch.zeros(
            (len(action_vecs), max_len, 2), dtype=action_vecs[0].dtype,
            device=self.device)
        for i, v in enumerate(action_vecs):
            batch[i, :v.shape[0], :] = v
            pass
        return batch


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
        map_emb = embs['map_emb']
        batch_size = map_emb.shape[0]
        max_shipyard = obs_vec_batch['my_shipyard']['mask'].shape[1]
        batch_index = embs['batch_index']
        shipyard_index = obs_vec_batch['my_shipyard']['index']

        shipyard_embs = map_emb[batch_index, shipyard_index]
        # shipyard_embs: [B, SY, C]

        v = self.model.forward_v(embs)

        score_first_layer = self.model.forward_first_layer(shipyard_embs)
        # [B, SY, 3]

        mask_cannot_launch = obs_vec_batch['my_shipyard']['ship_count']<21
        mask_cannot_convert = obs_vec_batch['my_shipyard']['ship_count']<50

        # if torch.logical_not(mask_cannot_convert).sum() > 0:
        #     print(torch.softmax(score_first_layer*2, dim=-1))
        #     exit(-1)

        mask_cannot_launch = mask_cannot_launch.unsqueeze(-1).repeat(1, 1, 3)
        mask_cannot_launch[:, :, 0] = False
        mask_cannot_launch[:, :, 2] = False
        mask_cannot_convert = mask_cannot_convert.unsqueeze(-1).repeat(1, 1, 3)
        mask_cannot_convert[:, :, 0] = False
        mask_cannot_convert[:, :, 1] = False

        score_first_layer[mask_cannot_launch] = -1e10
        score_first_layer[mask_cannot_convert] = -1e10
        score_spawn_branch = score_first_layer[..., 0]
        score_launch_branch = score_first_layer[..., 1]
        score_convert_branch = score_first_layer[..., 2]
        
        proba_first_layer = F.softmax(score_first_layer / self.first_layer_temperature, dim=-1)
        sampled_index = torch.multinomial(proba_first_layer.reshape(-1, 3), num_samples=1, replacement=False)
        sampled_index = sampled_index.reshape(batch_size, max_shipyard)
        # [B, SY]

        spawn_mask = sampled_index == 0
        launch_mask = sampled_index == 1
        convert_mask = sampled_index == 2

        # if spawn
        action_spawn, score_spawn, mask_spawn = self.decide_spawn(score_spawn_branch, obs_vec_batch, spawn_mask)

        # if launch, what's the plan
        action_launch, score_launch, mask_launch = self.decide_launch(
            score_launch_branch, embs, shipyard_index, obs_vec_batch, launch_mask)

        # if create new shipyard, what's the plan
        action_convert, score_convert, mask_convert = self.decide_convert(
            score_convert_branch, embs, shipyard_index, obs_vec_batch, convert_mask)

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
        device = feat_list[0].device
        batch_size = len(feat_list)
        feat_max_len = np.max([feat.shape[0] for feat in feat_list])
        # feat_dim = feat_list[0].shape[1]
        feat = torch.zeros(
            # (batch_size, feat_max_len, feat_dim),
            (batch_size, feat_max_len),
            dtype=feat_list[0].dtype, device=device)

        if return_mask:
            mask = torch.zeros(
                (batch_size, feat_max_len), dtype=torch.float32,
                device=device)
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

    def decide_spawn(self, score_spawn_branch, obs_vec_batch, spawn_mask):
        '''
        Parameters:
            score_spawn_branch: [B, SY]
            spawn_mask: [B, SY]

        Returns:
            action_spawn: [B, SY, 1]
            score_spawn: [B, SY, 1, 1]
            mask_spawn: [B, SY, 1, 1]
        '''
        max_spawn = obs_vec_batch['my_shipyard']['max_spawn']
        shipyard_mask = obs_vec_batch['my_shipyard']['mask']

        mask = torch.logical_and(shipyard_mask, spawn_mask)
        # shipyard_mask: [B, SY]

        B, SY = mask.shape
        kore_own = obs_vec_batch['player_kore'][:, 0]
        max_spawn2 = kore_own / self.spawn_cost
        action_spawn = [
            [
                [
                    ShipyardActionEx(
                        ShipyardAction(
                            ShipyardActionType.SPAWN,
                            int(min(max_spawn[b, i], max_spawn2[b])),
                            None),
                        latent_variables={'first_layer': 0})
                ] if mask[b, i] > 0 else [None] for i in range(SY)
            ]
            for b in range(B)
        ]

        score_spawn = score_spawn_branch.unsqueeze(-1)
        mask_spawn = mask.unsqueeze(-1).unsqueeze(-1)

        return action_spawn, score_spawn.unsqueeze(-1), mask_spawn

    def point_index2shift(self, point_index):
        return const_kore.point_index_to_xy[point_index].to(point_index.device)
        pass

    def map_index2xy(self, index, size=21):
        x = index % size
        y = torch.div(index, size, rounding_mode='trunc')
        # y = - y
        xy = torch.stack([x, y], axis=-1)

        return xy

    def decide_launch(self, score_launch_branch, embs, shipyard_index, obs_vec_batch, launch_mask):
        '''
        Parameters:
            score_launch_branch: [B, SY]
            shipyard_index: [B, SY]
            launch_mask: [B, SY]
        '''
        B, SY = score_launch_branch.shape
        batch_mask = torch.zeros((B, SY), device=score_launch_branch.device)
        ship_counts = obs_vec_batch['my_shipyard']['ship_count']

        batch_mask[ship_counts >= 21] = 1
        shipyard_mask = obs_vec_batch['my_shipyard']['mask']
        shipyard_action_mask = shipyard_mask * batch_mask * launch_mask

        if torch.sum(shipyard_action_mask) > 0:
            index_score = self.model.forward_launch_point(embs, shipyard_index)
            index_proba = torch.softmax(index_score / self.point_temperature, dim=-1)
            # index_proba: [B, SY, P(num_launch_points)]
            sampled_index = torch.multinomial(
                index_proba.reshape(-1, self.num_launch_points), num_samples=self.num_candidate_points, replacement=False)
            sampled_index = sampled_index.reshape(B, SY, self.num_candidate_points)

            # sampled_index: [B, SY, num_candidate_points]
            sampled_score = torch.gather(index_score, -1, sampled_index)
            # sample probability

            point_xy = self.map_index2xy(sampled_index)
            # point_xy: [B, SY, num_candidate_points, 2]

            shipyard_xy = self.map_index2xy(obs_vec_batch['my_shipyard']['index'])
            # shipyard_xy: [B, SY, 2]

            shift_xy = point_xy - shipyard_xy[:, :, None, :]
            shift_xy[shift_xy<-self.size_half] += self.size
            shift_xy[shift_xy>self.size_half] -= self.size

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
            sampled_score = torch.zeros((B, SY, 0), device=shipyard_mask.device)

            pass

        action_launch = [
            [
                [
                    ShipyardActionEx(
                        ShipyardAction(ShipyardActionType.LAUNCH, int(ship_counts[ibatch, ishipyard]), flight_plan),
                        latent_variables={'first_layer': 1, 'point_index': sampled_index[ibatch, ishipyard, iplan].item()}
                    ) if flight_plan is not None else None
                    for iplan, flight_plan in enumerate(shipyard_plan)
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

    def decide_convert(self, score_convert_branch, embs, shipyard_index, obs_vec_batch, convert_mask):
        B, SY = score_convert_branch.shape
        batch_mask = torch.zeros((B, SY), device=score_convert_branch.device)
        ship_counts = obs_vec_batch['my_shipyard']['ship_count']

        batch_mask[ship_counts >= 50] = 1
        shipyard_mask = obs_vec_batch['my_shipyard']['mask']
        shipyard_action_mask = shipyard_mask * batch_mask * convert_mask

        if torch.sum(shipyard_action_mask) > 0:
            index_score = self.model.forward_convert_point(embs, shipyard_index)
            index_proba = torch.softmax(index_score / self.point_temperature, dim=-1)
            # index_proba: [B, SY, P(num_convert_points)]

            sampled_index = torch.multinomial(
                index_proba.reshape(-1, self.num_launch_points), num_samples=self.num_candidate_points, replacement=False)
            sampled_index = sampled_index.reshape(B, SY, self.num_candidate_points)

            # sampled_index: [B, SY, num_candidate_points]
            sampled_score = torch.gather(index_score, -1, sampled_index)
            # sample probability

            point_xy = self.map_index2xy(sampled_index)
            # point_xy: [B, SY, num_candidate_points, 2]
            shipyard_xy = self.map_index2xy(obs_vec_batch['my_shipyard']['index'])
            # shipyard_xy: [B, SY, 2]

            shift_xy = point_xy - shipyard_xy[:, :, None, :]
            shift_xy[shift_xy<-self.size_half] += self.size
            shift_xy[shift_xy>self.size_half] -= self.size

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
            sampled_score = torch.zeros((B, SY, 0), device=shipyard_mask.device)
            pass

        action_convert = [
            [
                [
                    ShipyardActionEx(
                        ShipyardAction(
                            ShipyardActionType.LAUNCH, int(ship_counts[ibatch, ishipyard]),
                            flight_plan + 'C'
                        ),
                        latent_variables={'first_layer': 2, 'point_index': sampled_index[ibatch, ishipyard, iplan].item()}
                    ) if flight_plan is not None else None
                    for iplan, flight_plan in enumerate(shipyard_plan)
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
                            # raise RuntimeError('plan is empty')
                            plan = 'NS'

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

        nearest_shift = torch.cat(
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

        shift_xy = shift_xy.cpu().numpy().tolist()
        end_xy = end_xy.cpu().numpy().tolist()
        start_xy = start_xy.cpu().numpy().tolist()

        xstep = math.copysign(1, shift_xy[0])
        ystep = math.copysign(1, shift_xy[1])

        slice1_a = utils_kore.stack_value(
            torch.arange(
                start_xy[0] + xstep, end_xy[0] + xstep, xstep,
                device=kore_map.device,
                dtype=torch.int64),
            start_xy[1])
        slice2_a = utils_kore.stack_value(
            end_xy[0],
            torch.arange(
                start_xy[1] + ystep, end_xy[1] + ystep, ystep,
                device=kore_map.device,
                dtype=torch.int64)
        )
        slice1_b = utils_kore.stack_value(
            start_xy[0],
            torch.arange(
                start_xy[1] + ystep, end_xy[1] + ystep, ystep,
                device=kore_map.device,
                dtype=torch.int64)
        )
        slice2_b = utils_kore.stack_value(
            torch.arange(
                start_xy[0] + xstep, end_xy[0] + xstep, xstep,
                device=kore_map.device,
                dtype=torch.int64),
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

    def calculate_loss(
            self, obs_vec_batch, obs_next_vec_batch, act_vec_batch, reward_batch):

        embs = self.model.encode_obs(obs_vec_batch)
        # map_emb: [B, S, C], my_shipyard.index: [B, SY]
        batch_size = embs['map_emb'].shape[0]
        shipyard_mask = obs_vec_batch['my_shipyard']['mask'] > 0
        max_shipyard = shipyard_mask.shape[1]
        batch_index = embs['batch_index']
        shipyard_index = obs_vec_batch['my_shipyard']['index']
        shipyard_embs = embs['map_emb'][batch_index, shipyard_index]
        # shipyard_embs: [B, SY, C]

        v = self.model.forward_v(embs)
        with torch.no_grad():
            embs_next = self.model.encode_obs(obs_next_vec_batch)
            v_next = torch.sigmoid(self.model.forward_v(embs_next))
            pass

        loss_v = F.binary_cross_entropy_with_logits(v.flatten(), reward_batch)
        first_layer_scores = self.model.forward_first_layer(shipyard_embs)
        launch_scores = self.model.forward_launch_point(embs, shipyard_index)
        convert_scores = self.model.forward_convert_point(embs, shipyard_index)

        score1_index = act_vec_batch[:, :, 0]
        scores1 = torch.gather(
            first_layer_scores, -1, score1_index.unsqueeze(-1).clone()).squeeze(-1)
        score2_mask = torch.logical_and(score1_index > 0, shipyard_mask)


        launch_and_convert_scores = torch.cat(
            [launch_scores, convert_scores], dim=-1
        )
        score2_index = act_vec_batch[:, :, 1]
        score2_index *= score2_mask.long()

        scores2 = torch.gather(
            launch_and_convert_scores, -1, score2_index.unsqueeze(-1).clone()
        ).squeeze(-1)
        
        v_next = torch.tile(v_next, [1, max_shipyard])
        # v_next = torch.tile(reward_batch[:, None], [1, max_shipyard])

        loss1 = torch.mean(
            torch.masked_select(
                F.binary_cross_entropy_with_logits(scores1, v_next, reduction='none'),
            shipyard_mask))
        
        loss2 = torch.mean(
            torch.masked_select(
                F.binary_cross_entropy_with_logits(scores2, v_next, reduction='none'),
            score2_mask))

        # debug
        score3_mask = torch.logical_and(score1_index == 2, shipyard_mask)
        v_next_score3 = torch.masked_select(v_next, score3_mask)
        v_score3 = torch.masked_select(torch.tile(v, [1, max_shipyard]), score3_mask)
        reward_score3 = torch.masked_select(torch.tile(reward_batch[:, None], [1, max_shipyard]), score3_mask)
        vec_score3 = torch.tile(obs_vec_batch['vec'][:, None, :], [1, max_shipyard, 1])[score3_mask, :]

        mask4 = (torch.exp(vec_score3[:, 2*self.num_feat_player+2])-1) < 70

        if torch.sum(mask4) > 0:
            self.mask4_list.append(reward_score3[mask4].cpu().numpy())
            pass
        
        if len(self.mask4_list) > 100:
            self.mask4_list.pop(0)
            pass

        if torch.sum(mask4) >0:
            # print('debug')
            pass
        
        if len(self.mask4_list) > 0:
            print(f'{np.concatenate(self.mask4_list).mean()}')
            pass
        
        # debug
        # mymask = score1_index[shipyard_mask] == 1

        # score_mymask = scores1[shipyard_mask][mymask]
        # v_next_mymask = v_next[shipyard_mask][mymask]
        # self.sum_mask4 += torch.sum(v_next[shipyard_mask][mymask]).item()
        # self.cnt_mask4 += torch.sum(mymask).item()

        # if torch.sum(mymask) > 0:
        #     print('debug')
        #     pass

        # print(self.sum_mask4 / self.cnt_mask4)

        return loss_v, loss1, loss2
        # return loss1, loss2

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

    def get_next_obs_idx(self, act):
        idx = 999
        for a in act:
            if a.raw_action.action_type == ShipyardActionType.SPAWN:
                continue
            else:
                len = 0
                for ch in a.raw_action.flight_plan:
                    if ch == 'C':
                        len += 1
                    elif str.isdigit(ch):
                        len += int(ch)
                    else:
                        len += 1
                        pass
                    pass
                idx = min(idx, len)
                pass
            pass

        if idx == 999:
            return 1
        else:
            return idx
        pass


    def get_default_action(self, obs: Board):
        actions = []
        for _ in obs.players[obs.current_player_id].shipyards:
            actions.append(
                ShipyardActionEx(
                    ShipyardAction(ShipyardActionType.SPAWN, 0, None),
                    latent_variables={'first_layer': 0}
                ))
            pass
        return actions

    def get_action_weight(self, action: ShipyardActionEx):
        if action.raw_action.action_type == ShipyardActionType.SPAWN:
            return 1
        else:
            if action.raw_action.flight_plan[-1] == 'C':
                return 6
            else:
                return 3
            pass
        pass

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
    
    def to_device(self, device):
        self.device = device
        self.model.to(device)

    @classmethod
    def load_model(cls, path):
        config, state_dict = torch.load(path)
        agent = cls(config)
        agent.model.load_state_dict(state_dict)
        return agent
        pass

    def save_model(self, path):
        state_dict = self.model.state_dict()
        config = self.config
        torch.save((config, state_dict), path)
        pass
    pass
