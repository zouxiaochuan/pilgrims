import kaggle_environments
import numpy as np
from kaggle_environments.envs.kore_fleets.helpers import Board, ShipyardAction, PlayerId
from copy import deepcopy
from typing import Dict
import time
import os
import pickle


GLOBAL_INITIAL_ENV = kaggle_environments.make("kore_fleets")
GLOBAL_INITIAL_ENV.reset(num_agents=2)


class KoreEnv:
    def __init__(self, config, board: Board = None):
        global GLOBAL_INITIAL_ENV
        self.size = GLOBAL_INITIAL_ENV.configuration.size
        self.configuration = GLOBAL_INITIAL_ENV.configuration
        self.spawn_cost = GLOBAL_INITIAL_ENV.configuration.spawnCost

        if board is None:
            self.board = Board(GLOBAL_INITIAL_ENV.state[0].observation, self.configuration)
        else:
            self.board = deepcopy(board)
            pass
        self.done = False
        self.rewards = np.array([0, 0], dtype='float32')
        self.player_fail = [False, False]
        self.check_done()
        pass

    def reset(self):
        env = kaggle_environments.make("kore_fleets")
        env.reset(num_agents=2)
        self.board = Board(env.state[0].observation, self.configuration)
        self.done = False
        self.rewards.fill(0)
        self.player_fail = [False, False]
        pass

    def current_obs(self, player: int):
        self.board._current_player_id = PlayerId(player)

        return deepcopy(self.board)
        pass

    def check_done(self):
        if self.board.step > self.configuration.episodeSteps:
            self.done = True
            if self.board.players[0].kore > self.board.players[1].kore:
                self.rewards[0] = 1
            elif self.board.players[0].kore < self.board.players[1].kore:
                self.rewards[1] = 1
                pass
            pass
        else:
            # if one player fail, the game is done
            for player_id in range(2):
                player = self.board.players[player_id]
                num_shipyards = len(player.shipyard_ids)
                
                num_fleets = len(player.fleet_ids)
                num_kore = player.kore
                
                if num_fleets == 0:
                    if num_shipyards == 0:
                        self.player_fail[player_id] = True
                    else:
                        max_ships_in_shipyards = max(shipyard.ship_count for shipyard in player.shipyards)
                        if num_kore < self.spawn_cost:
                            if max_ships_in_shipyards < 21:
                                self.player_fail[player_id] = True
                                pass
                            pass
                        pass
                    pass
                pass

            if any(self.player_fail):
                self.done = True
                if not self.player_fail[0]:
                    self.rewards[0] = 1
                elif not self.player_fail[1]:
                    self.rewards[1] = 1
                    pass
                pass
            pass
        pass

    def step(self, shipyard_actions: Dict[str, ShipyardAction]):
        for shipyard_id, action in shipyard_actions.items():
            self.board.shipyards[shipyard_id].next_action = action
            pass

        self.board = self.board.next()

        self.check_done()

        pass

    def board_step(self):
        return self.board.step

    @classmethod
    def _next_board(cls, board: Board, shipyard_actions: Dict[str, ShipyardAction]):
        for shipyard_id, action in shipyard_actions.items():
            board.shipyards[shipyard_id].next_action = action
            pass

        return board.next()
        pass

    def get_reward(self):
        return self.rewards
        pass

    @classmethod
    def save_play(cls, folder, iplay, actions, obs_list):
        actions0 = [a[0] for a in actions]
        actions1 = [a[1] for a in actions]

        print(f'ha1{len(actions0)}')
        print(f'ha2{len(actions1)}')
        agent0 = AgentGivenAction(actions0, obs_list)
        agent1 = AgentGivenAction(actions1, obs_list)

        env = kaggle_environments.make("kore_fleets", debug=True)
        env.reset(num_agents=2)
        state = env.state
        initial_board = obs_list[0]
        state[0]['observation'] = kaggle_environments.utils.structify(initial_board.observation)
        state[0]['observation']['player'] = 0
        # env.__set_state(state)

        agents = [agent0, agent1]
        runner = env._Environment__agent_runner(agents)

        while not env.done:
            actions, logs = runner.act()
            env.step(actions, logs)
            pass

        res = env.render(mode="html", width=1000, height=800)

        filename = 'play_' + str(int(time.time())) + '_' + str(os.getpid()) + '_' + '.html'
        with open(folder + '/' + filename, 'w') as f:
            f.write(res)
            pass
        pass

    @classmethod
    def serialize_obs(cls, obs: Board):
        return pickle.dumps((obs.observation, obs.configuration), protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def deserialize_obs(cls, obs_str):
        obs, config = pickle.loads(obs_str)
        return Board(obs, config)
    pass


def check_board(board):
    spawn_cost = board.configuration['spawnCost']
    player_fail = [False, False]

    for player_id in range(2):
        player = board.players[player_id]
        num_shipyards = len(player.shipyard_ids)
        
        num_fleets = len(player.fleet_ids)
        num_kore = player.kore
        
        if num_fleets == 0:
            if num_shipyards == 0:
                player_fail[player_id] = True
            else:
                max_ships_in_shipyards = max(shipyard.ship_count for shipyard in player.shipyards)
                if num_kore < spawn_cost:
                    if max_ships_in_shipyards < 21:
                        player_fail[player_id] = True
                        pass
                    pass
                pass
            pass
        pass

    return any(player_fail)

class AgentGivenAction(object):
    def __init__(self, actions, obs_list):
        self.actions = actions
        self.obs_list = obs_list
        pass

    def __call__(self, obs, config):
        step = obs['step']
        
        if step >= len(self.actions):
            return {}

        # debug
        board = Board(obs, self.obs_list[0].configuration)
        if check_board(board):
            print('debug')
            pass

        # obs_ = self.obs_list[step]

        # kore1 = obs['kore']
        # kore2 = obs_.observation['kore']

        # for k1, k2, in zip(kore1, kore2):
        #     if k1 != k2:
        #         raise RuntimeError('kore not match')
        #     pass

        return {k: v.name for k, v in self.actions[step].items()}
    pass
