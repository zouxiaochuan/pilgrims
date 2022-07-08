import Pyro5.api as api
from Pyro5.nameserver import NameServer
import Pyro5.server as server
import click
import pyro_utils
import numpy as np
import common_utils
from typing import List
import transformers
from parameter_server import ParameterServer
from replay_buffer import ReplayBuffer
from agent_factory import AgentFactory
from agent import DecisionResult
from environment_factory import EnvironmentFactory
import proba_utils
import pickle
import random
import parameter_server_utils
from copy import deepcopy

pyro_utils.init()


class Actor():
    def __init__(self, config, ps_list, rb_list, device='cpu'):
        self.agent = AgentFactory.create(config)

        self.ps_list = ps_list
        self.rb_list = rb_list
        self.num_states_per_play = config['actor']['num_states_per_play']
        self.temperature = config['actor']['temperature']
        self.batch_size = config['actor']['batch_size']
        self.save_play_interval = config['actor']['save_play_interval']
        self.save_play_path = config['actor']['save_play_path']

        self.envs = [EnvironmentFactory.create(config) for _ in range(self.batch_size)]
        self.env_class = self.envs[0].__class__

        self.rb_idx = 0
        self.device = device
        self.agent.to_device(device)
        pass

    def loadParameter(self, ps_list: List[ParameterServer]):
        parameter_server_utils.load_parameter(self.agent, ps_list)
        pass

    def putReplayBuffer(self, obs, obs_next, act, reward: float, rb_list: List[ReplayBuffer]):
        s = self.env_class.serialize_obs(obs)
        if obs_next is None:
            s_next = None
        else:
            s_next = self.env_class.serialize_obs(obs_next)
            pass
        s_act = pickle.dumps(act, protocol=pickle.HIGHEST_PROTOCOL)
        s_reward = pickle.dumps(reward, protocol=pickle.HIGHEST_PROTOCOL)

        data = (s, s_next, s_act, s_reward)
        rb_list[self.rb_idx].put(data)

        self.rb_idx += 1
        if self.rb_idx >= len(rb_list):
            self.rb_idx = 0
            pass
        pass

    def selfPlay(self, iplay):
        self.loadParameter(self.ps_list)
        for env in self.envs:
            env.reset()
            pass

        is_save_play = iplay % self.save_play_interval == 0

        if is_save_play:
            save_play_idx = random.randint(0, len(self.envs) - 1)
            pass

        done_mask = [env.done for env in self.envs]
        obs0_list = [list() for _ in range(self.batch_size)]
        obs1_list = [list() for _ in range(self.batch_size)]
        action0_list = [list() for _ in range(self.batch_size)]
        action1_list = [list() for _ in range(self.batch_size)]

        action_saved = []
        obs_saved = []
        istep = 0

        for i, env in enumerate(self.envs):
            obs0_list[i].append(env.current_obs(player=0))
            obs1_list[i].append(env.current_obs(player=1))
            pass

        while True:
            if istep == 21:
                # print('debug')
                pass
            print([env.board_step() for env in self.envs])
            obs0 = [env.current_obs(player=0) for env in self.envs]
            obs1 = [env.current_obs(player=1) for env in self.envs]

            dresult0, v0 = self.agent.decide(obs0)
            dresult1, v1 = self.agent.decide(obs1)

            action0 = proba_utils.sample_action(
                dresult0.actions, dresult0.scores[..., 0],
                dresult0.mask[..., 0], num=1,
                temperature=self.temperature)
            action1 = proba_utils.sample_action(
                dresult1.actions, dresult1.scores[..., 0],
                dresult1.mask[..., 0], num=1,
                temperature=self.temperature)
            
            for i, (a0, a1) in enumerate(zip(action0, action1)):
                if not done_mask[i]:
                    # if len(a0[-1]) == 0:
                    #     raise RuntimeError('haha')
                    action0_list[i].append(a0)
                    action1_list[i].append(a1)
                    pass
                pass

            for ienv, (env, act0, act1, ids0, ids1) in enumerate(
                    zip(self.envs, action0, action1, dresult0.ids, dresult1.ids)):

                if done_mask[ienv]:
                    continue

                act0 = {id: a[0].raw_action for id, a in zip(ids0, act0)}
                act1 = {id: a[0].raw_action for id, a in zip(ids1, act1)}

                if is_save_play and ienv == save_play_idx and not done_mask[save_play_idx]:
                    action_saved.append((act0, act1))
                    obs_saved.append(env.board)
                    pass

                env.step({**act0, **act1})

                obs0_list[ienv].append(env.current_obs(player=0))
                obs1_list[ienv].append(env.current_obs(player=1))
                pass

            done_mask = [env.done for env in self.envs]

            if all(done_mask):
                break

            istep += 1
            pass

        rewards = [env.get_reward() for env in self.envs]

        for obs0, obs1, action0, action1, (reward0, reward1) in zip(
                obs0_list, obs1_list, action0_list, action1_list, rewards):
            self.sample_action_and_put_buffer(obs0, action0, reward0)
            self.sample_action_and_put_buffer(obs1, action1, reward1)
            pass

        if is_save_play:
            self.env_class.save_play(
                self.save_play_path, iplay, action_saved, obs_saved)
            pass
        pass

    def playForever(self):
        self.agent.eval()

        iplay = 0
        while True:
            self.selfPlay(iplay)
            iplay += 1
            pass
        pass


    def sample_action_and_put_buffer(self, obs_list, act_list, reward):
    
        play_length = len(obs_list)

        weights = []

        for act in act_list:
            if len(act) == 0:
                # no agents
                weights.append(1)
            else:
                weights.append(
                    max(self.agent.get_action_weight(a[0]) if len(a)>0 else 1 for a in act))
                pass
            pass

        weights.append(1)

        weights = np.array(weights) / np.sum(weights)

        selected_play_idx = np.random.choice(
            play_length, min(self.num_states_per_play, play_length), replace=False,
            p=weights)

        for idx in selected_play_idx:
            obs = obs_list[idx]
            if idx == play_length - 1:
                obs_next = deepcopy(obs)
                act = self.agent.get_default_action(obs_next)
            else:
                act = [a[0] for a in act_list[idx] if len(a)>0]
                idx_next = idx + self.agent.get_next_obs_idx(act)
                obs_next = obs_list[min(idx_next, play_length - 1)]
                pass

            r = reward
            self.putReplayBuffer(obs, obs_next, act, r, self.rb_list)
            pass
        pass
    pass


def run(config, device):
    master_ip = config['master_ip']
    master_port = config['master_port']

    ns: NameServer = api.locate_ns(host=master_ip, port=master_port)
    ps_list = pyro_utils.get_all_objects(ns, 'ps')
    rb_list = pyro_utils.get_all_objects(ns, 'rb')

    actor = Actor(config, ps_list, rb_list, device)

    actor.playForever()
    pass


@click.command()
@click.option('--config-file', default='config.json')
@click.option('--device', default='cpu')
def run_cmd(config_file, device):
    config = common_utils.load_config(config_file)
    run(config, device)
    pass


if __name__ == '__main__':
    run_cmd()
    pass
