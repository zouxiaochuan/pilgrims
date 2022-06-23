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

pyro_utils.init()


class Actor():
    def __init__(self, config, ps_list, rb_list):
        self.agent = AgentFactory.create(config)

        self.ps_list = ps_list
        self.rb_list = rb_list
        self.num_states_per_play = config['actor']['num_states_per_play']
        self.temperature = config['actor']['temperature']
        self.batch_size = config['actor']['batch_size']
        self.save_play_interval = config['actor']['save_play_interval']
        self.save_play_path = config['actor']['save_play_path']

        self.envs = [EnvironmentFactory.create(config) for _ in range(self.batch_size)]

        self.rb_idx = 0
        pass

    def loadParameter(self, ps_list: List[ParameterServer]):
        parameter_server_utils.load_parameter(self.agent, ps_list)
        pass

    def putReplayBuffer(self, states, reward: float, rb_list: List[ReplayBuffer]):
        for state in states:

            data = pickle.dumps((state.observation, state.configuration, reward), protocol=pickle.HIGHEST_PROTOCOL)
            rb_list[self.rb_idx].put(data)

            self.rb_idx += 1
            if self.rb_idx >= len(rb_list):
                self.rb_idx = 0
                pass
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

        action_saved = []
        obs_saved = []
        istep = 0
        while True:
            if istep == 141:
                print('debug')
                pass
            print([env.board_step() for env in self.envs])
            obs0 = [env.current_obs(player=0) for env in self.envs]
            obs1 = [env.current_obs(player=1) for env in self.envs]

            for i, (o0, o1) in enumerate(zip(obs0, obs1)):
                if not done_mask[i]:
                    obs0_list[i].append(o0)
                    obs1_list[i].append(o1)
                    pass
                pass

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

            for ienv, (env, act0, act1, ids0, ids1) in enumerate(
                    zip(self.envs, action0, action1, dresult0.ids, dresult1.ids)):

                act0 = {id: a[0].raw_action for id, a in zip(ids0, act0)}
                act1 = {id: a[0].raw_action for id, a in zip(ids1, act1)}

                if is_save_play and ienv == save_play_idx:
                    action_saved.append((act0, act1))
                    obs_saved.append(env.board)
                    pass

                env.step({**act0, **act1})
                pass

            done_mask = [env.done for env in self.envs]

            if all(done_mask):
                break

            istep += 1
            pass

        rewards = [env.get_reward() for env in self.envs]

        for obs0, obs1, (reward0, reward1) in zip(obs0_list, obs1_list, rewards):
            obs0 = np.array(obs0, dtype='object')
            obs0_sampled = np.random.choice(obs0, self.num_states_per_play, replace=False)
            obs1 = np.array(obs1, dtype='object')
            obs1_sampled = np.random.choice(obs1, self.num_states_per_play, replace=False)

            self.putReplayBuffer(obs0_sampled, reward0, self.rb_list)
            self.putReplayBuffer(obs1_sampled, reward1, self.rb_list)
            pass

        if is_save_play:
            self.envs[0].__class__.save_play(
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
    pass


def run(config):
    master_ip = config['master_ip']
    master_port = config['master_port']

    ns: NameServer = api.locate_ns(host=master_ip, port=master_port)
    ps_list = pyro_utils.get_all_objects(ns, 'ps')
    rb_list = pyro_utils.get_all_objects(ns, 'rb')

    actor = Actor(config, ps_list, rb_list)

    actor.playForever()
    pass


@click.command()
@click.option('--config-file', default='config.json')
def run_cmd(config_file):
    config = common_utils.load_config(config_file)
    run(config)
    pass


if __name__ == '__main__':
    run_cmd()
    pass
