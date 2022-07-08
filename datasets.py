import torch
import random
import Pyro5.api as api
import pickle
from kaggle_environments.envs.kore_fleets.helpers import Board
from environment_factory import EnvironmentFactory
from agent_factory import AgentFactory


def collate_fn(batch):
    return batch[0]


class ReplayBufferDataset(torch.utils.data.Dataset):
    def __init__(self, rb_uris, config, num=10000, batch_size=16):
        self.rb_uris = rb_uris
        self.num = num
        self.rb_list = None
        self.batch_size = batch_size

        self.env_class = EnvironmentFactory.create(config).__class__
        self.agent = AgentFactory.create(config)
        # self.agent.to_device('cpu')
        pass

    def init_proxy(self, uris):
        rb_list = []

        for uri in uris:
            rb_list.append(api.Proxy(uri))
            pass

        return rb_list

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        if self.rb_list is None:
            self.rb_list = self.init_proxy(self.rb_uris)
            pass

        rb = random.choice(self.rb_list)
        data = rb.sample(num=self.batch_size)

        obs_vecs = []
        obs_next_vecs = []
        act_vecs = []
        rewards = []
        for s_obs, s_obs_next, s_a, s_reward in data:
            obs = self.env_class.deserialize_obs(s_obs)
            obs_next = self.env_class.deserialize_obs(s_obs_next)
            act = pickle.loads(s_a)

            reward = pickle.loads(s_reward)

            obs_vec = self.agent.vectorize_env(obs)
            obs_next_vec = self.agent.vectorize_env(obs_next)
            act_vec = self.agent.vectorize_act(act)
            # if act_vec.shape[0] == 1:
            #     if act_vec[0, 0] == 2 and obs_vec['vec'][10] == obs_next_vec['vec'][10]:
            #         print('debug')
            #         pass
            #     pass

            obs_vecs.append(obs_vec)
            obs_next_vecs.append(obs_next_vec)
            act_vecs.append(act_vec)
            rewards.append(reward)
            pass

        obs_vec_batch = self.agent.collate_obs_vec(obs_vecs)
        obs_next_vec_batch = self.agent.collate_obs_vec(obs_next_vecs)
        act_vec_batch = self.agent.collate_act_vec(act_vecs)
        reward_batch = torch.tensor(rewards)
        return {'obs': obs_vec_batch, 'obs_next': obs_next_vec_batch, 'act': act_vec_batch, 'reward': reward_batch}
    pass
