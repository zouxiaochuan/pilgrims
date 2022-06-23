import torch
import random
import Pyro5.api as api
import pickle
from kaggle_environments.envs.kore_fleets.helpers import Board


def collate_fn(batch):
    batch = batch[0]
    obs = [b[0] for b in batch]
    reward = torch.tensor([b[1] for b in batch])
    
    return {'obs': obs, 'reward': reward}


class ReplayBufferDataset(torch.utils.data.Dataset):
    def __init__(self, rb_uris, num=10000, batch_size=16):
        self.rb_uris = rb_uris
        self.num = num
        self.rb_list = None
        self.batch_size = batch_size
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

        data_out = []
        for s in data:
            obs, conf, reward = pickle.loads(s)
            board = Board(obs, conf)
            data_out.append((board, reward))
            pass

        return data_out
    pass
