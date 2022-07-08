import Pyro5.api as api
from Pyro5.nameserver import NameServer
import Pyro5.server as server
import click
import pyro_utils
import numpy as np
import common_utils
from typing import List
from parameter_server import ParameterServer
from replay_buffer import ReplayBuffer
from agent_factory import AgentFactory
from environment_factory import EnvironmentFactory
from datasets import ReplayBufferDataset, collate_fn
import torch
import torch.utils
import torch.utils.data
import parameter_server_utils
import pickle
from datetime import datetime
import os


class Learner():
    def __init__(self, config, ps_uris, rb_uris):
        self.agent = AgentFactory.create(config)
        self.agent.to_device('cuda:0')
        self.env = EnvironmentFactory.create(config)
        self.config = config

        self.ps_list = [api.Proxy(uri) for uri in ps_uris]
        self.rb_uris = rb_uris
        self.num_step_per_push_ps = config['learner']['num_step_per_push_ps']
        self.num_data_workers = config['learner']['num_data_workers']
        self.batch_size = config['learner']['batch_size']
        self.save_model_interval = config['learner']['save_model_interval']
        self.save_model_path = config['learner']['save_model_path']
        self.total_step = 0

        self.ps_start_index = []
        self.ps_end_index = []

        for i in range(len(self.ps_list)):
            start, end = common_utils.chunk_parameter(
                self.agent.parameter_size(), len(self.ps_list), i)
            self.ps_start_index.append(start)
            self.ps_end_index.append(end)
            pass

        self.loadParameter(self.ps_list)
        pass

    def loadParameter(self, ps_list: List[ParameterServer]):
        parameter_server_utils.load_parameter(self.agent, ps_list)
        pass

    def pushParameter(self, ps_list: List[ParameterServer], starts, ends):
        for i, ps in enumerate(ps_list):
            start, end = starts[i], ends[i]
            v = self.agent.copy_parameter(start, end)
            print(v[-113620])
            s = pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
            ps.put_parameter(s)
            pass
        pass

    def learnForever(self):
        dataset = ReplayBufferDataset(self.rb_uris, self.config, batch_size=self.batch_size)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
            num_workers=self.num_data_workers)
        self.agent.train()
        optimizer = torch.optim.Adam(self.agent.parameters(), lr=0.0001)

        while True:
            for batch in dataloader:
                batch = common_utils.batch_to_device(batch, self.agent.device)
                losses = self.agent.calculate_loss(
                    batch['obs'], batch['obs_next'], batch['act'], batch['reward'])
                loss = sum(losses)
                
                if self.total_step % 10 == 0:
                    print(f'step: {self.total_step}, losses: {[l.item() for l in losses]}')
                    pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.total_step += 1

                if self.total_step % self.num_step_per_push_ps == 0:
                    self.pushParameter(self.ps_list, self.ps_start_index, self.ps_end_index)
                    pass

                if self.total_step % self.save_model_interval == 0:
                    filename = os.path.join(
                        self.save_model_path, datetime.now().strftime('%Y%m%d%H%M%S') + '.pth')
                    self.agent.save_model(filename)
                    pass
                pass
            pass
        pass
    pass


@click.command()
@click.option('--config-file', default='config.json')
@click.option('--device')
def run(config_file, device):
    config = common_utils.load_config(config_file)
    master_ip = config['master_ip']
    master_port = config['master_port']

    ns: NameServer = api.locate_ns(host=master_ip, port=master_port)
    rb_uris = ns.list(prefix='rb').values()
    ps_uris = ns.list(prefix='ps').values()

    learner = Learner(config, ps_uris, rb_uris)

    learner.learnForever()
    pass


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    __spec__ = 'nihao'
    # torch.multiprocessing.set_start_method('spawn')
    run()
    pass
