import Pyro5.api as api
from Pyro5.nameserver import NameServer
import Pyro5.server as server
import click
import pyro_utils
import numpy as np
import common_utils
import pickle
from agent_factory import AgentFactory


pyro_utils.init()

NAME_KEY = 'ps'


@api.expose
class ParameterServer():
    def __init__(self, server_index, num_parameters, num_servers):
        self.server_index = server_index
        self.num_parameters = num_parameters
        self.num_servers = num_servers

        self.start_index, self.end_index = common_utils.chunk_parameter(
            self.num_parameters, self.num_servers, self.server_index)
        self.parameters = np.zeros(self.end_index - self.start_index, dtype='float32')
        pass

    def get_parameter(self):
        s = pickle.dumps(self.parameters, protocol=pickle.HIGHEST_PROTOCOL)
        return s

    @api.oneway
    def put_parameter(self, parameter):
        v = pickle.loads(parameter)
        self.parameters[:] = v
        pass


@click.command()
@click.option('--config-file', default='config.json')
def run(config_file):
    config = common_utils.load_config(config_file)
    master_ip = config['master_ip']
    master_port = config['master_port']
    ns: NameServer = api.locate_ns(host=master_ip, port=master_port)

    agent = AgentFactory.create(config)
    num_parameter = agent.parameter_size()
    num_server = config['parameter_server']['num_workers']

    id = pyro_utils.get_next_id(ns, NAME_KEY)
    name = f'{NAME_KEY}_{id}'
    daemon = server.Daemon()
    uri = daemon.register(
        ParameterServer(
            server_index=id, num_parameters=num_parameter,
            num_servers=num_server))
    ns.register(name, uri)
    try:
        daemon.requestLoop()
    finally:
        ns.remove(name)
        pass
    pass


if __name__ == '__main__':
    run()
    pass