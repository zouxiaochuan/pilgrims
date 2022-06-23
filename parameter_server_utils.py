from agent import Agent
from typing import List
from parameter_server import ParameterServer
import common_utils
import pickle


def load_parameter(agent: Agent, ps_list: List[ParameterServer]):
    parameter_size = agent.parameter_size()
    num_servers = len(ps_list)

    for i, ps in enumerate(ps_list):
        start, end = common_utils.chunk_parameter(parameter_size, num_servers, i)
        param = pickle.loads(ps.get_parameter())
        agent.set_parameter(start, end, param)
        pass
    pass