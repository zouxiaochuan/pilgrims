import torch
import numpy as np
import random
torch.manual_seed(2)
np.random.seed(2)
random.seed(2)

import actor
import json
import os

from agent_factory import AgentFactory
# import environments.kore.agent_kore

current_module_path = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    with open(os.path.join(current_module_path, 'config.json')) as f:
        config = json.load(f)
        pass

    actor.run(config)
    pass