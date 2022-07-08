from kaggle_environments.envs.kore_fleets.kore_fleets import miner_agent
from random import randint
import time

tick = time.time()

def agent(obs, config):
    return miner_agent(obs, config)