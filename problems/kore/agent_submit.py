from .agent_kore import KoreAgent
from kaggle_environments.envs.kore_fleets.helpers import (
    Board, ShipyardAction, ShipyardActionType)
import proba_utils

class AgentFromRL(object):
    def __init__(self, model_path, device='cuda:0'):
        self.agent = KoreAgent.load_model(model_path)
        self.agent.first_layer_temperature = 0.0001
        self.agent.point_temperature = 0.0001
        self.agent.eval()
        self.agent.to_device(device)
        pass


    def __call__(self, obs, config):
        board = Board(obs, config)
        dresult, _ = self.agent.decide([board])

        action0 = proba_utils.sample_action(
            dresult.actions, dresult.scores[..., 0],
            dresult.mask[..., 0], num=1,
            temperature=0.001)[0]
        
        for ia, aid in enumerate(dresult.ids[0]):
            board.shipyards[aid].next_action = action0[ia][0].raw_action
            pass


        return board.current_player.next_actions
        pass