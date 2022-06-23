from kaggle_environments.envs.kore_fleets.helpers import *
from random import randint

def agent(obs, config):
    board = Board(obs, config)
    me=board.current_player

    me = board.current_player
    turn = board.step
    spawn_cost = board.configuration.spawn_cost
    kore_left = me.kore

    period = 4 + config.size + 1
    
    for shipyard in me.shipyards:
        action = None
        if turn % period == 4:
            action = ShipyardAction.launch_fleet_with_flight_plan(2, "ES")
        elif turn % period == 6: 
            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E2S")
        elif turn % period == 4 + config.size:
            action = ShipyardAction.launch_fleet_with_flight_plan(3, "E3W")
            shipyard.next_action = action
        elif kore_left >= spawn_cost:
            action = ShipyardAction.spawn_ships(1)
        shipyard.next_action = action

    return me.next_actions