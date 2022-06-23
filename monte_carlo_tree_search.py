
def MCTS(root, max_iterations, exploration_constant):
    """
    MCTS algorithm
    """
    for _ in range(max_iterations):
        node = root
        while node.untried_actions:
            node = node.select_child()
            node.rollout()
        node.update_stats()
    return root.select_best_child(exploration_constant)


class MCTS:
    def __init__(self,):
        pass


    def search(self, agent, env, width, depth, player_id=0):
        obs_list = [(env.current_obs(player=player_id), 1)]
        for idepth in range(depth):
            candidates = []
            for obs, value_pre in obs_list:
                actions, proba, value = agent.decide(obs)
                proba *= value_pre

                candidates.append((obs, ))
                pass
            pass
        pass

