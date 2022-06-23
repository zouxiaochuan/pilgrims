
class DecisionResult:
    def __init__(self, actions, scores, score_mask, ids):
        self.actions = actions
        self.scores = scores
        self.mask = score_mask
        self.ids = ids
        pass
    pass


class Agent:
    pass