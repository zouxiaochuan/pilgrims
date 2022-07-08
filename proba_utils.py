import torch


def masked_softmax(scores, mask, dim):
    '''masked softmax
    '''
    scores_ = scores.clone()
    scores_[mask==0] = -1e10
    return torch.nn.functional.softmax(scores_, dim=dim)


def sample_action(
        actions, scores: torch.Tensor, mask: torch.Tensor, num, temperature):
    '''sample actions according to scores

    Parameters:
        actions: action list of shape [B, A, N]
        scores: probability tensor of shape [B, A, N]
        mask: score mask of shape [B, A, N]
    '''

    # number A maybe zero
    if scores.shape[1] == 0:
        return [[] for _ in range(scores.shape[0])]
        pass
    
    B, A, N = scores.shape
    proba = masked_softmax(scores / temperature, mask, dim=-1)
    sampled_index = torch.multinomial(proba.reshape(-1, N), num_samples=num, replacement=False)
    sampled_index = sampled_index.reshape(B, A, -1)
    # sampled_index: [B, A, num]

    actions_sampled = []
    for ibatch, action_agent in enumerate(actions):
        action_agent_sampled = []
        for ia, action_row in enumerate(action_agent):
            action_row_sampled = []

            for i in sampled_index[ibatch, ia, :]:
                if mask[ibatch, ia, i] > 0:
                    action_row_sampled.append(action_row[i])
                    pass
                pass
            action_agent_sampled.append(action_row_sampled)
            pass
        actions_sampled.append(action_agent_sampled)
        pass

    return actions_sampled


def sample_action_single(
        actions, scores: torch.Tensor, mask: torch.Tensor, num, temperature):
    '''sample actions according to scores

    Parameters:
        actions: action list of shape [A, N]
        scores: probability tensor of shape [A, N]
        mask: score mask of shape [A, N]
    '''

    A, N = scores.shape

    max_actions = torch.min(torch.sum(mask, dim=-1))

    assert max_actions >= num

    proba = masked_softmax(scores / temperature, mask, dim=-1)
    sampled_index = torch.multinomial(proba, num_samples=num, replacement=False)
    # sampled_index: [A, num]

    actions_sampled = []
    for ia, action_row in enumerate(actions):
        action_row_sampled = []

        for i in sampled_index[ia, :]:
            if mask[ia, i] > 0:
                action_row_sampled.append(action_row[i])
                pass
            pass
        actions_sampled.append(action_row_sampled)
        pass

    return actions_sampled