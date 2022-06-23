import torch


def masked_softmax(scores, mask, dim):
    '''masked softmax
    '''
    scores_exp = torch.exp(scores)
    scores_exp = scores_exp * mask.float()
    scores_exp[torch.sum(scores_exp, dim=-1)==0, :] = 1
    scores_exp = scores_exp / (scores_exp.sum(dim=dim, keepdim=True) + 1e-8)
    return scores_exp


def sample_action(
        actions, scores: torch.Tensor, mask: torch.Tensor, num, temperature):
    '''sample actions according to scores

    Parameters:
        actions: action list of shape [B, A, N]
        scores: probability tensor of shape [B, A, N, D]
        mask: score mask of shape [B, A, N, D]
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
