import sys
from .agent_submit import AgentFromRL
from kaggle_environments.envs.kore_fleets.kore_fleets import miner_agent
import kaggle_environments
import os
import multiprocessing as mp


agent_model = AgentFromRL('save/model/20220709011542.pth', device='cuda:1')
# agent_other = AgentFromRL('save/model/20220708050155.pth', device='cuda:1')
agent_other = miner_agent
save_path = 'save/combat_fail/'


def play(param):
    i, cnt = param
    env = kaggle_environments.make("kore_fleets", debug=True)
    agents = [None, None]
    agents[i] = agent_model
    agents[1-i] = agent_other
    states = env.run(agents)

    r = [s.reward for s in states[-1]]

    if r[i] > r[i-1]:
        return 1
    else:
        res = env.render(mode="html", width=1000, height=800)
        with open(save_path + f'fail_{cnt+i:03d}.html', 'w') as f:
            f.write(res)
            pass
        return 0
    pass


if __name__ == '__main__':
    save_path = 'save/combat_fail/'

    

    os.system('rm -rf ' + save_path + '/*')
    win_cnt = 0
    cnt = 0
    mp.set_start_method('spawn')
    while True:
        env = kaggle_environments.make("kore_fleets", debug=True)
        pool = mp.Pool(processes=2)

        results = pool.map(play, [(0, cnt), (1, cnt)])

        for r in results:
            win_cnt += r
            pass
        cnt += 2

        print(f'model win: {win_cnt}, total: {cnt}, rate: {win_cnt / cnt}')
        pass
        
    pass