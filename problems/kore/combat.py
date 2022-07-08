import sys
from .agent_submit import AgentFromRL
from kaggle_environments.envs.kore_fleets.kore_fleets import miner_agent
import kaggle_environments
import os


if __name__ == '__main__':
    save_path = 'save/combat_fail/'
    agent_model = AgentFromRL('save/model/20220708050155.pth', device='cuda:1')
    agent_other = AgentFromRL('save/model/20220707074330.pth', device='cuda:1')
    # agent_other = miner_agent

    os.system('rm -rf ' + save_path + '/*')
    win_cnt = 0
    cnt = 0
    while True:
        env = kaggle_environments.make("kore_fleets", debug=True)
        states = env.run([agent_model, agent_other])
        # print(states[-1])
        r = [s.reward for s in states[-1]]

        if r[0] > r[1]:
            win_cnt += 1
        else:
            res = env.render(mode="html", width=1000, height=800)
            with open(save_path + f'fail_{cnt:03d}.html', 'w') as f:
                f.write(res)
                pass
            pass

        cnt += 1

        print(f'model win: {win_cnt}, total: {cnt}, rate: {win_cnt / cnt}')
        pass
        
    pass