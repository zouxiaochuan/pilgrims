import copy

candidates_list = [
    list(range(4)),
    list(range(10))
]

x_map = [1, 1, -1, -1]

def check_back(plan):
    if len(plan) == 0:
        return False
    
    pos = [0, 0]
    for i, p in enumerate(plan):
        aidx = i % 2

        if aidx == 0:
            pp = p
        else:
            pp = plan[i-1]
            pass
        pos[1-(pp % 2)] += x_map[pp]
        pass
    
    if pos[pp % 2] == 0:
        # print(plan)
        return True

    return False


def plan_recur(cur, length):
    global candidates_list

    if len(cur) == length:
        if check_back(cur):
            return 1
        else:
            return 0
    else:
        cand_idx = len(cur) % 2
        candidates = copy.copy(candidates_list[cand_idx])

        if cand_idx == 0 and len(cur) > 0:
            candidates.remove(cur[-2])
            pass

        num = 0
        for cand in candidates:
            cur.append(cand)
            num += plan_recur(cur, length)
            cur.pop()
            pass

        return num
    pass

def all_plans(length=5):
    num = 0
    for i in range(length):
        num += plan_recur([], i+1)
        pass

    return num


if __name__ == '__main__':
    print(all_plans(8))