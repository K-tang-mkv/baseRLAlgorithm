
def select_action(state):
    if state[3]+state[2] < 0:
        return 0
    else:
        return 1