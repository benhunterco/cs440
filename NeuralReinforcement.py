import numpy as np
import random

def epsilonGreedy(Qnet, state, epsilon, validMovesF):
    moves = validMovesF(state)
    if np.random.uniform() < epsilon:
        move = moves[random.sample(range(len(moves)), 1)[0]] #take the randome move
        Q = Qnet.use(state.stateMoveVectorForNN(move))
    else:
        qs = []
        for m in moves:
            qs.append(Qnet.use(state.stateMoveVectorForNN(m)))
        move = moves[np.argmin(qs)]
        Q = np.min(qs)
    return move, Q