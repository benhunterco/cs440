import numpy as np
import random
import neuralnetworks as nn
from board import *
import copy

def epsilonGreedy(Qnet, state, epsilon):
    moves = state.validMoves()
    if np.random.uniform() < epsilon:
        move = moves[random.sample(range(len(moves)), 1)[0]] #take the randome move
        Q = Qnet.use(state.stateMoveVectorForNN(move)) if Qnet.Xmeans is not None else 0 #checks to see if initialized
    else:
        qs = []
        for m in moves:
            qs.append(Qnet.use(state.stateMoveVectorForNN(m))) if Qnet.Xmeans is not None else 0
        move = moves[np.argmax(qs)]
        Q = np.min(qs)
    return move, Q


## removed validmovesf and makemove f
def trainQnet(nBatches, nRepsPerBatch, hiddenLayers, nIterations, nReplays, epsilon, epsilonDecayFactor):
    outcomes = np.zeros(nBatches*nRepsPerBatch) # holds number of steps to victory.
    ##create a 68 to one mapping.
    ##Also, hiddenlayers can be any structure..
    Qnet = nn.NeuralNetwork(68, hiddenLayers, 1)
    Qnet._standardizeT = lambda x: x
    Qnet._unstandardizeT = lambda x: x

    # not neededsamples = []  # collect all samples for this repetition, then update the Q network at end of repetition.
    repk = -1  # not sure what this does, investigate maybe? Could be some sort of rep counter

    # Big batch, each of these creates something on which to train the Q network
    for batch in range(nBatches):
        # decay epsilon after first repitition, then cap it at .01
        if batch > 0:
            epsilon *= epsilonDecayFactor
            epsilon = max(0.01, epsilon)

        samples = []
        samplesNextStateForReplay = []

        for rep in range(nRepsPerBatch):
            repk += 1
            step = 0
            done = False

            state = Board()  # create a new board to represent the state
            move, _ = epsilonGreedy(Qnet, state, epsilon) #<- not necesssary
            # Red goes first!

            while not done:
                step += 1

                # Make this move to get to nextState. Find the board state for the next move.
                stateNext = copy.copy(state)
                stateNext.makeMove(move)

                # see if that move won and if so give reinforcement?
                # At this point its blacks turn, should I let black play or no?
                # stateNext = makeMoveF(state, move)  #! MUST change board.py to not change itself. Or copy into statenext an
                r = 1
                # Choose move from nextState. This part isn't exactly correct I think.


                if len(stateNext.validMoves()) == 0: #GG, red won
                    # goal found. Q is one for winners
                    Qnext = 1
                    done = True
                    outcomes[repk] = step
                    if rep % 10 == 0 or rep == nRepsPerBatch - 1:
                        print('batch={:d} rep={:d} epsilon={:.3f} steps={:d}'.format(batch, repk, epsilon,
                                                                                     int(outcomes[repk])), end=', ')
                elif False: # state.draw()???
                    #add a q of zero
                    pass
                else:
                    #blacks turn
                    # choose a random choice for black.
                    blackMoves = stateNext.validMoves()
                    moveBlack = np.random.choice(blackMoves)
                    stateNextBlack = copy.copy(stateNext)
                    stateNextBlack.makeMove(moveBlack)
                    if len(stateNextBlack.validMoves()) == 0: #BG, red lost
                        Qnext = 0
                        done = True
                    #not sure what else to do....




                moveNext, Qnext = epsilonGreedy(Qnet, stateNext, epsilon)
                samples.append([state.stateMoveVectorForNN(move), r, Qnext])
                samplesNextStateForReplay.append([stateNext.stateMoveVectorForNN(moveNext), *moveNext])

                state = copy.deepcopy(stateNext)
                move = copy.deepcopy(moveNext)

            # retraining.
            samples = np.array(samples)
            X = samples[:, :5]
            T = samples[:, 5:6] + samples[:, 6:7]
            Qnet.train(X, T, nIterations, verbose=False)

            # Experience Replay: Train on recent samples with updates to Qnext.
            # Not 100% needed, could just use the top part.
            samplesNextStateForReplay = np.array(samplesNextStateForReplay)
            for replay in range(nReplays):
                # for sample, stateNext in zip(samples, samplesNextStateForReplay):
                # moveNext, Qnext = epsilonGreedy(Qnet, stateNext, epsilon, validMovesF)
                # sample[6] = Qnext
                # print('before',samples[:5,6])
                QnextNotZero = samples[:, 6] != 0
                samples[QnextNotZero, 6:7] = Qnet.use(samplesNextStateForReplay[QnextNotZero, :])
                # print('after',samples[:5,6])
                T = samples[:, 5:6] + samples[:, 6:7]
                Qnet.train(X, T, nIterations, verbose=False)

        print('DONE')
        return Qnet, outcomes, samples
