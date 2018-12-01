import numpy as np
import random
import neuralnetworks as nn
from board import *
import copy

def epsilonGreedy(Qnet, state, epsilon):
    moves = state.validMoves()
    if np.random.uniform() < epsilon:
        move = moves[np.random.choice(range(len(moves)))] #take the randome move
        X = state.stateMoveVectorForNN(move)
        # without this, we give it a list. Needs an np array.
        X = np.array(X)
        # expects a 2d array. We want one row of a that array, so reshape first.
        X = X.reshape(1, 68)
        Q = Qnet.use(X) if Qnet.Xmeans is not None else 0 #checks to see if initialized
    else:
        qs = []
        for m in moves:
            X = state.stateMoveVectorForNN(m)
            X = np.array(X)
            X = X.reshape(1, 68)
            qs.append(Qnet.use(X)) if Qnet.Xmeans is not None else 0
        move = moves[np.argmax(qs)]
        Q = np.max(qs)
    return move, Q


## removed validmovesf and makemove f
def trainQnet(nBatches, nRepsPerBatch, hiddenLayers, nIterations, nReplays, epsilon, epsilonDecayFactor):
    outcomes = np.zeros(nBatches*nRepsPerBatch) # holds number of steps to victory. Should hold the number of outcome, win loss or draw
    ##create a 68 to one mapping.
    ##Also, hiddenlayers can be any structure..
    Qnet = nn.NeuralNetwork(68, hiddenLayers, 1)
    Qnet._standardizeT = lambda x: x
    Qnet._unstandardizeT = lambda x: x
    samples = []
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
                r = 0 # step reinforcement should be zero, like in tic tac toe.
                # Choose move from nextState. This part isn't exactly correct I think.
                Qnext = None
                gameOver, winner = stateNext.isOver()  # returns boolean, [None|COLOR] tuple
                # Now check to see if the game is over. Red just played,
                # so we shouldn't need to check if red is the winner.
                if gameOver and winner == Color.RED:  # GG, red won
                    # goal found. Q is one for winners
                    Qnext = 1
                    done = True
                    outcomes[repk] = 1
                    if rep % 10 == 0 or rep == nRepsPerBatch - 1:
                        print('Red won: batch={:d} rep={:d} epsilon={:.3f} steps={:d}'.format(batch, repk, epsilon,
                                                                                     int(outcomes[repk])), end=', ')
                elif gameOver:  # state.draw()???
                    # add a 0 for a draw
                    Qnext = 0
                    outcomes[repk] = 0
                    done = True
                    if rep % 10 == 0 or rep == nRepsPerBatch - 1:
                        print('batch={:d} rep={:d} epsilon={:.3f} steps={:d}'.format(batch, repk, epsilon,
                                                                                     int(outcomes[repk])), end=', ')
                else:
                    # blacks turn
                    # choose a random choice for black.
                    blackMoves = stateNext.validMoves()
                    moveBlack = blackMoves[np.random.choice(range(len(blackMoves)))]
                    stateNextBlack = copy.copy(stateNext)
                    stateNextBlack.makeMove(moveBlack)
                    gameOver, winner = stateNextBlack.isOver()
                    if gameOver:  # BG, red lost
                        Qnext = -1  # <-  negative reinforcement for loss
                        outcomes[repk] = -1
                        done = True
                        if rep % 10 == 0 or rep == nRepsPerBatch - 1:
                            print('Black won: batch={:d} rep={:d} epsilon={:.3f} steps={:d}'.format(batch, repk, epsilon,
                                                                                     int(outcomes[repk])), end=', ')
                # not sure what else to do....
                # At this point, were back at red's turn and can get the q from epsilon greedy if not found
                if Qnext is None:
                    moveNext, Qnext = epsilonGreedy(Qnet, stateNext, epsilon)
                else:
                    if len(stateNext.validMoves()) > 0:
                        moveNext, _ = epsilonGreedy(Qnet, stateNext, epsilon)
                    else:
                        moveNext = ((0, 0), [(0, 0)])  #placeholder, really there isn't a next move in this case because we lost.

                samples.append([*state.stateMoveVectorForNN(move), r, Qnext])
                # Don't worry about what this does, not 100% necessary
                samplesNextStateForReplay.append([*stateNext.stateMoveVectorForNN(moveNext), *moveNext])

                state = copy.deepcopy(stateNext)
                move = copy.deepcopy(moveNext)

            # Train on samples collected from batch.
            npsamples = np.array(samples)
            X = npsamples[:, :68]  # somethings wrong with the dimensionality. Instead of n by n np.array, its 1D list of rows. Fix tomorrow!
            T = npsamples[:, 68:69] + npsamples[:, 69:70]  # adds reinforcement plus Q
            Qnet.train(X, T, nIterations, verbose=False)

            # Experience Replay: Train on recent samples with updates to Qnext.
            # Not 100% needed, could just use the top part.
            #samplesNextStateForReplay = np.array(samplesNextStateForReplay)
            for replay in range(nReplays):
                # for sample, stateNext in zip(samples, samplesNextStateForReplay):
                # moveNext, Qnext = epsilonGreedy(Qnet, stateNext, epsilon, validMovesF)
                # sample[6] = Qnext
                # print('before',samples[:5,6])
                QnextNotZero = npsamples[:, 6] != 0
                npsamples[QnextNotZero, 6:7] = Qnet.use(samplesNextStateForReplay[QnextNotZero, :])
                # print('after',samples[:5,6])
                T = npsamples[:, 5:6] + npsamples[:, 6:7]
                Qnet.train(X, T, nIterations, verbose=False)

    print('DONE')
    return Qnet, outcomes, samples
