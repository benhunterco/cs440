
# coding: utf-8

# # Reinforcement Learning Solution to the Towers of Hanoi Puzzle

# For this assignment, you will use reinforcement learning to solve the [Towers of Hanoi](https://en.wikipedia.org/wiki/Tower_of_Hanoi) puzzle.  
# 
# To accomplish this, you must modify the code discussed in lecture for learning to play Tic-Tac-Toe.  Modify the code  so that it learns to solve the three-disk, three-peg
# Towers of Hanoi Puzzle.  In some ways, this will be simpler than the
# Tic-Tac-Toe code.  
# 
# Steps required to do this include the following:
# 
#   - Represent the state, and use it as a tuple as a key to the Q dictionary.
#   - Make sure only valid moves are tried from each state.
#   - Assign reinforcement of $1$ to each move, even for the move that results in the goal state.
# 
# Make a plot of the number of steps required to reach the goal for each
# trial.  Each trial starts from the same initial state.  Decay epsilon
# as in the Tic-Tac-Toe code.

# ## Requirements

# First, how should we represent the state of this puzzle?  We need to keep track of which disks are on which pegs. Name the disks 1, 2, and 3, with 1 being the smallest disk and 3 being the largest. The set of disks on a peg can be represented as a list of integers.  Then the state can be a list of three lists.
# 
# For example, the starting state with all disks being on the left peg would be `[[1, 2, 3], [], []]`.  After moving disk 1 to peg 2, we have `[[2, 3], [1], []]`.
# 
# To represent that move we just made, we can use a list of two peg numbers, like `[1, 2]`, representing a move of the top disk on peg 1 to peg 2.

# Now on to some functions. Define at least the following functions. Examples showing required output appear below.
# 
#    - `printState(state)`: prints the state in the form shown below
#    - `validMoves(state)`: returns list of moves that are valid from `state`
#    - `makeMove(state, move)`: returns new (copy of) state after move has been applied.
#    - `trainQ(nRepetitions, learningRate, epsilonDecayFactor, validMovesF, makeMoveF)`: train the Q function for number of repetitions, decaying epsilon at start of each repetition. Returns Q and list or array of number of steps to reach goal for each repetition.
#    - `testQ(Q, maxSteps, validMovesF, makeMoveF)`: without updating Q, use Q to find greedy action each step until goal is found. Return path of states.
# 
# A function that you might choose to implement is
# 
#    - `stateMoveTuple(state, move)`: returns tuple of state and move.  
#     
# This is useful for converting state and move to a key to be used for the Q dictionary.
# 
# Show the code and results for testing each function.  Then experiment with various values of `nRepetitions`, `learningRate`, and `epsilonDecayFactor` to find values that work reasonably well, meaning that eventually the minimum solution path of seven steps is found consistently.
# 
# Make a plot of the number of steps in the solution path versus number of repetitions. The plot should clearly show the number of steps in the solution path eventually reaching the minimum of seven steps, though the decrease will not be monotonic.  Also plot a horizontal, dashed line at 7 to show the optimal path length.
# 
# Add markdown cells in which you describe the Q learning algorithm and your implementation of Q learning as applied to the Towers of Hanoi problem.  Use at least 15 sentences, in one or more markdown cells.

# Printing methods! Struggdling for each int to be at the bottom. Can get them into rows successfully though. ??? I'm going to need some more calculations based on the length of the lists I guess....
# 

# In[1]:


import copy
#Prints based on how many rows there are.
def printState(state):
    printState = copy.deepcopy(state)
    for s in printState:
        while len(s) < 3:
            s.insert(0, " ")
    for i in range(3):
        print(printState[0][i],printState[1][i],printState[2][i], sep = " ")
    print("-----")


# TUPLENATE? Looks like it works correctly, returns a tuple of tuples.

# In[2]:


def stateMoveTuple(state, move):
    statetup = tuple([tuple(l) for l in state])
    movetup = tuple(move)
    return(statetup, movetup)


# In[3]:


stateMoveTuple([[1,2,3],[],[]], [1,2])


# make a move. Looks like this works fine too, note how we need to subtract from move[1] in order to make the correct move (second post is index one). 

# In[4]:


import copy as cp
def makeMove(state, move):
    newState = cp.deepcopy(state)
    #um, this isn't at all how this should work...
    #I seriously don't know How I even wrote this or why I would think this was correct
    #for l in newState:
    #    if move[0] in l:
    #        l.remove(move[0])
    #        newState[move[1] - 1].append(move[0])
    disk = newState[move[0] - 1].pop(0)
    newState[move[1] - 1].insert(0, disk)
    return newState


# Finds the valid moves from a given state by, stuff

# In[5]:


def nullCheck(state, i, j):
    if len(state) > i and len(state[i]) > j:
        return True
    else:
        return False
def validMoves(state):
    retList = []
    #Pretty much just going to check each possible move. 
    #Don't see an easier way to do it, or a reason why. 
    if nullCheck(state, 0, 0):
        if not nullCheck(state, 1, 0) or state[0][0] < state[1][0]:
            retList.append([1,2])
        if not nullCheck(state, 2, 0) or state[0][0] < state[2][0]:
            retList.append([1,3])
    if nullCheck(state, 1, 0):
        if not nullCheck(state, 0, 0) or state[1][0] < state[0][0]:
            retList.append([2,1])
        if not nullCheck(state, 2, 0) or state[1][0] < state[2][0]:
            retList.append([2,3])
    if nullCheck(state, 2, 0):
        if not nullCheck(state, 0, 0) or state[2][0] < state[0][0]:
            retList.append([3,1])
        if not nullCheck(state, 1, 0) or state[2][0] < state[1][0]:
            retList.append([3,2])
    return retList


# In[6]:


state = [[1,2,3],[],[]]
printState(state)
moves = validMoves(state)
print(moves)
state = makeMove(state, moves[1])
printState(state)
moves = validMoves(state)
print(moves)
state = makeMove(state, moves[0])
printState(state)
moves = validMoves(state)
print(moves)
state = makeMove(state,moves[2])
printState(state)
moves = validMoves(state)
print(moves)
state = makeMove(state, moves[0])
printState(state)
moves  = validMoves(state)
print(moves)
state = makeMove(state, moves[0])
printState(state)
moves  = validMoves(state)
print(moves)
state = makeMove(state, moves[2])
printState(state)
moves  = validMoves(state)
print(moves)
state = makeMove(state, moves[1])
printState(state)
moves  = validMoves(state)
print(moves)


# Epsilon greedy function. Just a cleaner way of writing trainQ. Either chooses a random move, or based on Q

# In[8]:


import numpy as np
def epsilonGreedy(epsilon, Q, state):
    moveList = validMoves(state)
    if np.random.uniform() < epsilon:
        #This is the random move, just choose one of our valid moves
        choice = np.random.choice(range(len(moveList)))
        return moveList[choice]
    else:
        # Use our Q function to greedily pick the next move. 
        statesAndMovesTuples = [stateMoveTuple(state, move) for move in moveList]
        #Theoretically, we want to choose the lowest Q, so if we haven't seen it
        #assume that it isn't good: its value is inf. 
        #This is reflected in choosing argmin, not argmax. 
        Qs = np.array([Q.get(tup, float('inf')) for tup in statesAndMovesTuples])
        return moveList[np.argmin(Qs)]


# In[12]:



state = [[1,2,3],[],[]]
print(validMoves(state))
Q = {}
#Q[stateMoveTuple(state, [1,3])] = 0
#epsilonGreedy(0, Q, state)


# This is the Q training algorithm from the tictactoe version.

# In[10]:


def trainQ(nRepetitions, learningRate, epsilonDecayFactor, validMovesF, makeMoveF, showMoves = False):
    Q = {}
    #Dictionary to represent our Q function. 
    #MakeTuple of the state can get the unique value.
    numberSteps = []
    epsilon = 1.0
    for trial in range(nRepetitions):
        epsilon *= epsilonDecayFactor
        step = 0
        state = [[1,2,3],[],[]] #Default starting state.
        done = False
        
        while not done:
            step += 1 #not sure what this does. Reinforcement?
            #choose a move from the current state
            move = epsilonGreedy(epsilon, Q, state)
            #make a new state from the greedy move.
            newState = makeMoveF(state, move)
           
            #if showMoves:
            #    printState(newState)
            Tup = stateMoveTuple(state, move)
            newTup = stateMoveTuple(newState, move)
            #print(step, newTup)
            if Tup not in Q:
                Q[Tup] = 0 #initiallizes unseen Qs
            
            if newState == [[],[],[1,2,3]]: #this means we found the solution.
                if showMoves:
                    print("Found solution in " + str(step))
                Q[Tup] = 1
                Q[newTup] = 0
                done = True
                numberSteps.append(step)
                
            #Do not need the move for O like in tic tac toe, because this is not adversarial.
            
            if step > 1: #Don't do this on first step, got it. 
                #Also, old state and move are set in the first term, so other we can reference them here.
                #Q[(tuple(boardOld),moveOld)] += rho * (Q[(tuple(board),move)] - Q[(tuple(boardOld),moveOld)])
                #print(step, newTup)
                #print(stateMoveTuple(stateOld, moveOld))
                Q[(stateMoveTuple(stateOld, moveOld))] += learningRate * (1 + Q[Tup] -
                                                                         Q[stateMoveTuple(stateOld, moveOld)])
            
            #set old variables
            stateOld, moveOld = state, move
            state = newState
    return Q, numberSteps
            


# In[21]:


def testQ(Q, maxSteps, validMovesF, makeMoveF):
    state = [[1,2,3],[],[]]
    path = []
    path.append(state)
    for i in range(maxSteps):
        state = makeMove(state, epsilonGreedy(0, Q, state)) # 0 ensures we always take greedy Q
        path.append(state)
        if state == [[],[],[1,2,3]]:
            return path
        


# # Examples

# In[13]:


state = [[1],[2],[3]]
printState(state)
state = [[1,2], [], [3]]
printState(state)
state = [[],[1],[2,3]]
printState(state)
state = [[1, 2, 3], [], []]
printState(state)


# In[14]:


move =[1, 2]

stateMoveTuple(state, move)


# In[15]:


newstate = makeMove(state, move)
newstate


# In[16]:


printState(newstate)


# In[17]:


Q, stepsToGoal = trainQ(50, 0.5, 0.7, validMoves, makeMove)


# In[18]:


Q


# In[19]:


stepsToGoal


# In[22]:


path = testQ(Q, 20, validMoves, makeMove)


# In[214]:


path


# In[23]:


for s in path:
    printState(s)
    print()


# ## Grading

# Download and extract `A4grader.py` from [A4grader.tar](http://www.cs.colostate.edu/~anderson/cs440/notebooks/A4grader.tar).

# In[227]:


get_ipython().run_line_magic('run', '-i A4grader.py')


# ## Extra Credit

# Modify your code to solve the Towers of Hanoi puzzle with 4 disks instead of 3.  Name your functions
# 
#     - printState_4disk
#     - validMoves_4disk
#     - makeMove_4disk
# 
# Find values for number of repetitions, learning rate, and epsilon decay factor for which trainQ learns a Q function that testQ can use to find the shortest solution path.  Include the output from the successful calls to trainQ and testQ.
