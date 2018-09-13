def findBlank_8p(state):
    # find index of 0
    index = state.index(0)
    #return modulo and python op // for row and column
    return index // 3, index % 3

def printState_8p(state):
    state = state.copy()
    state[state.index(0)] = "-"
    #make each line as its own list
    l1,l2,l3 = state[:3], state[3:6], state[6:]
    #print theses lists seperated by a new line
    print(*[l1,l2,l3], sep = "\n")
    return

def actionsF_8p(state):
    i = state.index(0)
    if i % 3 > 0:
        yield "left"
    if i % 3 < 2:
        yield "right"
    if i // 3 > 0:
        yield "up"
    if i // 3 < 2:
        yield "down"

def takeActionF_8p(state, action):
    #this does not check if action is allowed
    state = state.copy()
    i = state.index(0)
    if action == "right":
        state[i], state[i+1] = state[i+1], state[i]
    elif action == "left":
        state[i], state[i-1] = state[i-1], state[i]
    elif action == "up":
        state[i], state[i-3] = state[i-3], state[i]
    elif action == "down":
        state[i], state[i+3] = state[i+3], state[i]
    return state

def depthLimitedSearch(state, goalState, actionsF, takeActionF, depthLimit):
    if state == goalState:
        return []
    if depthLimit == 0:
        return "cutoff"
    cutoffoccurred = False
    for action in actionsF(state):
        childState = takeActionF(state, action)
        result = depthLimitedSearch(childState, goalState, actionsF, takeActionF, depthLimit - 1)
        if result == "cutoff":
            cutoffoccurred = True
        elif result != "failure":
            result.insert(0, childState)
            return result
        if cutoffoccurred:
            return "cutoff"
        else:
            return "failure"

def iterativeDeepeningSearch(startState, goalState, actionsF, takeActionF, maxDepth):
    for depth in range(0, maxDepth):
        result = depthLimitedSearch(startState, goalState, actionsF, takeActionF, depth)
        if result == "failure":
            return "failure"
        if result != "cutoff":
            result.insert(0, startState)
            return result
    return "cutoff"



import random

def actionsF_maze(state):
    i = state.index("O")
    if i % 10 > 0 and state[i-1] != "x":
        yield "left"
    if i % 10 < 9 and state[i+1] != "x":
        yield "right"
    if i // 10 > 0 and state[i-10] != "x":
        yield "up"
    if i // 10 < 9 and state[i+10] != "x":
        yield "down"

def takeActionF_maze(state, action):
    #this does not check if action is allowed
    state = state.copy()
    i = state.index("O")
    if action == "right":
        state[i], state[i+1] = state[i+1], state[i]
    elif action == "left":
        state[i], state[i-1] = state[i-1], state[i]
    elif action == "up":
        state[i], state[i-10] = state[i-10], state[i]
    elif action == "down":
        state[i], state[i+10] = state[i+10], state[i]
    return state

def printMaze_10(state):
    for i in range(0,10):
        print(*state[i*10:(i+1)*10], sep = " ")

def regenerateMaze(state):
    state = [random.sample(['x','-','-'],1)[0] for _ in range(0,100)] 
    return state

def printMazePath(result):
    path = result[0].copy()
    for i in result:
        path[i.index("O")] = "~"
    printMaze_10(path)

    
#create a random start state, not guaranteed to be solvable
startState = [random.sample(['x','-','-'],1)[0] for _ in range(0,100)] 
#our goal will be to move the circle to the bottom right of the maze.
goalState = startState.copy()
startState[0], goalState[99] = 'O','O'







