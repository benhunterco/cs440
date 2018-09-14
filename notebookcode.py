
# coding: utf-8

# # Assignment 2: Iterative-Deepening Search

# Ben Newell

# ## Overview

# Implement the iterative-deepening search algorithm as discussed in our Week 2 lecture notes and as shown in figures 3.17 and 3.18 in our text book. Apply it to the 8-puzzle and a second puzzle of your choice. 

# ## Required Code

# In this jupyter notebook, implement the following functions:
# 
#   * `iterativeDeepeningSearch(startState, goalState, actionsF, takeActionF, maxDepth)`
#   * `depthLimitedSearch(startState, goalState, actionsF, takeActionF, depthLimit)`
#   
# `depthLimitedSearch` is called by `iterativeDeepeningSearch` with `depthLimit`s of $0, 1, \ldots, $ `maxDepth`. Both must return either the solution path as a list of states, or the strings `cutoff` or `failure`.  `failure` signifies that all states were searched and the goal was not found. 
# 
# Each receives the arguments
# 
#   * the starting state, 
#   * the goal state,
#   * a function `actionsF` that is given a state and returns a list of valid actions from that state,
#   * a function `takeActionF` that is given a state and an action and returns the new state that results from applying the action to the state,
#   * either a `depthLimit` for `depthLimitedSearch`, or `maxDepth` for `iterativeDeepeningSearch`.

# Use your solution to solve the 8-puzzle.
# Implement the state of the puzzle as a list of integers. 0 represents the empty position. 
# 
# Required functions for the 8-puzzle are the following.
# 
#   * `findBlank_8p(state)`: return the row and column index for the location of the blank (the 0 value).
#   * `actionsF_8p(state)`: returns a list of up to four valid actions that can be applied in `state`. Return them in the order `left`, `right`, `up`, `down`, though only if each one is a valid action.
#   * `takeActionF_8p(state, action)`: return the state that results from applying `action` in `state`.
#   * `printPath_8p(startState, goalState, path)`: print a solution path in a readable form.  You choose the format.

# <font color='red'>Also</font>, implement a second search problem of your choice.  Apply your `iterativeDeepeningSearch` function to it.

# Insert your function definitions in this notebook.

# Here are some example results.

# ## Funcitons

# In[17]:


def findBlank_8p(state):
    # find index of 0
    index = state.index(0)
    #return modulo and python op // for row and column
    return index // 3, index % 3


# In[18]:


def printState_8p(state):
    state = state.copy()
    state[state.index(0)] = "-"
    #make each line as its own list
    l1,l2,l3 = state[:3], state[3:6], state[6:]
    #print theses lists seperated by a new line
    print(*[l1,l2,l3], sep = "\n")
    return


# In[19]:


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


# In[20]:


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


# In[21]:


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


# In[22]:


def iterativeDeepeningSearch(startState, goalState, actionsF, takeActionF, maxDepth):
    for depth in range(0, maxDepth):
        result = depthLimitedSearch(startState, goalState, actionsF, takeActionF, depth)
        if result == "failure":
            return "failure"
        if result != "cutoff":
            result.insert(0, startState)
            return result
    return "cutoff"


# In[23]:


def printPath_8p(startState, goalState, path):
    pass


# ## Maze Functions

# In[24]:


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


# In[25]:


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


# In[26]:


def printMaze_10(state):
    for i in range(0,10):
        print(*state[i*10:(i+1)*10], sep = " ")


# In[34]:


import random
def regenerateMaze(state):
    state = [random.sample(['x','-','-'],1)[0] for _ in range(0,100)] 
    return state


# In[32]:


def printMazePath(result):
    path = result[0].copy()
    for i in result:
        path[i.index("O")] = "~"
    printMaze_10(path)


# ## Testing 8p

# In[191]:


startState = [1, 0, 3, 4, 2, 5, 6, 7, 8]


# In[192]:


printState_8p(startState)  # not a required function for this assignment, but it helps when implementing printPath_8p


# In[193]:


assert(findBlank_8p(startState) == (0,1))
assert(findBlank_8p([1,2,3,0,5,6,7,8,4]) == (1,0))
assert(findBlank_8p([1,2,3,8,5,6,7,4,0]) == (2,2))
assert(findBlank_8p([1,2,3,8,0,6,7,4,5]) == (1,1))
print("All tests passed for findBlank_8p")


# In[194]:


for action in actionsF_8p(startState):
    print(action)


# In[195]:


actionList = list(actionsF_8p(startState))
assert(actionList == ['left', 'right', 'down'])
bottomRight, bottomLeft = [1,2,3,8,5,6,7,4,0], [1,2,3,8,5,6,0,4,7]
topRight, topLeft = [1,2,0,8,5,6,7,4,1],[0,2,3,8,5,6,7,4,1]
center = [1,2,3,4,0,5,6,7,8]
actionList = list(actionsF_8p(bottomRight))
assert(actionList == ['left', 'up'])
actionList = list(actionsF_8p(bottomLeft))
assert(actionList == ['right', 'up'])
actionList = list(actionsF_8p(topRight))
assert(actionList == ['left', 'down'])
actionList = list(actionsF_8p(topLeft))
assert(actionList == ['right', 'down'])
actionList = list(actionsF_8p(center))
assert(actionList == ['left', 'right', 'up', 'down'])


# In[196]:


takeActionF_8p(startState, 'down')


# In[197]:


printState_8p(startState)
print("Moves down to")
printState_8p(takeActionF_8p(startState, 'down'))


# In[198]:


goalState = takeActionF_8p(startState, 'down')


# In[199]:


newState = takeActionF_8p(startState, 'down')


# In[200]:


newState == goalState


# In[201]:


startState


# In[202]:


path = depthLimitedSearch(startState, goalState, actionsF_8p, takeActionF_8p, 3)
path


# Notice that `depthLimitedSearch` result is missing the start state.  This is inserted by `iterativeDeepeningSearch`.
# 
# But, when we try `iterativeDeepeningSearch` to do the same search, it finds a shorter path!

# In[203]:


path = iterativeDeepeningSearch(startState, goalState, actionsF_8p, takeActionF_8p, 3)
path


# Also notice that the successor states are lists, not tuples.  This is okay, because the search functions for this assignment do not

# In[204]:


startState = [4, 7, 2, 1, 6, 5, 0, 3, 8]
path = iterativeDeepeningSearch(startState, goalState, actionsF_8p, takeActionF_8p, 3)
path


# In[205]:


startState = [4, 7, 2, 1, 6, 5, 0, 3, 8]
path = iterativeDeepeningSearch(startState, goalState, actionsF_8p, takeActionF_8p, 5)
path


# Humm...maybe we can't reach the goal state from this state.  We need a way to randomly generate a valid start state.

# In[206]:


import random


# In[207]:


random.choice(['left', 'right'])


# In[208]:


def randomStartState(goalState, actionsF, takeActionF, nSteps):
    state = goalState
    for i in range(nSteps):
        l = list(actionsF(state))
        state = takeActionF(state, random.choice(l))
    return state


# In[209]:


goalState = [1, 2, 3, 4, 0, 5, 6, 7, 8]
randomStartState(goalState, actionsF_8p, takeActionF_8p, 10)


# In[210]:


startState = randomStartState(goalState, actionsF_8p, takeActionF_8p, 50)
startState


# In[211]:


path = iterativeDeepeningSearch(startState, goalState, actionsF_8p, takeActionF_8p, 20)
path


# Let's print out the state sequence in a readable form.

# In[212]:


for p in path:
    printState_8p(p)
    print()


# Here is one way to format the search problem and solution in a readable form.

# In[213]:


printPath_8p(startState, goalState, path)


# ## Testing the maze

# First, we have our start and end mazes

# In[11]:


startState = ['O', 'x', '-', '-', '-', 'x', '-', 'x', 'x','-',
 '-', '-', '-', '-', 'x', '-', '-',
 '-', 'x', '-', 'x', 'x', 'x', '-', '-',
 '-', 'x', 'x', '-', 'x', 'x', 'x', '-',
 '-', '-', '-', '-', '-', 'x', 'x', 'x',
 '-', 'x', 'x', 'x', '-', 'x',
 '-', 'x', '-', '-', 'x', '-', 'x', 'x',
 '-', '-', '-', '-', 'x', '-', '-', '-',
 'x', '-', '-', '-', '-', '-', 'x', 'x',
 '-', 'x', 'x', '-', '-', '-', 'x', '-',
 '-', 'x', '-', 'x', '-', '-', 'x', 'x',
 '-', '-', 'x', 'x', 'x', '-', 'x', 'x',
 '-', '-', 'x', '-', '-']
printMaze_10(startState)
goalState = startState.copy()
goalState[0] = "-"
goalState[99] = "O"
print("The Goal: ")
printMaze_10(goalState)


# Some verification

# In[13]:


list(actionsF_maze(startState))


# In[15]:


downone = takeActionF_maze(startState,"down")
printMaze_10(downone)


# In[16]:


list(actionsF_maze(downone))


# In[29]:


result = iterativeDeepeningSearch(startState, goalState, actionsF_maze, takeActionF_maze, 20)


# In[33]:


printMazePath(result)


# In[35]:


startState = regenerateMaze(startState)
printMaze_10(startState)


# In[41]:


startState[92] = "-"
goalState = startState.copy()
startState[92] = "O"
printMaze_10(startState)
goalState[97] = "O"
print("goal: ")
printMaze_10(goalState)


# In[52]:


result = iterativeDeepeningSearch(startState, goalState, actionsF_maze, takeActionF_maze, 20)
if result != "cutoff" and result != "failure":
    printMazePath(result)


# Download [A2grader.tar](A2grader.tar) and extract A2grader.py from it.

# In[214]:


get_ipython().run_line_magic('run', '-i A2grader.py')


# ## Extra Credit
# 
# For extra credit, apply your solution to the grid example in Assignment 1 with the addition of a horizontal and vertical barrier at least three positions long.  Demonstrate the solutions found in four different pairs of start and goal states.
