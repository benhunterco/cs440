
# coding: utf-8

# # Assignment 2: Iterative-Deepening Search

# Ben Newell

# ## Overview

# This project gives an implemintation of the iterative deepening function and depth limited search. These strategies  are applied to the 8p, 15p and simple maze problem. 
# 
# The 8 puzzle problem was discussed in class. 8 numbered pieces are shuffled across a board with only one free space to manipulate. By moving this free space in a series of steps, a goal state (usually having the numbers in order) can be reached. We represent it with a three by three grid, implemented as a list, where the 0 represents the blank space. The blank piece is then swapped with its neighbors to represent sliding across the board.  
# 
# The maze puzzle here is fairly simple. The goal is to find the path from one state to the other while navigating a grid with obstacles in the way. The problem space is a ten by ten grid where "-" represents a open space, "x" represents a blocked space, and "O" represents the current location of the maze solver. The "O" can only move left, right, up, or down.  The path is found by finding a path from the "O" in startState to the "O" goalState. When printed by `printMazePath` the path is designated by "~". 
# 
# Note: I realized at the end of the assignment the extra credit is mostly the same as the maze problem I came up with here. I had written the entire maze problem by the time I read the extra credit, so I'll treat that as extra credit and provide an implimentation for the 15 puzzle as well. The 15 puzzle is the same as the 8 puzzle, just with extra complexity. Instead of numbers 1 through 8, 1 through 15 are present on a four by four grid. 

# # Functions and Explanations

# ## Search Functions

# In[1]:


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


# `depthLimitedSearch` is a modification of depth FirstSearch that limits the maximum depth into the graph that the search can go. In this implementation we use a recursive strategy. `state`, `goalState`, `actionsF`, `takeActionF` and `depthLimit` are all passed into the function at each step of the recursion. First, it checks some base cases, that either we have found the goalState or that depthLimit has been reached. In this case we return [] or "cutoff" respectively. Otherwise, we set `cutoff` to false, indicating that another recursion can occur. For each action possible at this state, we call depthLimitedSearch with depthLimit-1 in order to continue the recursion. Then depending on the results of this call, we can either return cutoff, failure or the result path. 

# In[2]:


def iterativeDeepeningSearch(startState, goalState, actionsF, takeActionF, maxDepth):
    for depth in range(0, maxDepth):
        result = depthLimitedSearch(startState, goalState, actionsF, takeActionF, depth)
        if result == "failure":
            return "failure"
        if result != "cutoff":
            result.insert(0, startState)
            return result
    return "cutoff"


# `iterativeDeepeningSearch` is simple, it calls depthLimitedSearch at increasing depths until it reaches the max depth. If the result is a path, it prepends the startState to the path and returns it. It may also return "cutoff" indicating that the search did not find the goal, but unexplored nodes exist at further depth levels. Finally, a return of "failure" indicates that the all nodes have been searched and the result was not found. By searching in this manner, we can combine the low memory cost of depth first search with the other benefits of breadth first search. 

# ## 8 Puzzle Functions

# In[3]:


def findBlank_8p(state):
    # find index of 0
    index = state.index(0)
    #return modulo and python op // for row and column
    return index // 3, index % 3


# This function uses a strategy of indexing that most other functions in this project use. We can use integer division and modulo in python to treat the list as a grid. Index // x where x is the number of values in a row will return the row of index. This is because of the properties of integer devision. As an example, index 5 // 3 = 1, meaning it is in the second row. Similarly, % can be used to find the column of an index. 5 % 3 = 2, meaning it is in the third column. Using this strategy, we can return the row and the column of the index as if state was stored in an array.

# In[4]:


def printState_8p(state):
    state = state.copy()
    state[state.index(0)] = "-"
    #make each line as its own list
    l1,l2,l3 = state[:3], state[3:6], state[6:]
    #print theses lists seperated by a new line
    for l in [l1,l2,l3]:
        print(*l, sep = " ")
    return


# This function prints the state of an 8p in a readable fashion. 0 is replaced with "-" to represent the blank space. The list is broken up into three sub lists, each representing a row in the puzzle. Then we print each of these on a new line using some features of `print()` that allow the characters to appear without quotes and be seperated by a space. 

# In[5]:


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


# `actionsF_8p` is critical for iterative deepening search to function. Here it is implemented as a generator, so that actions are not found until they are needed. We determine the possible actions using // and %. Essentially, if the blank space is not up against a wall, it can move in that given direction. We check if it is against a wall with % for the left and right sides and // for the top and bottom. As an example, say the blank space is in the bottom left corner with an index of 6. We know that it can only move up and to the right logically. The function represents this because 6 % 3 = 0, so it will not yield "left". Similarly, it will not yield "down" because 6 // 3 = 2. In this way, the actions of a blank space can be calculated. 

# In[6]:


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


# Continuing on the list indexing system, `takeActionF_8p` takes the action on a state by swapping the blank in the direciton of the desired move. For going left and right, this is as simple as a +1 or a -1. In order to go up or down, the length of the row is subtracted or added. In this case the rows are length 3.

# In[7]:


def printPath_8p(startState, goalState, path):
    for state in path:
        printState_8p(state)
        print()
    printState_8p(goalState)


# This simply prints out the solution path by printing out each state in `path` using `printState_8p`. The goal state is then added to the end. The solution, then, is represented by each successive step the search took to find the goal. Unfortunately, there is not a compact way of showing the solution like in the maze below, because the location of the numbers changes at each step along with the blank space. 

# ## 15 Puzzle Functions

# In[8]:


def printState_15p(state):
    state = state.copy()
    state[state.index(0)] = "-"
    #make each line as its own list
    l1,l2,l3,l4 = state[:4], state[4:8], state[8:12], state[12:]
    #print theses lists seperated by a new line
    for l in [l1, l2, l3, l4]:
        print(*l, sep = " ")
    return


# This is implemented the same as the 8p, but instead we add the extra row of the 15p and adjust the indices to match. 

# In[9]:


def actionsF_15p(state):
    i = state.index(0)
    if i % 4 > 0:
        yield "left"
    if i % 4 < 3:
        yield "right"
    if i // 4 > 0:
        yield "up"
    if i // 4 < 3:
        yield "down"


# This uses the same logic as the 8p as well. Except here we must check `i` against four because the puzzle has four rows and four columns.

# In[10]:


def takeActionF_15p(state, action):
    #this does not check if action is allowed
    state = state.copy()
    i = state.index(0)
    if action == "right":
        state[i], state[i+1] = state[i+1], state[i]
    elif action == "left":
        state[i], state[i-1] = state[i-1], state[i]
    elif action == "up":
        state[i], state[i-4] = state[i-4], state[i]
    elif action == "down":
        state[i], state[i+4] = state[i+4], state[i]
    return state


# Once again the same, only we need to add or subtract four in order to go up or down a row in the 15p.

# In[11]:


def printPath_15p(startState, goalState, path):
    for state in path:
        printState_15p(state)
        print()
    printState_15p(goalState)


# This is identical to the 8p, but calls the correct printState function.

# ## Maze Functions

# In[12]:


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


# While this actions function is similar to the previous two, there are more conditions that must be met for a move to be possible. In addition to not going out of bounds, the action function must also avoid stepping into an "x". At each check the function looks to see whether the destination is out of bounds and then checks to see whether it is a valid move withing the maze construct. In this way we get an action function that behaves as if it was navigating through a maze by avoiding obstacles.

# In[13]:


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


# Because the actions function only creates valid moves, we do not need to check for anything here in the take action function. It behaves similarly to the previous two, only adjusted for a grid size of ten.

# In[14]:


def printMaze_10(state):
    for i in range(0,10):
        print(*state[i*10:(i+1)*10], sep = " ")


# This function is a reduced form of the previous two. Here we loop from 0 to 10 and print out a line for each. Each line prints the ten indices at that iteration of the loop. For example, on the second iteration, i = 1, we print `state[10:20]`. Each value is seperated by a space. As a result, we get a readable ten by ten representation of the maze. 

# In[15]:


import random
def generateMaze():
    state = [random.sample(['x','-','-'],1)[0] for _ in range(0,100)] 
    return state


# This function can be used to generate a random maze template. It generates a ten by ten grid where around 1/3 of the values are "x" and 2/3 of the values are "-". It is not guaranteed to be solvable, and the user must create the start and goal states by placing a "O" by hand. However, in my trials it usually created an interesting board to place starts and goals in. 

# In[16]:


def printMazePath(startState, goalState, path):
    printingPath = path[0].copy()
    for i in path:
        printingPath[i.index("O")] = "~"
    printingPath[startState.index("O")] = "S"
    printingPath[goalState.index("O")] = "G"
    print("Path from startState S to goalState G")
    printMaze_10(printingPath)


# Printing the maze solution is a little more flexible than the 8p or 15p solutions. Instead of printing a list of the steps, we can instead show the path of the solution directly. This works by iterating through the solution path and replacing the location of "O" in the printing list with "~". Then at the end we replace the start index of "O" with "S" to represent start, and the goal index of "O" with "G" to represent goal. At the end we are left with the solved maze drawn out in a readable form. 

# ## Testing 8p

# A list representation the 8p.

# In[17]:


startState = [1, 0, 3, 4, 2, 5, 6, 7, 8]
goalState = [1, 2, 3, 4, 0, 5, 6, 7, 8]


# Printed out by the printState function. Here we can visualize how the list is representing a grid or matrix even though it is one dimensional. 

# In[18]:


printState_8p(startState)


# Testing the `findBlank_8p` function. Although it is required, I stick to index and some division to implement my functions instead of using tuple indexing.

# In[19]:


assert(findBlank_8p(startState) == (0,1))
assert(findBlank_8p([1,2,3,0,5,6,7,8,4]) == (1,0))
assert(findBlank_8p([1,2,3,8,5,6,7,4,0]) == (2,2))
assert(findBlank_8p([1,2,3,8,0,6,7,4,5]) == (1,1))
print("All tests passed for findBlank_8p")


# Testing `actionsF_8p` using "corner" cases.

# In[20]:


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
print("Tests for actionsF_8p passed")


# Demonstrating moving the blank tile in the puzzle. This shows the state before and after the move.

# In[21]:


printState_8p(startState)
print("Moves down to")
printState_8p(takeActionF_8p(startState, 'down'))


# First, a quick demonstration of `depthLimitedSearch`.

# In[22]:


path = depthLimitedSearch(startState, goalState, actionsF_8p, takeActionF_8p, 3)
path


# As pointed out in the assignment, the `depthLimitedSearch` does not contain the start state.  This is inserted by `iterativeDeepeningSearch`.
# 
# When we use `iterativeDeepeningSearch` a shorter path with the start state present is found. We also gain the benefits of increasing the depth limit in an iterative fashion, so the goal may be found earlier.

# In[23]:


path = iterativeDeepeningSearch(startState, goalState, actionsF_8p, takeActionF_8p, 3)
path


# Here we demonstrate `iterativeDeepeningSearch` not finding the goal.

# In[24]:


startState = [4, 7, 2, 1, 6, 5, 0, 3, 8]
path = iterativeDeepeningSearch(startState, goalState, actionsF_8p, takeActionF_8p, 5)
path


# Here I have compacted the assignment code to generate random startStates. It works by taking a random choice from the actionsF of each state, so that we know it started from a valid start. 

# In[25]:


import random
def randomStartState(goalState, actionsF, takeActionF, nSteps):
    state = goalState
    for i in range(nSteps):
        l = list(actionsF(state))
        state = takeActionF(state, random.choice(l))
    return state


# In[26]:


goalState = [1, 2, 3, 4, 0, 5, 6, 7, 8]
randomStartState(goalState, actionsF_8p, takeActionF_8p, 10)


# In[27]:


startState = randomStartState(goalState, actionsF_8p, takeActionF_8p, 50)
startState


# Here we solve the randomly created startState and print it out using `printPath_8p` which prints out each step in a readable format. 

# In[28]:


path = iterativeDeepeningSearch(startState, goalState, actionsF_8p, takeActionF_8p, 20)
printPath_8p(startState, goalState, path)


# Looks like it worked! On to the next puzzle.

# ## Testing 15p

# Here is a sample start and goal state for the 15p problem.

# In[29]:


goalState = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
startState = [0, 2, 3, 4, 1, 6, 7, 8, 5, 14, 10, 11, 9, 13, 15, 12]


# In[30]:


print("startState: ")
printState_15p(startState) 
print("goalState: ")
printState_15p(goalState)


# First, we use assertions to test `actionsF_15p`

# In[31]:


assert(list(actionsF_15p(startState)) == ["right", "down"])  
assert(list(actionsF_15p(goalState)) == ["left", "up"])
print("Tests of actionsF_15p  Passed")


# Then, visually confirm that the desired moves are done by `takeActionF_15p`

# In[32]:


print("startState: ")
printState_15p(startState)
print("Moves down to")
printState_15p(takeActionF_15p(startState, 'down'))
print("and moves right to")
printState_15p(takeActionF_15p(startState, 'right'))


# Finally, a demonstration of solving the 15p with `iterativeDeepeningSearch`.

# In[33]:


path = iterativeDeepeningSearch(startState, goalState, actionsF_15p, takeActionF_15p, 15)
printPath_15p(startState, goalState, path)


# Looks like it works correctly for solvable 15 puzzles! What about an unsolvable 15p? This uses the simple [example](https://en.wikipedia.org/wiki/File:15-puzzle-loyd.svg) of an unsolvable puzzle from wikipedia. 

# In[34]:


startState = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 0]
goalState = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
print("startState: ")
printState_15p(startState)
print("goalState: ")
printState_15p(goalState)


# Here we see `iterativeDeepeningSearch` try to find a goal that cannot be found and reach a cutoff.

# In[35]:


iterativeDeepeningSearch(startState, goalState, actionsF_15p, takeActionF_15p, 15)


# Finally, a random startState.

# In[42]:


startState = randomStartState(goalState, actionsF_15p, takeActionF_15p, 20)
result = iterativeDeepeningSearch(startState, goalState,actionsF_15p, takeActionF_15p, 17)
printPath_15p(startState, goalState, result)


# This can be run several times to see different outcomes of iterative deepening search.

# ## Testing the maze

# First, we have our start and goal states of the maze. "O" will move about the maze similar to how "-" did in the 8 puzzle. 

# In[36]:


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
print("The Start: ")
printMaze_10(startState)
goalState = startState.copy()
goalState[0] = "-"
goalState[99] = "O"
print("The Goal: ")
printMaze_10(goalState)


# First, test the actions function.

# In[37]:


assert(list(actionsF_maze(startState)) == ['down'])
assert(list(actionsF_maze(goalState)) == ['left'])
print("Test Passed")


# Next, test the take action function from a few states.

# In[38]:


print("We expect \"O\" to have moved down a space")
downone = takeActionF_maze(startState,"down")
printMaze_10(downone)
print("We expect \"O\" to have moved right a space")
rightone = takeActionF_maze(downone, "right")
printMaze_10(rightone)


# An additional test of `actionsF_maze` with the new state we generated

# In[39]:


assert(list(actionsF_maze(downone)) == ['right', 'up'])
print("Test Passed")


# Finally, try out `iterativeDeepeningSearch` to see whether it works!

# In[40]:


result = iterativeDeepeningSearch(startState, goalState, actionsF_maze, takeActionF_maze, 20)
printMazePath(startState, goalState, result)


# Looks like the first startState and goalState worked! Next we can generate more cases to test.

# In[41]:


startState = generateMaze()
printMaze_10(startState)


# This created an empty template where the user can place an "O" for the startState and the goalState. It is not guaranteed to be solvable however. 

# In[42]:


goalState = startState.copy()
startState[0] = "O"
print("startState: ")
printMaze_10(startState)
goalState[0] = "-"
goalState[99] = "O"
print("goalState: ")
printMaze_10(goalState)


# You can usually eyeball whether the maze is solvable or not in a few seconds. Lets see how long `iterativeDeepeningSearch` takes to find out! You may run the previous cells over again to try a few different random states.

# In[43]:


result = iterativeDeepeningSearch(startState, goalState, actionsF_maze, takeActionF_maze, 20)
if result != "cutoff" and result != "failure":
    printMazePath(startState, goalState, result)
else:
    print(result + " at this Max Depth")


# It may take awhile if there is no solution. But it does demonstrate the "cutoff" part of `iterativeDeepeningSearch`.

# Here are a few pre-made solvable cases to demonstrate the maze solver. It is a messy block of hard coded lists, feel free to skip over. 

# In[44]:


start1 = ['-', '-', '-', '-', 'x', 'x', '-', '-', 'x', 'x', '-', '-', '-', 'x', 'x', 'x',
 '-', '-', 'x', '-', 'x', '-', '-', '-', '-', '-', 'x', '-', '-', '-', '-', '-',
 'x', 'x', 'x', '-', '-', '-', '-', '-', 'x', 'x', '-', '-', '-', '-', 'x', 'x',
 'x', 'x', 'x', '-', '-', '-', '-', '-', '-', 'x', '-', '-', '-', '-', '-', '-',
 '-', '-', '-', '-', 'x', '-', '-', '-', '-', '-', '-', 'x', '-', '-', '-', 'x',
 '-', 'x', '-', '-', 'x', '-', 'x', 'x', '-', 'x', '-', '-', 'O', '-', 'x', 'x',
 '-', '-', '-', 'x']
goal1 = ['-', '-', '-', '-', 'x', 'x', '-', '-', 'x', 'x', '-', '-', '-', 'x', 'x', 'x',
 '-', '-', 'x', '-', 'x', '-', '-', '-', '-', '-', 'x', '-', '-', '-', '-', '-',
 'x', 'x', 'x', '-', '-', '-', '-', '-', 'x', 'x', '-', '-', '-', '-', 'x', 'x',
 'x', 'x', 'x', '-', '-', '-', '-', '-', '-', 'x', '-', '-', '-', '-', '-', '-',
 '-', '-', '-', '-', 'x', '-', '-', '-', '-', '-', '-', 'x', '-', '-', '-', 'x',
 '-', 'x', '-', '-', 'x', '-', 'x', 'x', '-', 'x', '-', '-', '-', '-', 'x', 'x',
 '-', 'O', '-', 'x']
case2 = ['-', '-', '-', '-', '-', '-', '-', '-',
 '-', 'x', '-', 'x', 'x', '-', 'x', '-', '-', '-', 'x', '-', '-', 'x', 'x', '-',
 '-', '-', '-', '-', 'x', '-', '-', 'x', 'x', '-', 'x', '-', '-', 'x', 'x', '-',
 '-', '-', 'x', '-', 'x', '-', 'x', 'x', 'x', 'x', '-', '-', 'x', 'x', '-', '-',
 '-', 'x', '-', '-', 'x', 'x', 'x', '-', '-', '-', '-', '-', '-', '-', '-', '-',
 '-', '-', 'x', '-', 'x', '-', 'x', 'x', '-', '-', '-', 'x', '-', '-', '-', 'x',
 '-', 'x', '-', '-', '-', '-', 'x', '-', '-', '-', 'x', '-']
goal2 = case2.copy()
start2 = case2.copy()
start2[0] = "O"
goal2[90] = "O"
case3 = ['-', '-',
 '-', '-', '-', '-', '-', '-', '-', 'x', '-', 'x', 'x', '-', 'x', '-', '-',
 '-', 'x', '-', '-', 'x', 'x', '-', '-', '-', '-', '-', 'x', '-', '-', 'x', 'x',
 '-', 'x', '-', '-', 'x', 'x', '-', '-', '-', 'x', '-', 'x', '-', 'x', 'x', 'x',
 'x', '-', '-', 'x', 'x', '-', '-', '-', 'x', '-', '-', 'x', 'x', 'x', '-', '-',
 '-', '-', '-', '-', '-', '-', '-', '-', '-', 'x', '-', 'x', '-', 'x', 'x', '-',
 '-', '-', 'x', '-', '-', '-', 'x', '-', 'x', '-', '-', '-', '-', 'x', '-', '-',
 '-', 'x', '-']
goal3 = case3.copy()
start3 = case3.copy()
start3[51] = "O"
goal3[54] = "O"
case4 = ['x', '-', '-', '-', '-', '-', '-', '-', '-', 'x', '-', '-', 'x', 'x', 'x', '-',
 '-', 'x', '-', '-', 'x', '-', 'x', '-', '-', '-', 'x', 'x', '-', '-', '-', '-',
 '-', 'x', '-', '-', '-', '-', 'x', '-', 'x', '-', '-', '-', 'x', 'x', '-', '-',
 'x', 'x', '-', '-', '-', '-', '-', '-', 'x', '-', '-', '-', '-', 'x', '-', '-',
 'x', '-', '-', '-', 'x', 'x', 'x', '-', '-', '-', '-', 'x', '-', '-', '-', '-',
 '-', '-', '-', 'x', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'x', 'x', 'x',
 '-', 'x', 'x', '-']
goal4 = case4.copy()
start4 = case4.copy()
start4[39] = "O"
goal4[50] = "O"


# **Demonstration of Extra Credit**
# 
# Here we demonstrate four cases of the maze solver. More cases can be generated with the `generateMaze` function, which gives a ten by ten grid with a distribution of impassable x's and passable -'s. The start and goal locations must be placed by hand by setting the desired start index to "O" and the desired goal index to "O" in their respective lists. I've generated a few sample problems to demonstrate the maze solver. It is worth mentioning that although this is not exactly like the description of the extra credit, I came up with this problem and implemented it before I noticed the extra credit option. Because they are essentially the same (a maze is just multiple verticle and horizontal obstacles) I left it in. This took a little while to run on my laptop, about a minute.  

# In[45]:


cases = [(start1, goal1),(start2, goal2),(start3, goal3),(start4, goal4)]
for case in cases:
    result = iterativeDeepeningSearch(case[0], case[1], actionsF_maze, takeActionF_maze, 20)
    if result != "cutoff" and result != "failure":
        printMazePath(case[0], case[1], result)
    else:
        print(result + " at this Max Depth")


# download [A2grader.tar](A2grader.tar) and extract A2grader.py from it.

# ## Conclusion
# The `iterativeDeepeningSearch` function has been applied successfully to three different types of puzzles. While it is not always fast, it does work well in the examples demonstrated here. In the process of this project, I encountered a few problems. The 8p was fairly straightforward following the example given in the class notebook. However, it was difficult to think of other problems to try it out on. At first I thought of the maze problem, which does work well with this search strategy. The biggest problem I faced was that I read the extra credit late into the assignment and wanted to come up with an additional problem. While it was not too much work, implementing the 15p did stretch my time a little thin. Other than that, I did not have many other issues. 
