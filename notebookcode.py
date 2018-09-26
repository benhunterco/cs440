
# coding: utf-8

# # A3: A\*, IDS, and Effective Branching Factor

# For this assignment, implement the Recursive Best-First Search
# implementation of the A\* algorithm given in class.  Name this function `aStarSearch`.  Also in this notebook include your `iterativeDeepeningSearch` functions.  Define a new function named `ebf` that returns an estimate of the effective
# branching factor for a search algorithm applied to a search problem.
# 
# So, the required functions are
# 
#    - `aStarSearch(startState, actionsF, takeActionF, goalTestF, hF)`
#    - `iterativeDeepeningSearch(startState, goalState, actionsF, takeActionF, maxDepth)`
#    - `ebf(nNodes, depth, precision=0.01)`, returns the effective branching factor, given the number of nodes expanded and depth reached during a search.
# 
# Apply `iterativeDeepeningSearch` and `aStarSearch` to several eight-tile sliding puzzle
# problems. For this you must include your implementations of these functions, from Assignment 2:
# 
#   * `actionsF_8p(state)`: returns a list of up to four valid actions that can be applied in `state`. With each action include a step cost of 1. For example, if all four actions are possible from this state, return [('left', 1), ('right', 1), ('up', 1), ('down', 1)].
#   * `takeActionF_8p(state, action)`: return the state that results from applying `action` in `state` and the cost of the one step,
#   
# plus the following function for the eight-tile puzzle:
# 
#   * `goalTestF_8p(state, goal)`
#   
# Compare their results by displayng
# solution path depth, number of nodes 
# generated, and the effective branching factor, and discuss the results.  Do this by defining the following function that prints the table as shown in the example below.
# 
#    - runExperiment(goalState1, goalState2, goalState3, [h1, h2, h3])
#    
# Define this function so it takes any number of $h$ functions in the list that is the fourth argument.

# ## Heuristic Functions
# 
# For `aStarSearch` use the following two heuristic functions, plus one more of your own design, for a total of three heuristic functions.
# 
#   * `h1_8p(state, goal)`: $h(state, goal) = 0$, for all states $state$ and all goal states $goal$,
#   * `h2_8p(state, goal)`: $h(state, goal) = m$, where $m$ is the Manhattan distance that the blank is from its goal position,
#   * `h3_8p(state, goal)`: $h(state, goal) = ?$, that you define.  It must be admissible, and not constant for all states.

# ## Comparison

# Apply all four algorithms (`iterativeDeepeningSearch` plus `aStarSearch` with the three heuristic
# functions) to three eight-tile puzzle problems with start state
# 
# $$
# \begin{array}{ccc}
# 1 & 2 & 3\\
# 4 & 0 & 5\\
# 6 & 7 & 8
# \end{array}
# $$
# 
# and these three goal states.
# 
# $$
# \begin{array}{ccccccccccc}
# 1 & 2 & 3  & ~~~~ & 1 & 2 & 3  &  ~~~~ & 1 & 0 &  3\\
# 4 & 0 & 5  & & 4 & 5 & 8  & & 4 & 5 & 8\\
# 6 & 7 & 8 &  & 6 & 0 & 7  & & 2 & 6 & 7
# \end{array}
# $$

# Print a well-formatted table like the following.  Try to match this
# format. If you have time, you might consider learning a bit about the `DataFrame` class in the `pandas` package.  When displayed in jupyter notebooks, `pandas.DataFrame` objects are nicely formatted in html.
# 
#            [1, 2, 3, 4, 0, 5, 6, 7, 8]    [1, 2, 3, 4, 5, 8, 6, 0, 7]    [1, 0, 3, 4, 5, 8, 2, 6, 7] 
#     Algorithm    Depth  Nodes  EBF              Depth  Nodes  EBF              Depth  Nodes  EBF          
#          IDS       0      0  0.000                3     43  3.086               11 225850  2.954         
#         A*h1       0      0  0.000                3    116  4.488               11 643246  3.263         
#         A*h2       0      0  0.000                3     51  3.297               11 100046  2.733         
# 
# Of course you will have one more line for `h3`.

# # Function Definitions and Explanations

# ### A* Functions and Node class

# Here we have the node class. This represents the extra information that each node in a A* search needs to carry out the search successfully. H represents the value of the heuristic function at that node. G represents the total cost so far to get to that node, and F is the total of the two, making it an estimate of the total path cost. 

# In[1]:


class Node:
    def __init__(self, state, f=0, g=0 ,h=0):
        self.state = state
        self.f = f
        self.g = g
        self.h = h
    def __repr__(self):
        return "Node(" + repr(self.state) + ", f=" + repr(self.f) +                ", g=" + repr(self.g) + ", h=" + repr(self.h) + ")"


# Here is the `aStarSearch` which kicks off the A* search algorithm. It takes a startState representing the starting state of the puzzle, an actions function that allows successor states to be found, an action function that allows the states to be modified, a goal test function that can test for a successful state, as well as the heuristic function. It initializes the root `Node` with the given heuristic function, then passes it off to the helper.
# 
# Node counting has also been implemented in addition to the supplied code. An additional default initialized variable is included so that the search returns the expected output normally. Python passes primitives by value, but to avoid using globals the node count is stored in an array. This way we can always access the same int by just using node_count[0]. If countNodes is set to True, we return the expected result in addition to the value of node_count.

# In[2]:


def aStarSearch(startState, actionsF, takeActionF, goalTestF, hF, countNodes = False):
    h = hF(startState)
    startNode = Node(state=startState, f=0+h, g=0, h=h)
    node_count = [0] 
    if countNodes: 
        return aStarSearchHelper(startNode, actionsF, takeActionF, goalTestF, hF, float('inf'), 
                                 node_count), node_count[0]
    else:
        return aStarSearchHelper(startNode, actionsF, takeActionF, goalTestF, hF, float('inf'), node_count[0])


# Here is the recursive helper method for A* search. It takes a parent node and fmax, which represents a cutoff f value, as well as the functions that were passed into `aStarSearch` above. First it checks the simple case of whether the goal has been found or not. Then it creates a list of possible actions using the action function. If no actions exist, then moves can be made from this state. In that case "failure" and $\infty$ are returned because the goal state cannot ever be found. Otherwise, a list of children are generated using the action function. Each of these has a heuristic value H and path cost estimate F. We find the best cost estimate by sorting the children on F, and if this is higher than fmax the search is ended. Next we make the recursive call passing in the current best child and the second best as the new fmax. This means that if a higher f value is found than the second best, we end the search at that level and try again with the second best. 
# 
# node_count is an additional variable added on top of the supplied code. In order to avoid using globals, nodes are counted by using the zeroth index of the array node_count. It is pretty easy to tell when to count a node here, any time the `Node` constructor is called, we increment the total node count.

# In[3]:


def aStarSearchHelper(parentNode, actionsF, takeActionF, goalTestF, hF, fmax, node_count):
    if goalTestF(parentNode.state):
        return ([parentNode.state], parentNode.g)
    ## Construct list of children nodes with f, g, and h values
    actions = actionsF(parentNode.state)
    if not actions:
        return ("failure", float('inf'))
    children = []
    for action in actions:
        (childState, stepCost) = takeActionF(parentNode.state, action)
        h = hF(childState)
        g = parentNode.g + stepCost
        f = max(h+g, parentNode.f)
        childNode = Node(state=childState, f=f, g=g, h=h)
        if node_count: node_count[0] += 1
        children.append(childNode)
    while True:
        # find best child
        children.sort(key = lambda n: n.f) # sort by f value
        bestChild = children[0]
        if bestChild.f > fmax:
            return ("failure",bestChild.f)
        # next lowest f value
        alternativef = children[1].f if len(children) > 1 else float('inf')
        # expand best child, reassign its f value to be returned value
        result,bestChild.f = aStarSearchHelper(bestChild, actionsF, takeActionF, goalTestF,
                                            hF, min(fmax,alternativef), node_count)
        if result is not "failure":               
            result.insert(0,parentNode.state)
            retlist = [result, bestChild.f]
            return tuple(retlist)        


# ### Heuristic functions

# **H = 0**:
# 
# The most simple possible heuristic function. Just estimates the cost as zero, which is never an over-estimate so it is admissible. 

# In[4]:


def h1_8p(state, goal):
    return 0


# **Manhattan Distance:**
# 
# Finds the manhattan distance of the current state from the goal state by using a manhattan distance. This is calculating by adding the verticle moves and the horizontal moves together to get to the location, like moving through blocks in Manhattan. This will never over estimate the goal because the blank tile must move in a over and up fashion like the Manhattan distance calculates, and the best case is that only the blank tile has to move to the goal state. 

# In[5]:


def h2_8p(state, goal):
    currentIndex = state.index(0)
    goalIndex = goal.index(0)
    rowDelta = abs(currentIndex // 3 - goalIndex // 3)
    colDelta = abs(currentIndex % 3 - goalIndex % 3)
    return colDelta + rowDelta


# **Nodes Out Of Place:**
# 
# Counts the total number of none-blank tiles that are not in their goalstate. This is an admissible heuristic because the total number of tiles out of place can never be more that the moves required to make the goalstate. This is because a tile must move for a cost of at least one to go from one state to another. It is also a reasonable guess because as we get closer to the solution, fewer tiles will be out of place. 

# In[6]:


def h3_8p(state, goal):
    h = 0
    for i in range(0,9):
        if state[i] != 0 and state[i] != goal[i]:
            h += 1
    return h


# **Not a Correct Heuristic**:
# 
# Counts the out number of tiles out of their goal column and out of their goal row. [Here](http://www.cs.rpi.edu/academics/courses/fall00/ai/assignments/assign3heuristics.html) is the source for the heuristic. We accomplish this goal by using the symmetric difference between the goal row or column and the state row or column. The length of the symmetric difference tells us how many nodes are out of place. The heuristic is, as it turns out, is not admissible. I wrote the function for it and it seems to work. Howerver, it overestimates the cost remaining. A simple example illustrates why it isn't admissible. Say the startState was the solved goal state with the blank spot moved up one so that 0 and 2 had switched places. This heuristic would estimate the cost of two, because it would count the zero out of place and the two out of place. This is more than the actual cost of one, so the heuristic is not admissible. I left in just to look at how it might behave.

# In[7]:


def h4_8p(state, goal):
    stateRows = []
    goalRows = []
    stateCols = []
    goalCols = []
    
    # Populate the individual rows
    for i in range(0,3):
        stateRows.append(state[i*3:(i+1)*3])
        goalRows.append(state[i*3:(i+1)*3])
    
    # Populate the individual columns
    for i in range(0, 3):
        stateCols.append([state[i],state[i+3],state[i+6]])
        goalCols.append([goal[i],goal[i+3],goal[i+6]])
    
    # The estimate of the distance, heuristic
    h = 0    
    for i in range(0,3):
        h += len(set(stateRows[i]).symmetric_difference(goalRows[i]))
        h += len(set(stateCols[i]).symmetric_difference(goalCols[i]))
    
    return h


# ### Effective Branching Factor

# This is a helper function that calculates the quantity of $\frac{1-b^{d+1}}{1-b}$, which finds the number of nodes, $n$ given the depth, $d$ and branching factor $b$. This helps to compartmentalize the code a bit. It also has a few hard coded values where there is an expected result but the algorithm fails to work correctly. If branch_guess = 1, then a /0 will occur. Although the algorithm does not work with this value, we know that if branch_guess is 1 the number of nodes is just depth * 1. Depth = 0 is also not anticipated by this algorithm and branching factor does not really exist if there is not a single branch, so 1 is hard coded when depth = 0. 

# In[8]:


def ebf_calculation(depth, branch_guess):
    if depth == 0:
        return 1
    if branch_guess == 1:
        return depth
    return (1-branch_guess ** (depth + 1)) /(1 - branch_guess)


# This is the helper method that does binary search in order to find EBF. It takes a lower and upper bound, as well as the real number of nodes, depth and precision. It calculates the midpoint of upper and lower, then finds the guess of the nodes using `ebf_calculation`. If the difference between this calculated quantity and the true number of nodes is less than precision, the search is done and we can return midpoint as the value of the branching factor. Otherwise we check to see whether the guessed number of nodes is less than the real number, and recursively call `ebf_helper` using midpoint as the lower bound. Otherwise, we know that the branch factor is lower than our guess, so we call `ebf_helper` with lower and midpoint as the bounds. 

# In[9]:


def ebf_helper(lower, upper, nNodes, depth, precision):
    midpoint = (lower + upper) / 2
    guessed_nodes = ebf_calculation(depth, midpoint)
    if abs(nNodes - guessed_nodes) < precision:
        return midpoint
    if guessed_nodes < nNodes:
        return ebf_helper(midpoint, upper, nNodes, depth, precision)
    else:
        return ebf_helper(lower, midpoint, nNodes, depth, precision)    


# This starts of the search for ebf. It has a hard coded case for when both nNodes and depth = 0 to protect against infinite recursion. It uses 1 as the minimum branching factor, and nNodes as the maximum to give to `ebf_helper` because these are the minimum and maximum possible values for the given branching factor.

# In[10]:


def ebf(nNodes, depth, precision=0.01):
    if nNodes == 0 and depth == 0:
        return 0.000
    return ebf_helper(1, nNodes, nNodes, depth, precision)


# ### Goal Test Function

# This is a simple goal test function, it just uses pythons list comparison to see whether the two lists are equal.

# In[11]:


def goalTestF_8p(state, goal):
    return state == goal


# ### Experiment Function
# 
# This is quite the function. It takes in three goalstates and runs them with `iterativeDeepeningSearch` and each heuristic in heuristicList with A*. Although it is a long function, all the work is going into repetitive function calls and formatting the output. The only calculation that is independent of the functions is correcting the depth by subtracting one from the solution list. We need to do this because both search methods add in the start state to the solution path, which is not counted as part of the depth. 

# In[41]:


import pandas
import time
def runExperiment(goalState1, goalState2, goalState3, heuristicList):
    startState = [1, 2, 3, 4, 0, 5, 6, 7, 8]
    print("\t" + str(goalState1) + "   "
         + str(goalState2) + "   "
         + str(goalState3))
    
    # Building the output
    # How to concat https://stackoverflow.com/questions/13079852/how-do-i-stack-two-dataframes-next-to-each-other-in-pandas
    rowNames = ["IDS"] + ["A*h" + str(s + 1) for s, _ in enumerate(heuristicList)] #NOICE
    dataFrames = []
    for goalState in [goalState1, goalState2, goalState3]:
    
        # Blank initialization
        nodes = []
        EBF = []
        depths = []
        times = []
        # Get data from IDS
        startTime = time.time()
        idsResult, nodeCount = iterativeDeepeningSearch(startState, goalState, actionsF_8p, takeActionF_8p, 20, True)
        endTime = time.time()
        times.append(endTime - startTime)
        nodes.append(nodeCount)
        EBF.append(ebf(nodeCount, len(idsResult) - 1)) # Note the -1 on depth, because start was appended
        depths.append(len(idsResult) - 1)
        # Loop through and do the previous for all h in heuristicList
        for h in heuristicList:
            startTime = time.time()
            hResult, nodeCount = aStarSearch(startState, actionsF_8p, takeActionF_8p, 
                                 lambda s: goalTestF_8p(s, goalState),
                                 lambda s: h(s, goalState), True)
            endTime = time.time()
            times.append(endTime - startTime)
            nodes.append(nodeCount)
            EBF.append(ebf(nodeCount, len(hResult[0]) - 1))
            depths.append(len(hResult[0]) - 1)
        dataFrames.append(pandas.DataFrame({"Algorithm":rowNames,
                           "Depths": depths,
                           "Nodes":nodes,
                            "EBF": EBF, 
                            "Time": times}).set_index("Algorithm"))#Maybe a little hacky
    # print(dataFrames)
    pandas.set_option("precision", 3)
    pandas.option_context("display.colheader_justify", "right")
    #pandas.set_option("expand_frame_repr", False)
    keys = [str(l) for l in [goalState1, goalState2, goalState3]]
    print(pandas.concat(dataFrames, axis=1))
    


# Example output of `runExperiment`. I put quite a lot of effort into matching the output, which did work! 
# , 
#                             "   ":["" for _ in range(0, len(heuristicList) + 1)]

# ## Old Functions
# From A2. Only small changes were made to `depthLimitedSearch`, `iterativeDeepeningSearch` and `actionsF_8p`. Those changes are noted.

# Added a variable for cost now that takeActionF returns a tuple. In addition, added the node counting functionality. Node_count is an array passed in with the total node count at index 0. Each action we explore in the actionsF for loop represents a created node, so for each iteration of this loop we increment node_count[0].

# In[13]:


def depthLimitedSearch(state, goalState, actionsF, takeActionF, depthLimit, node_count):
    if state == goalState:
        return []
    if depthLimit == 0:
        return "cutoff"
    cutoffoccurred = False
    for action in actionsF(state):          
        (childState, _) = takeActionF(state, action) # This was modified to deal with the new cost.
        node_count[0] += 1
        result = depthLimitedSearch(childState, goalState, actionsF, takeActionF, depthLimit - 1, node_count)
        if result == "cutoff":
            cutoffoccurred = True
        elif result != "failure":
            result.insert(0, childState)
            return result
    if cutoffoccurred:
        return "cutoff"
    else:
        return "failure"


# countNodes is added to indicate whether to return just the result or the result with node_count. 

# In[14]:


def iterativeDeepeningSearch(startState, goalState, actionsF, takeActionF, maxDepth, countNodes = False):
    node_count = [0]
    for depth in range(0, maxDepth):
        result = depthLimitedSearch(startState, goalState, actionsF, takeActionF, depth, node_count)
        if result == "failure":
            return "failure"
        if result != "cutoff":
            result.insert(0, startState)
            if countNodes:
                return result, node_count[0]
            else: 
                return result
    return "cutoff"


# Converted this from a generator because of assignment requirements. Also added in the step cost of one to each action. 

# In[15]:


def actionsF_8p(state):
    i = state.index(0)
    actions = []
    if i % 3 > 0:
        actions.append(("left",1))
    if i % 3 < 2:
        actions.append(("right",1))
    if i // 3 > 0:
        actions.append(("up",1))
    if i // 3 < 2:
        actions.append(("down",1))
    return actions


# In[16]:


def takeActionF_8p(state, action):
    #this does not check if action is allowed
    newState = state.copy()
    i = newState.index(0)
    #print(action[0])
    if action[0] == "right":
        newState[i], newState[i+1] = newState[i+1], newState[i]
    elif action[0] == "left":
        newState[i], newState[i-1] = newState[i-1], newState[i]
    elif action[0] == "up":
        newState[i], newState[i-3] = newState[i-3], newState[i]
    elif action[0] == "down":
        newState[i], newState[i+3] = newState[i+3], newState[i]
    return newState, action[1]


# In[17]:


def printPath_8p(startState, goalState, path):
    for state in path:
        printState_8p(state)
        print()
    printState_8p(goalState)


# # Tests

# ### Updated old Functions

# These test to make sure that the costs are correctly handled by the old A2 functions.

# In[18]:


startState = [1,2,3,4,0,5,6,7,8]
actions = actionsF_8p(startState)
assert(('left', 1) == actions[0])
down = takeActionF_8p(startState, actions[3])
assert(([1, 2, 3, 4, 7, 5, 6, 0, 8], 1) == down)
up = takeActionF_8p(startState, ("up", 1))
assert(([1, 0, 3, 4, 2, 5, 6, 7, 8], 1) == up)
print("Tests for old functions passed")


# ### Heuristic Functions

# Pretty simple to test this one.

# In[19]:


assert(0 == h1_8p("literally", "anything"))
print("Test for h1_8p passed!")


# Here we test the expected values of the Manhattan Heuristic. It is simple to visually confirm the values by counting the horizontal and verticle moves required. This test checks the value of the heuristic as the blank approaches the goal state. 

# In[20]:


startState = [0,1,2,3,4,5,6,7,8]
goalState = [1,2,3,4,5,6,7,8,0]
assert(4 == h2_8p(startState, goalState))
goalState,_ = takeActionF_8p(goalState, ("left",1))
assert(3 == h2_8p(startState, goalState))
goalState,_ = takeActionF_8p(goalState, ("left",1))
assert(2 == h2_8p(startState, goalState))
goalState,_ = takeActionF_8p(goalState, ("up",1))
assert(1 == h2_8p(startState, goalState))
print("Tests for h2_8p passed!")


# We test the tiles out of place heuristic in a similar fashion to the previous. The first case makes sure that it estimates one when only one remove remains. The others check other values. These were verified by hand by counting the out of place numbers excluding zero. Eight is the highest it can return as a predicted distance, because no more than eight pieces can be out of place in the 8-puzzle. 

# In[21]:


startState = [1,2,3,4,0,5,6,7,8]
goalState1 = [1,0,3,4,2,5,6,7,8]
goalState2 = [1,2,3,4,5,8,6,0,7]
goalState3 = [8,7,6,5,4,3,2,1,0]
assert(1 == h3_8p(startState, goalState1))
assert(3 == h3_8p(startState, goalState2))
assert(8 == h3_8p(startState, goalState3))
print("h3_8p tests successful!")


# ### Goal Test Functions

# `goaltTestF_8p` is fairly simple to test. We make two identical lists and test if the function returns true. Then modify one of those lists and see if it returns false. 

# In[22]:


startState = [1, 2, 3, 4, 5, 6, 7, 8, 0]
goalState = startState.copy()
assert(True == goalTestF_8p(startState, goalState))
goalState = takeActionF_8p(goalState, "left")
assert(False == goalTestF_8p(startState, goalState))
print("Tests of goalTestF_8p passed!")


# ### Effective Branching Factor Testing

# First, test the calculation helper function first by using some handmade examples.

# In[23]:


assert(7 == ebf_calculation(2,2))
assert(6 == ebf_calculation(6, 1))
assert(13 == ebf_calculation(2, 3))
print("Tests of ebf_calculation passed!")


# Now, test the main recursive funciton `ebf`. Tests a few intermediate values and to make sure the hardcoding works. 

# In[24]:


assert(-0.0001 < ebf(0,0) < 0.0001)
assert(-1.0001 < ebf(1,0) < 1.0001)
assert(1.0078 < ebf(2,1) < 1.00789)
assert(1.0000000 < ebf(2,1,precision = 0.000001) < 1.000001)
assert(1.6613 < ebf(10,3) < 1.664)
assert(11.2755 < ebf(200000, 5) < 11.2756)
print("Tests of ebf passed!")


# ### Testing Node Counting
# This is used to find the count of nodes in an experiment. When the variable `countNodes` is set to True, the nodes created by the search is retuned in addition to the regular result. We test this against some given values.

# In[25]:


startState = [1,2,3,4,0,5,6,7,8]
goalState = [1,2,3,4,5,8,6,0,7]
a, nodeCount = aStarSearch(startState, actionsF_8p, takeActionF_8p, 
                                 lambda s: goalTestF_8p(s, goalState),
                                 lambda s: h1_8p(s, goalState), True)
assert(nodeCount == 116)
_, nodeCount = aStarSearch(startState, actionsF_8p, takeActionF_8p, 
                                 lambda s: goalTestF_8p(s, goalState),
                                 lambda s: h2_8p(s, goalState), True)
assert(nodeCount == 51)
_, nodeCount = iterativeDeepeningSearch(startState, goalState, actionsF_8p, takeActionF_8p, 10, True)
assert(nodeCount == 43)
print("Tests of counting funcitonality passed!")


# ### Given Trial Run

# Here is a simple example using our usual simple graph search.

# In[26]:


def actionsF_simple(state):
    succs = {'a': ['b', 'c'], 'b':['a'], 'c':['h'], 'h':['i'], 'i':['j', 'k', 'l'], 'k':['z']}
    return [(s, 1) for s in succs.get(state, [])]

def takeActionF_simple(state, action):
    return action

def goalTestF_simple(state, goal):
    return state == goal

def h_simple(state, goal):
    return 1


# In[27]:


actions = actionsF_simple('a')
actions


# In[28]:


takeActionF_simple('a', actions[0])


# In[29]:


goalTestF_simple('a', 'a')


# In[30]:


h_simple('a', 'z')


# In[31]:


iterativeDeepeningSearch('a', 'z', actionsF_simple, takeActionF_simple, 10)


# In[32]:


aStarSearch('a',actionsF_simple, takeActionF_simple,
            lambda s: goalTestF_simple(s, 'z'),
            lambda s: h_simple(s, 'z'))


# ### Experiment Trials

# Here we will use the `runExperiment` function to test out the algorithms on different goal states. 

# In[1]:


g1 = [1,2,3,4,0,5,6,7,8]
g2 = [1,2,3,4,5,8,6,0,7]
g3 = [1,0,3,4,5,8,2,6,7]
hlist = [h1_8p, h2_8p, h3_8p]
runExperiment(g1,g2,g3,hlist)


# ## Grading

# Download [A3grader.tar](http://www.cs.colostate.edu/~anderson/cs440/notebooks/A3grader.tar) and extract A3grader.py from it.

# In[35]:


get_ipython().run_line_magic('run', '-i A3grader.py')


# ## Extra Credit

# Add a third column for each result (from running `runExperiment`) that is the number of seconds each search required.  You may get the total run time when running a function by doing
# 
#      import time
#     
#      start_time = time.time()
#     
#      < do some python stuff >
#     
#      end_time = time.time()
#      print('This took', end_time - start_time, 'seconds.')
# 
