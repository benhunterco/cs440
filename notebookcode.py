
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

# # Function Definitions

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

# In[2]:


def aStarSearch(startState, actionsF, takeActionF, goalTestF, hF, countNodes = False):
    h = hF(startState)
    startNode = Node(state=startState, f=0+h, g=0, h=h)
    node_count = [0] 
    if countNodes: 
        return aStarSearchHelper(startNode, actionsF, takeActionF, goalTestF, hF, float('inf'), 
                                 node_count), node_count[0]
    else:
        return aStarSearchHelper(startNode, actionsF, takeActionF, goalTestF, hF, float('inf'), node_count)


# Here is the recursive helper method for A* search. It takes a parent node and fmax, which represents a cutoff f value, as well as the functions that were passed into `aStarSearch` above. First it checks the simple case of whether the goal has been found or not. Then it creates a list of possible actions using the action function. If no actions exist, then moves can be made from this state. In that case "failure" and $\infty$ are returned because the goal state cannot ever be found. Otherwise, a list of children are generated using the action function. Each of these has a heuristic value H and path cost estimate F. We find the best cost estimate by sorting the children on F, and if this is higher than fmax the search is ended. Next we make the recursive call passing in the current best child and the second best as the new fmax. This means that if a higher f value is found than the second best, we end the search at that level and try again with the second best. 

# In[3]:


def aStarSearchHelper(parentNode, actionsF, takeActionF, goalTestF, hF, fmax, node_count):
    if goalTestF(parentNode.state):
        return ([parentNode.state], parentNode.g)
    ## Construct list of children nodes with f, g, and h values
    actions = actionsF(parentNode.state)
    if not actions:
        return ("failure", float('inf'))
    children = []
    for action, stepCost in actions:
        childState = takeActionF(parentNode.state, action)
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

# The most simple possible heuristic function. Just estimates the cost as zero, which is never an over-estimate so it is admissible. 

# In[4]:


def h1_8p(state, goal):
    return 0


# Finds the manhattan distance of the current state from the goal state by using a manhattan distance. This is calculating by adding the verticle moves and the horizontal moves together to get to the location, like moving through blocks in Manhattan. 

# In[5]:


def h2_8p(state, goal):
    currentIndex = state.index(0)
    goalIndex = goal.index(0)
    rowDelta = abs(currentIndex // 3 - goalIndex // 3)
    colDelta = abs(currentIndex % 3 - goalIndex % 3)
    return colDelta + rowDelta


# ### Effective Branching Factor

# In[6]:


def ebf_calculation(depth, branch_guess):
    if depth == 0:
        return 1
    if branch_guess == 1:
        return depth
    return (1-branch_guess ** (depth + 1)) /(1 - branch_guess)


# In[7]:


def ebf_helper(lower, upper, nNodes, depth, precision):
    midpoint = (lower + upper) / 2
    guessed_nodes = ebf_calculation(depth, midpoint)
    if abs(nNodes - guessed_nodes) < precision:
        return midpoint
    if guessed_nodes < nNodes:
        return ebf_helper(midpoint, upper, nNodes, depth, precision)
    else:
        return ebf_helper(lower, midpoint, nNodes, depth, precision)    


# In[115]:


def ebf(nNodes, depth, precision=0.01):
    if nNodes == 0 and depth == 0:
        return 0.000
    return ebf_helper(1, nNodes, nNodes, depth, precision)


# ### Goal Test Function

# This is a simple goal test function, it just uses pythons list comparison to see whether the two lists are equal.

# In[9]:


def goalTestF_8p(state, goal):
    return state == goal


# ### Experiment Function
# 
# This is quite the function. It takes in three goalstates and runs them with `iterativeDeepeningSearch` and each heuristic in heuristicList with A*. Although it is a long function, all the work is going into repetitive function calls and formatting the output. The only calculation that is independent of the functions is correcting the depth by subtracting one from the solution list. We need to do this because both search methods add in the start state to the solution path, which is not counted as part of the depth. 

# In[184]:


import pandas
def runExperiment(goalState1, goalState2, goalState3, heuristicList):
    startState = [1, 2, 3, 4, 0, 5, 6, 7, 8]
    print("\t" + str(goalState1) + "   "
         + str(goalState2) + "   "
         + str(goalState3))
    
    # Building the output
    # How to join https://stackoverflow.com/questions/13079852/how-do-i-stack-two-dataframes-next-to-each-other-in-pandas
    # First column is done here
    rowNames = ["IDS"] + ["A*h" + str(s + 1) for s, _ in enumerate(heuristicList)] #NOICE
    dataFrames = []
    for goalState in [goalState1, goalState2, goalState3]:
    
        # Blank initialization
        nodes = []
        EBF = []
        depths = []
        # Get data from IDS
        idsResult, nodeCount = iterativeDeepeningSearch(startState, goalState, actionsF_8p, takeActionF_8p, 20, True)
        nodes.append(nodeCount)
        EBF.append(ebf(nodeCount, len(idsResult) - 1)) # Note the -1 on depth, because start was appended
        depths.append(len(idsResult) - 1)
        # Loop through and do the previous for heuristicList
        for h in heristicList:
            hResult, nodeCount = aStarSearch(startState, actionsF_8p, takeActionF_8p, 
                                 lambda s: goalTestF_8p(s, goalState),
                                 lambda s: h(s, goalState), True)
            nodes.append(nodeCount)
            EBF.append(ebf(nodeCount, len(hResult[0]) - 1))
            depths.append(len(hResult[0]) - 1)
        dataFrames.append(pandas.DataFrame({"Algorithm":rowNames,
                           "Depths": depths,
                           "Nodes":nodes,
                            "EBF": EBF, 
                            "      ":["" for _ in range(0, len(hueristicList) + 1)]}).set_index("Algorithm"))#Maybe a little hacky
    # print(dataFrames)
    pandas.set_option("precision", 5)
    #pandas.set_option("expand_frame_repr", False)
    #keys = [str(l) for l in [goalState1, goalState2, goalState3]]
    print(pandas.concat(dataFrames, axis=1))
    


# Example output of `runExperiment`. I put quite a lot of effort into matching the output. 

# In[185]:


g1 = [1,2,3,4,0,5,6,7,8]
g2 = [1,2,3,4,5,8,6,0,7]
g3 = [1,0,3,4,5,8,2,6,7]
hlist = [h1_8p, h2_8p]
runExperiment(g1,g2,g3,hlist)


# ## Old Functions
# From A2. Only small changes were made to `depthLimitedSearch` and `actionsF_8p`. Those changes are noted.

# Added a variable for cost now that takeActionF usually returns a tuple. 

# In[12]:


def depthLimitedSearch(state, goalState, actionsF, takeActionF, depthLimit, node_count):
    if state == goalState:
        return []
    if depthLimit == 0:
        return "cutoff"
    cutoffoccurred = False
    for action, _ in actionsF(state):           # NEWER, this was actually the modified line
        childState = takeActionF(state, action) # This was modified to deal with the new cost.
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


# In[57]:


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

# In[14]:


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


# In[15]:


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


# In[16]:


def printPath_8p(startState, goalState, path):
    for state in path:
        printState_8p(state)
        print()
    printState_8p(goalState)


# # Tests

# ### Hueristic Functions

# Pretty simple to test this one.

# In[17]:


assert(0 == h1_8p("literally", "anything"))
print("Test for h1_8p passed!")


# In[18]:


startState = [0,1,2,3,4,5,6,7,8]
goalState = [1,2,3,4,5,6,7,8,0]
assert(4 == h2_8p(startState, goalState))
goalState = takeActionF_8p(goalState, "left")
assert(3 == h2_8p(startState, goalState))
goalState = takeActionF_8p(goalState, "left")
assert(2 == h2_8p(startState, goalState))
goalState = takeActionF_8p(goalState, "up")
assert(1 == h2_8p(startState, goalState))
print("Tests for h2_8p passed!")


# ### Goal Test Functions

# In[19]:


startState = [1, 2, 3, 4, 5, 6, 7, 8, 0]
goalState = startState.copy()
assert(True == goalTestF_8p(startState, goalState))
goalState = takeActionF_8p(goalState, "left")
assert(False == goalTestF_8p(startState, goalState))
print("Tests of goalTestF_8p passed!")


# ### Effective Branching Factor Testing

# First, some example output for the ebf function.  During execution, this example shows debugging output which is the low and high values passed into a recursive helper function.

# In[20]:


ebf(10, 3)


# The smallest argument values should be a depth of 0, and 1 node.

# In[109]:


ebf(0,0)


# In[110]:


ebf(1, 0)


# In[111]:


ebf(2, 1)


# In[112]:


ebf(2, 1, precision=0.000001)


# In[113]:


ebf(200000, 5)


# In[114]:


ebf(200000, 50)


# ### Testing Node Counting
# This is used to find the count of nodes in an experiment. 

# In[26]:


startState = [1,2,3,4,0,5,6,7,8]
goalState = [1,2,3,4,5,8,6,0,7]
a, nodeCount = aStarSearch(startState, actionsF_8p, takeActionF_8p, 
                                 lambda s: goalTestF_8p(s, goalState),
                                 lambda s: h1_8p(s, goalState), True)
print(len(a))
assert(nodeCount == 116)
_, nodeCount = aStarSearch(startState, actionsF_8p, takeActionF_8p, 
                                 lambda s: goalTestF_8p(s, goalState),
                                 lambda s: h2_8p(s, goalState), True)
print(len(_))
assert(nodeCount == 51)
_, nodeCount = iterativeDeepeningSearch(startState, goalState, actionsF_8p, takeActionF_8p, 10, True)
assert(nodeCount == 43)
print(len(_))
print("Tests of counting funcitonality passed!")


# ### Given Trial Run

# Here is a simple example using our usual simple graph search.

# In[27]:


def actionsF_simple(state):
    succs = {'a': ['b', 'c'], 'b':['a'], 'c':['h'], 'h':['i'], 'i':['j', 'k', 'l'], 'k':['z']}
    return [(s, 1) for s in succs.get(state, [])]

def takeActionF_simple(state, action):
    return action

def goalTestF_simple(state, goal):
    return state == goal

def h_simple(state, goal):
    return 1


# In[28]:


actions = actionsF_simple('a')
actions


# In[29]:


takeActionF_simple('a', actions[0])


# In[30]:


goalTestF_simple('a', 'a')


# In[31]:


h_simple('a', 'z')


# In[32]:


iterativeDeepeningSearch('a', 'z', actionsF_simple, takeActionF_simple, 10)


# In[33]:


aStarSearch('a',actionsF_simple, takeActionF_simple,
            lambda s: goalTestF_simple(s, 'z'),
            lambda s: h_simple(s, 'z'))


# ## Grading

# Download [A3grader.tar](http://www.cs.colostate.edu/~anderson/cs440/notebooks/A3grader.tar) and extract A3grader.py from it.

# In[34]:


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
