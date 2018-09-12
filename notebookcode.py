
# coding: utf-8

# # Assignment 1: Uninformed Search

# Ben Newell

# ## Overview

# Breadth-first and depth-first are two algorithms for performing
# uninformed search---a search that does not use
# knowledge about the goal of the search.  You will implement both
# search algorithms in python and test them on a simple graph.

# ## Required Code

# In this jupyter notebook, you must implement at least the following functions:
# 
#   * `breadthFirstSearch(startState, goalState, successorsf)` 
#   * `depthFirstSearch(startState, goalState, successorsf)`
#   
# Each receives as arguments the starting state, the goal state, and a successors function.  `breadthFirstSearch` returns the breadth-first solution path as a list of states starting with the `startState` and ending with the `goalState`.  `depthFirstSearch` returns the depth-first solution path.
# 
# <font color="red">You must</font> implement the search algorithm as specified in [03 Problem-Solving Agents](http://nbviewer.jupyter.org/url/www.cs.colostate.edu/~anderson/cs440/notebooks/03 Problem-Solving Agents.ipynb) lecture notes.

# If you prefer to develop your python code in a separate editor or IDE, you may do so.  If it is stored in a file called `A1mysolution.py`, you can use it here by executing the following cell.
# 
# When your solution works, <font color="red">remember</font> to remove or comment out the following import statement and instead, paste in all of your function definintions into this notebook.

# # Explanation
# Here is my python implimentation based off of the method specified in [03 Problem-Solving Agents](http://nbviewer.jupyter.org/url/www.cs.colostate.edu/~anderson/cs440/notebooks/03 Problem-Solving Agents.ipynb).
# It starts simply by taking the inputs required by the search functions as well as a boolean to tell the difference between breadth and depth first search. Then it initializes variables `expanded` and `unexpanded` which will be used to keep track of what states to look for. Afterwards is a quick check to see if we are done.
# Then it continues to follow the algorithm...
# 
# 
# Tricky lines
# * list comprehension to set children. In plain english it adds i to children if it is not in expanded or unExpanded
# * The additions at the end. If breadthFirst, we want to add the children to the beginning of the list because we pop off the back
# * In depthFirst, we add to the end so that the lastest child is expanded in the next call to general search.

# In[51]:


def generalSearch(startState, goalState, successorsf, breadthFirst):
    expanded = {}
    unExpanded = [(startState, None)]
    if startState == goalState:
        return startState
    while unExpanded:
        state, parent = unExpanded.pop()
        children = successorsf(state)
        expanded[state] = parent
        #print(expanded)
        children = [i for i in children if i not in expanded and i not in [c for (c,_) in unExpanded]]
        if goalState in children:
            solution = [state, goalState]
            while parent:
                solution.insert(0,parent)
                parent = expanded[parent]
            return solution
        children.sort(reverse = True)
        children = [(i, state) for i in children]
        #print(children)
        if breadthFirst:
            unExpanded = children + unExpanded
        else:
            unExpanded += children
    return "Goal not found"


# In[52]:


def breadthFirstSearch(startState, goalState, successorsf):
    return generalSearch(startState, goalState, successorsf, True)


# In[53]:


def depthFirstSearch(startState, goalState, successorsf):
    return generalSearch(startState, goalState, successorsf, False)


# # Example

# Here is a simple example.  States are defined by lower case letters.  A dictionary stores a list of successor states for each state in the graph that has successors.

# In[54]:


successors = {'a':  ['b', 'c', 'd'],
              'b':  ['e', 'f', 'g'],
              'c':  ['a', 'h', 'i'],
              'd':  ['j', 'z'],
              'e':  ['k', 'l'],
              'g':  ['m'],
              'k':  ['z']}
successors


# In[55]:


import copy

def successorsf(state):
    return copy.copy(successors.get(state, []))


# In[56]:


successorsf('e')


# In[57]:


print('Breadth-first')
print('path from a to a is', breadthFirstSearch('a', 'a', successorsf))
print('path from a to m is', breadthFirstSearch('a', 'm', successorsf))
print('path from a to z is', breadthFirstSearch('a', 'z', successorsf))


# In[58]:


print('Depth-first')
print('path from a to a is', depthFirstSearch('a', 'a', successorsf))
print('path from a to m is', depthFirstSearch('a', 'm', successorsf))
print('path from a to z is', depthFirstSearch('a', 'z', successorsf))


# Let's try a navigation problem around a grid of size 10 x 10.

# In[59]:


def gridSuccessors(state):
    row, col = state
    # succs will be list of tuples () rather than list of lists [] because state must
    # be an immutable type to serve as a key in dictionary of expanded nodes
    succs = []
    for r in [-1, 0, 1]:
        for c in [-1, 0, 1]:
            newr = row + r
            newc = col + c
            if 0 <= newr <= 9 and 0 <= newc <= 9:  # cool, huh?
                succs.append( (newr, newc) )
    return succs


# In[60]:


gridSuccessors([3,4])


# In[61]:


gridSuccessors([3,9])


# In[62]:


gridSuccessors([0,0])


# In[63]:


print('Breadth-first')
print('path from (0, 0) to (9, 9) is', breadthFirstSearch((0, 0), (9, 9), gridSuccessors))


# In[64]:


print('Depth-first')
print('path from (0, 0) to (9, 9) is', depthFirstSearch((0, 0), (9, 9), gridSuccessors))


# Oooo, what kind of path is that?  Let's plot it.

# In[65]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[66]:


path = depthFirstSearch((0, 0), (9, 9), gridSuccessors)
path


# In[67]:


rows = [location[0] for location in path]
cols = [location[1] for location in path]
plt.plot(rows,cols,'o-');


# In[68]:


path = breadthFirstSearch((0, 0), (9, 9), gridSuccessors)
path


# In[69]:


rows = [location[0] for location in path]
cols = [location[1] for location in path]
plt.plot(rows,cols,'o-');


# In[70]:


depthFirstSearch((0, 0), (9, 20), gridSuccessors)


# # Extra Credit
# 
# For extra credit, use your functions to solve the Camels Puzzle, described at [Logic Puzzles](http://www.folj.com/puzzles/).
# The following code illustrates one possible state representation and shows results of a breadth-first and a dept-first search.  You must define a new successors function, called `camelSuccessorsf`. 

# In[108]:


def camelSuccessorsf(camelState):
    successors = []
    index = camelState.index(' ')
    if index - 1 >= 0 and camelState[index - 1] == 'R':
        moveLeft = list(camelState)
        moveLeft[index - 1], moveLeft[index] = moveLeft[index], moveLeft[index - 1]
        successors.append(tuple(moveLeft))
    if index + 2 < len(camelState) and camelState[index + 1] == 'R' and camelState[index + 2] == 'L':
        stepLeft = list(camelState)
        stepLeft[index], stepLeft[index + 2] = stepLeft[index + 2], stepLeft[index]
        successors.append(tuple(stepLeft))
    if index + 1 < len(camelState) and camelState[index + 1] == 'L':
        moveRight = list(camelState)
        moveRight[index + 1], moveRight[index] = moveRight[index], moveRight[index + 1]
        successors.append(tuple(moveRight))
    if index - 2 > 0 and camelState[index - 1] == "L" and camelState[index - 2] == 'R':
        stepRight = list(camelState)
        stepRight[index], stepRight[index - 2] = stepRight[index -2], stepRight[index]
        successors.append(tuple(stepRight))
    return successors


# In[114]:


camelStartState = ('R','R','R','R', ' ', 'L', 'L', 'L','L')


# In[115]:


camelGoalState=('L','L','L','L', ' ', 'R', 'R', 'R', 'R')


# In[116]:


camelSuccessorsf(camelStartState)


# In[117]:


children = camelSuccessorsf(camelStartState)
print(children)
grandChildren = camelSuccessorsf(children[0])
print(grandChildren)
camelSuccessorsf(grandChildren[0])


# In[118]:


bfs = breadthFirstSearch(camelStartState, camelGoalState, camelSuccessorsf)
print('Breadth-first solution: (', len(bfs), 'steps)')
for s in bfs:
    print(s)

dfs = depthFirstSearch(camelStartState, camelGoalState, camelSuccessorsf)
print('Depth-first solution: (', len(dfs), 'steps)')
for s in dfs:
    print(s)


# ## Grading
# 
# Your notebook will be run and graded automatically. Download [A1grader.tar](http://www.cs.colostate.edu/~anderson/cs440/notebooks/A1grader.tar) <font color="red">(COMING SOON)</font> and extract A1grader.py from it. Run the code in the following cell to demonstrate an example grading session. You should see a perfect score of 80/100 if your functions are defined correctly. 
# 
# The remaining 20% will be based on your writing.  In markdown cells, explain what your functions are doing and summarize the algorithms.
# 
# Add at least one markdown cell that describes problems you encountered in trying to solve this assignment.

# ## Check-in

# Do not include this section in your notebook.
# 
# Name your notebook ```Lastname-A1.ipynb```.  So, for me it would be ```Anderson-A1.ipynb```.  Submit the file using the ```Assignment 1``` link on [Canvas](https://colostate.instructure.com/courses/55296).
# 
# Grading will be based on 
# 
#   * correct behavior of the required functions, and
#   * readability of the notebook.

# In[27]:


get_ipython().run_line_magic('run', '-i A1grader.py')

