
# coding: utf-8

# $\newcommand{\xv}{\mathbf{x}}
# \newcommand{\Xv}{\mathbf{X}}
# \newcommand{\yv}{\mathbf{y}}
# \newcommand{\zv}{\mathbf{z}}
# \newcommand{\av}{\mathbf{a}}
# \newcommand{\Wv}{\mathbf{W}}
# \newcommand{\wv}{\mathbf{w}}
# \newcommand{\tv}{\mathbf{t}}
# \newcommand{\Tv}{\mathbf{T}}
# \newcommand{\muv}{\boldsymbol{\mu}}
# \newcommand{\sigmav}{\boldsymbol{\sigma}}
# \newcommand{\phiv}{\boldsymbol{\phi}}
# \newcommand{\Phiv}{\boldsymbol{\Phi}}
# \newcommand{\Sigmav}{\boldsymbol{\Sigma}}
# \newcommand{\Lambdav}{\boldsymbol{\Lambda}}
# \newcommand{\half}{\frac{1}{2}}
# \newcommand{\argmax}[1]{\underset{#1}{\operatorname{argmax}}}
# \newcommand{\argmin}[1]{\underset{#1}{\operatorname{argmin}}}$

# # Assignment 6: Min-Conflicts

# *Type your name here and rewrite all of the following sections.  Add more sections to present your code, results, and discussions.*

# For this assignment, you will use the `min-conflicts` code from the lecture notes to solve the following scheduling problem. <font color="red">Do not change this code in completing this assignment.</font>

# You are in charge of assigning classes to classrooms and times for the
# department. The scheduling is simplified by the fact at this imaginary university each class meets every day. 
# 
# You have been asked to schedule only thee class rooms:
# 
#   * CSB 130
#   * CSB 325
#   * CSB 425
#   
# Classes start on the hour. You can only assign classes to the hours of
# 
#   * 9 am
#   * 10 am
#   * 11 am
#   * 12 pm
#   *  1 pm
#   *  2 pm
#   *  3 pm
#   *  4 pm
#   
# You must schedule these 22 classes:
# 
#   * CS160, CS163, CS164,
#   * CS220, CS270, CS253,
#   * CS320, CS314, CS356, CS370,
#   * CS410, CS414, CS420, CS430, CS440, CS445, CS453, CS464,
#   * CS510, CS514, CS535, CS540, CS545
# 
# Your schedule must not violate any of the following constraints.
# 
#   * Two classes cannot meet in the same room at the same time.
#   * Classes with the same first digit cannot meet at the same time, because students might take a subset of these in one semester. There is one exception to this rule. CS163 and CS164 can meet at the same time.

# The variables for this CSP problem are the classes.  The values you assign to each class will be a tuple containing a room and a time.

# ## Required Functions

#      assignments, steps = schedule(classes, times, rooms, max_steps)
#      
#      # classes: list of all class names, like 'CS410'
#      # times: list of all start times, like '10 am' and ' 1 pm'
#      # rooms: list of all rooms, like 'CSB 325'
#      # max_steps: maximum number of assignments to try
#      # assignments: dictionary of values assigned to variables, like {'CS410': ('CSB 425', '10 am'), ...}
#      # steps: actual number of assignments tested before solution found
#      #    assignments and steps will each be None if solution was not found.
#      
#      result = constraints_ok(class_name_1, value_1, class_name_2, value_2)
#      
#      # class_name_1: as above, like 'CS410'
#      # value_1: tuple containing room and time.
#      # class_name_2: a second class name
#      # value_2: another tuple containing a room and time
#      # result: True of the assignment of value_1 to class_name 1 and value_2 to class_name 2 does
#      #         not violate any constraints.  False otherwise.
#      
#      display(assignments, rooms, times)
#      
#      # assignments: returned from call to your schedule function
#      # rooms: list of all rooms as above
#      # times: list of all times as above
#      

# ## Given Functions
# 

# In[4]:


import random

def min_conflicts(vars, domains, constraints, neighbors, max_steps=1000): 
    """Solve a CSP by stochastic hillclimbing on the number of conflicts."""
    # Generate a complete assignment for all vars (probably with conflicts)
    current = {}
    for var in vars:
        val = min_conflicts_value(var, current, domains, constraints, neighbors)
        current[var] = val
    # Now repeatedly choose a random conflicted variable and change it
    for i in range(max_steps):
        conflicted = conflicted_vars(current,vars,constraints,neighbors)
        if not conflicted:
            return (current,i)
        var = random.choice(conflicted)
        val = min_conflicts_value(var, current, domains, constraints, neighbors)
        current[var] = val
    return (None,None)


# In[5]:


def min_conflicts_value(var, current, domains, constraints, neighbors):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(domains[var],
                             lambda val: nconflicts(var, val, current, constraints, neighbors)) 


# In[6]:


def conflicted_vars(current,vars,constraints,neighbors):
    "Return a list of variables in current assignment that are in conflict"
    return [var for var in vars
            if nconflicts(var, current[var], current, constraints, neighbors) > 0]


# In[7]:


def nconflicts(var, val, assignment, constraints, neighbors):
    "Return the number of conflicts var=val has with other variables."
    # Subclasses may implement this more efficiently
    def conflict(var2):
        val2 = assignment.get(var2, None)
        return val2 != None and not constraints(var, val, var2, val2)
    return len(list(filter(conflict, neighbors[var])))


# In[8]:


def argmin_random_tie(seq, fn):
    """Return an element with lowest fn(seq[i]) score; break ties at random.
    Thus, for all s,f: argmin_random_tie(s, f) in argmin_list(s, f)"""
    best_score = fn(seq[0]); n = 0
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score; n = 1
        elif x_score == best_score:
            n += 1
            if random.randrange(n) == 0:
                    best = x
    return best


# ## Implemented Functions

# In[29]:


def build_tuples(times, rooms):
    retlist = []
    for t in times:
        for r in rooms:
            retlist.append((r,t))
    return retlist


# In[51]:


def schedule(classes, times, rooms, max_steps):
    #classes are your variables
    # times and rooms are your domains, these are the things constraints ok checkts
    #max_steps is the time to run the min_constraints.
    
    #build full list of tuples.
    domainList = build_tuples(times, rooms)

    # create domains. To start, each class has each room and time
    domains = {key: domainList for key in classes} 
    
    #create neighbors
    neighbors = {key: [c for c in classes if c != key] for key in classes}
    
    solution, steps = min_conflicts(classes, domains, constraints_ok, neighbors, max_steps)
    
    return solution, steps


# In[15]:


def constraints_ok(class_name_1, value_1, class_name_2, value_2):
    ##class_name_1&2 are just string names.
    ##value_1&2 are tuples of (class name, time). Both strings
    class1_room, class1_time = value_1
    class2_room, class2_time = value_2
    
    ##should we check to see if names are the same? that doesn't violate constraints.
    ##Can't be in the same room at the same time. True for everyone.
    if (class1_time != class2_time):
        #we're good no matter what in this case.
        return True
    ##classes with the same first digit cannot be at the same time. 
    elif (class1_room != class2_room and class_name_1[2] != class_name_2[2]):
        #in this case their in different rooms, so same time is ok unless they have same level
        return True
    
    
    return False


# In[103]:


def display(assignments, rooms, times):
    length = len(rooms)
    print('   ',('   {} '*length).format(*rooms))
    print('---' + '-----------' * length)
    for t in times:
        classes = [class_number for class_number,pair in assignments.items() if pair[1] == t]
        print(t + ('   {}   '*len(classes)).format(*classes))
        #print(classes)


# ## Testing 

# Testing `constraints_ok`

# In[22]:


assert(not constraints_ok("CS160", ("CLARK 101", '9 am'), "CS200", ("CLARK 101", '9 am')))
assert(not constraints_ok("CS160", ("CLARK 101", '9 am'), "CS161", ("CSB130", '9 am')))
assert(constraints_ok("CS160", ("CLARK 101", '9 am'), "CS161", ("CSB130", '8 am')))
assert(constraints_ok("CS160", ("CLARK 101", '9 am'), "CS270", ("CSB130", '9 am')))


# Testing `build_tuples`

# In[49]:


domainList = build_tuples(times, rooms)
domains = {key: domainList for key in classes}
domains


# ## Examples

# In[25]:


classes = ['CS160', 'CS163', 'CS164',
           'CS220', 'CS270', 'CS253',
           'CS320', 'CS314', 'CS356', 'CS370',
           'CS410', 'CS414', 'CS420', 'CS430', 'CS440', 'CS445', 'CS453', 'CS464',
           'CS510', 'CS514', 'CS535', 'CS540', 'CS545']

times = [' 9 am',
         '10 am',
         '11 am',
         '12 pm',
         ' 1 pm',
         ' 2 pm',
         ' 3 pm',
         ' 4 pm']

rooms = ['CSB 130', 'CSB 325', 'CSB 425']


# In[110]:


max_steps = 100
assignments, steps = schedule(classes, times, rooms, max_steps)
print('Took', steps, 'steps')
print(assignments)


# In[111]:


display(assignments, rooms, times)


# ## Check-in

# Do not include this section in your notebook.
# 
# Name your notebook ```Lastname-A6.ipynb```.  So, for me it would be ```Anderson-A6.ipynb```.  Submit the file using the ```Assignment 6``` link on [Canvas](https://colostate.instructure.com/courses/68135).

# ## Grading
# 
# Download [A6grader.tar](http://www.cs.colostate.edu/~anderson/cs440/notebooks/A6grader.tar) and extract `A6grader.py` from it.  Grader will be available soon.

# In[1]:


get_ipython().run_line_magic('run', '-i A6grader.py')


# # Extra Credit
# 
# Solve the scheduling problem again but with the addition of
# these preferences:
# 
#   * prefer schedules that do not schedule classes at 9 am, 12 pm or 4 pm.
#   * prefer schedules with CS163 and CS164 meeting at 1 pm or 2 pm.
# 
# To accomplish this, do not modify `min_conflicts`.  Write another function that calls `min_conflicts` repeatedly, and for each solution found count the number of these new preferences that are not satisfied.  Remember the solution with the smallest count of unsatisfied preferences.  
