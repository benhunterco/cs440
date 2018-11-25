
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

# In[2]:


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


# In[3]:


def min_conflicts_value(var, current, domains, constraints, neighbors):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(domains[var],
                             lambda val: nconflicts(var, val, current, constraints, neighbors)) 


# In[4]:


def conflicted_vars(current,vars,constraints,neighbors):
    "Return a list of variables in current assignment that are in conflict"
    return [var for var in vars
            if nconflicts(var, current[var], current, constraints, neighbors) > 0]


# In[5]:


def nconflicts(var, val, assignment, constraints, neighbors):
    "Return the number of conflicts var=val has with other variables."
    # Subclasses may implement this more efficiently
    def conflict(var2):
        val2 = assignment.get(var2, None)
        return val2 != None and not constraints(var, val, var2, val2)
    return len(list(filter(conflict, neighbors[var])))


# In[6]:


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

# `build_tuples` is a helper function for `schedule`. It takes a list of times and a list of rooms and builds a list containing each combination of times and rooms. 

# In[7]:


def build_tuples(times, rooms):
    retlist = []
    for t in times:
        for r in rooms:
            retlist.append((r,t))
    return retlist


# First, `schedule` builds the list of domains. This is simply all the possible combinations of times and rooms that is built by `build_tuples`. Then it creates the domains dictionary by giving this list to each class as a key to a dictionary. The neighbors dictionary is then created where each class is a key that has every other class as a neighbor in its value list. Finally, `min_conflicts` is called which returns a non-conflicting solution if one is found.

# In[8]:


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


# `constraints_ok` takes a two class value pairs and compares them to see if they have any scheduling conflicts. First it checks to see whether the classes occur at the same time. If they don't then there cannot be a conflict, so True is returned. Then it checks two conditions at once. One is valid for all classes, if they occur at the same time and the same room, then there is a conflict so the function returns false. In addition, it checks to see whether the third digit, which indicates the class level, is the same. If it is, then the the constraints are broken so False is returned. If both of these constraints are passed, the we return True.

# In[9]:


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


# The `display` function takes the assignments result from `schedule` along with a list of room and times and displays them in a reasonable manner. It uses format strings to adapt to the number of rooms and times passed in and display them in a reasonable way. 

# In[137]:


def display(assignments, rooms, times):
    length = len(rooms)
    print('   ',('   {} '*length).format(*rooms))
    print('---' + '-----------' * length)
    for t in times:
        classes = [class_number for class_number,pair in assignments.items() if pair[1] == t]
        print(t + ('   {}   '*len(classes)).format(*classes))
        #print(classes)


# ### Extra Credit Functions
# 
# This will prefer a solution that has later meeting times and schedules the entry level classes around one and two.
# 

# `schedule_advanced` works by calling schedule for `max_tries` and keeping track of whichever trial has the lowest number of the violated preferences (specific meeting times). 

# In[117]:


def schedule_advanced(classes, times, rooms, max_steps, max_tries):
    # call schedule and get the returns
    best_schedule, best_steps = schedule(classes, times, rooms, max_steps)
    min_count = preference_count(best_schedule)
    # min solution is set to that one.
    # min bad things is set to bad_things(schedule)
    # for number of steps
    for _ in range(max_tries):
        new_schedule, new_steps = schedule(classes, times, rooms, max_steps)
        new_count = preference_count(new_schedule)
        if new_count < min_count:
            best_schedule = new_schedule
            min_count = new_count
            best_steps = new_steps
    return best_schedule, best_steps
    # if min bad things > badThings[step], replace


# `preference_count` returns the number of special preferences violated by a given schedule. It goes through each class's meeting time and checks to see whether it occurs early or late or a lunch because people prefere not to have classes at these times. Then it has a special preference for scheduling the intro level classes during the middle of the day so it is easier for students to fit them in. After counting all of these, it returns the count. 

# In[127]:


def preference_count(schedule):
    #returns a count of violated preference.
    count = 0
    # count each class that occurs at 9, 12 or 4.
    for class_name in schedule:
        _, time = schedule[class_name]

        if time == " 9 am" or time == "12 pm" or time == " 4 pm":
            count += 1
        if class_name == "CS163" or class_name == "CS164":
            if not (time == " 1 pm" or time == " 2 pm"):
                count += 1
    return count
        
    # do not count if CS163 and CS164 meeting at 1 pm or 2 pm, otherwise count for each non occurance.


# ## Testing 

# Testing `constraints_ok`

# In[141]:


assert(not constraints_ok("CS160", ("CLARK 101", '9 am'), "CS200", ("CLARK 101", '9 am')))
assert(not constraints_ok("CS160", ("CLARK 101", '9 am'), "CS161", ("CSB130", '9 am')))
assert(constraints_ok("CS160", ("CLARK 101", '9 am'), "CS161", ("CSB130", '8 am')))
assert(constraints_ok("CS160", ("CLARK 101", '9 am'), "CS270", ("CSB130", '9 am')))
print("constraints_ok passed all tests!")


# Testing `build_tuples`

# In[142]:


domainList = build_tuples(times, rooms)
domains = {key: domainList for key in classes}
domains
#each class should have all combos as its list


# Testing `schedule` 

# In[143]:


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


# In[144]:


max_steps = 100
assignments, steps = schedule(classes, times, rooms, max_steps)
print('Took', steps, 'steps')
print(assignments)


# Testing `display`.

# In[145]:


display(assignments, rooms, times)


# Using larger sets 

# In[148]:


classes_ex = ['CS100', 'CS160', 'CS163', 'CS164',
           'CS220', 'CS270', 'CS253',
           'CS320', 'CS314', 'CS356', 'CS370',
           'CS410', 'CS414', 'CS420', 'CS430', 'CS440', 'CS445', 'CS453', 'CS464',
           'CS510', 'CS514', 'CS535', 'CS540', 'CS545']

times_ex = [' 8 am',
         ' 9 am',
         '10 am',
         '11 am',
         '12 pm',
         ' 1 pm',
         ' 2 pm',
         ' 3 pm',
         ' 4 pm']

rooms_ex = ['CSB 130', 'CSB 325', 'CSB 425', 'CLARK 101']


# In[150]:


assignments, steps = schedule(classes_ex, times_ex, rooms_ex, max_steps)
display(assignments, rooms, times)


# Testing `schedule_advanced` and `preference_count` with a reduced class list to see whether it adjusts for preferences. 

# In[138]:


# shortened set to see what solutions it comes up with
classes = ['CS160', 'CS163', 'CS164',
           'CS220', 'CS270', 'CS253',
           'CS320', 'CS314', 'CS356', 'CS370',
           'CS410', 'CS414', 'CS420']
times = [' 9 am',
         '10 am',
         '11 am',
         '12 pm',
         ' 1 pm',
         ' 2 pm',
         ' 3 pm',
         ' 4 pm']
rooms = ['CSB 130', 'CSB 325', 'CSB 425']


# In[139]:


best, steps = schedule_advanced(classes, times, rooms, 100, 5000)
display(best, rooms, times)


# In[140]:


preference_count(best)


# In[151]:


best, steps = schedule_advanced(classes_ex, times_ex, rooms_ex, 100, 5000)
display(best, rooms_ex, times_ex)


# In[152]:


preference_count(best)


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
