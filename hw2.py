import random
def printMaze_10(state):
    for i in range(0,10):
        print(*state[i*10:(i+1)*10], sep = " ")

def regenerateMaze(state):
    state = [random.sample(['x','-','-'],1)[0] for _ in range(0,100)] 

def takeMazeAction:
    pass
def mazeActions:
    pass
def printMazePath(result):
    path = result[0].copy()
    for i in result:
        path[i.index("O")] = "~"
    print(path)

    
#create a random start state, not guaranteed to be solvable
startState = [random.sample(['x','-','-'],1)[0] for _ in range(0,100)] 
#our goal will be to move the circle to the bottom right of the maze.
goalState = startState.copy()
startState[0], goalState[99] = 'O','O'
