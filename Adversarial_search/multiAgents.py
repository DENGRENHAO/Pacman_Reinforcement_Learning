# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        return childGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (par1-1)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        numGhosts = gameState.getNumAgents() - 1
        # pacman agent wants max value, so call maxValue(), not minValue()
        return self.maxValue(gameState, 1, numGhosts)

        util.raiseNotDefined()  
        # End your code

    def maxValue(self, gameState, depth, numGhosts):
        # Check if in terminal state
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # maxVal stores the max score we can take
        maxVal = float("-inf")
        # bestAction stores the action we do when we have the highest score
        bestAction = Directions.STOP
        # iterate through every action taken from gameState.getLegalActions(0)
        # (0) means the agentIndex=0 (pacman's agentIndex=0)
        for action in gameState.getLegalActions(0):
            # get min value from minValue because we assume that the ghosts take best actions
            # this means that the ghosts wants us to take the lowest scores
            ####################################################################################################
            # parameters explanation for self.minValue(gameState.getNextState(0, action), depth, numGhosts, 1)
            # gameState.getNextState(0, action): nextState of agentIndex=0, action=action
            # depth=depth (Why don't we pass depth+1? It's because
            # "A single level of the search is considered to be one pacman move and all the ghosts’ responses"
            # numGhosts=numGhosts
            # pass 1 to agentIndex because we want to run the first ghost
            ####################################################################################################
            val = self.minValue(gameState.getNextState(0, action), depth, numGhosts, 1)
            if val > maxVal:
                maxVal = val
                bestAction = action
        # if depth > 1, then return max value we can get
        # else, depth==1, we return bestAction to def getAction(self, gameState):
        if depth > 1:
            return maxVal
        return bestAction

    def minValue(self, gameState, depth, numGhosts, agentIndex):
        # Check if in terminal state
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # minVal stores the min score we can take
        minVal = float("inf")
        # if agentIndex == numGhosts check that whether we're going through the last ghost
        if agentIndex == numGhosts:
            # if depth == self.depth check that whether we're at the last layer
            if depth == self.depth:
                for action in gameState.getLegalActions(agentIndex):
                    minVal = min(minVal, self.evaluationFunction(gameState.getNextState(agentIndex, action)))
            else:
                # we're going to next layer, so depth=depth+1
                # we'll start from the pacman agent, so calling maxValue(), not minValue()
                for action in gameState.getLegalActions(agentIndex):
                    minVal = min(minVal, self.maxValue(gameState.getNextState(agentIndex, action), depth+1, numGhosts))
        else:
            # we want to go through next ghost at the same layer, so depth=depth, agentIndex=agentIndex+1
            for action in gameState.getLegalActions(agentIndex):
                minVal = min(minVal, self.minValue(gameState.getNextState(agentIndex, action), depth, numGhosts, agentIndex+1))
        return minVal


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (part1-2)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        numGhosts = gameState.getNumAgents() - 1
        return self.maxValue(gameState, 1, numGhosts)

        util.raiseNotDefined()
        # End your code

    # this maxValue() part is almost the same as the one in class MinimaxAgent(MultiAgentSearchAgent)
    def maxValue(self, gameState, depth, numGhosts):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        maxVal = float("-inf")
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            # we call expectValue(), not minValue() because not all situations will be deterministic
            val = self.expectValue(gameState.getNextState(0, action), depth, numGhosts, 1)
            if val > maxVal:
                maxVal = val
                bestAction = action
        if depth > 1:
            return maxVal
        return bestAction

    # this expectValue() part is almost the same as minValue() in class MinimaxAgent(MultiAgentSearchAgent)
    def expectValue(self, gameState, depth, numGhosts, agentIndex):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # initial expected value to 0.0, because we still want the pacman agent to lower scores
        expectVal = 0.0
        # we assume that we have the same probability for every action, so probability=1/(the number of actions)
        prob = 1.0 / len(gameState.getLegalActions(agentIndex))
        # expectVal += prob * (...) means that we sum up all values for next state with timing each probability
        if agentIndex == numGhosts:
            if depth == self.depth:
                for action in gameState.getLegalActions(agentIndex):
                    expectVal += prob * self.evaluationFunction(gameState.getNextState(agentIndex, action))
            else:
                for action in gameState.getLegalActions(agentIndex):
                    expectVal += prob * self.maxValue(gameState.getNextState(agentIndex, action), depth + 1, numGhosts)
        else:
            for action in gameState.getLegalActions(agentIndex):
                expectVal += prob * self.expectValue(gameState.getNextState(agentIndex, action), depth, numGhosts,
                                                   agentIndex + 1)
        return expectVal

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (part1-3).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Begin your code
    # get pacman position
    newPos = currentGameState.getPacmanPosition()
    # get all food situation
    ############################################################
    # def getFood(self):
    #     """
    #     Returns a Grid of boolean food indicator variables.
    #
    #     Grids can be accessed via list notation, so to check
    #     if there is food at (x,y), just call
    #
    #     currentFood = state.getFood()
    #     if currentFood[x][y] == True: ...
    #     """
    #     return self.data.food
    ############################################################
    newFood = currentGameState.getFood()
    # get ghost states
    newGhostStates = currentGameState.getGhostStates()
    # get scared times of the ghosts
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # get the minimum scared time among all the ghosts
    minScaredTime = min(newScaredTimes)
    # get the minimum distance between pacman and all ghosts
    closestGhostDist = min(manhattanDistance(newPos, ghostState.configuration.pos) for ghostState in newGhostStates)
    # initialize minimum distance between pacman and all foods to infinity
    closestFoodDist = float("inf")
    remainingFoodCount = 0
    # if there are no foods remaining, then set closestFoodDist to 0
    if not newFood:
        closestFoodDist = 0
    else:
        # newFood is of type "Grid", so we can call newFood.width, newFood.height to iterate through every place in the screen
        for x in range(newFood.width):
            for y in range(newFood.height):
                # check if there's food at (x, y)
                if currentGameState.hasFood(x, y):
                    closestFoodDist = min(closestFoodDist, manhattanDistance(newPos, [x, y]))
                    remainingFoodCount += 1
    # the higher ghostScore is, the better situation the pacman is in
    ghostScore = 0
    # We / (closestGhostDist+1) because the more the distance is, it is less important
    # (closestGhostDist+1) because we don't want denominator to be 0
    if minScaredTime == 0:
        # if some ghosts are not scared, we are in bad situation, so the value is minus
        ghostScore = -2.5 / (closestGhostDist+1)
    else:
        # if all ghosts are scared, then our situation is better, so the value is plus
        ghostScore = 1 / (closestGhostDist+1)
    # the higher currentGameState.getScore() is, the better situation the pacman is in
    # the higher minScaredTime is, the better situation the pacman is in
    # the higher remainingFoodCount is, the worse situation the pacman is in because remaining foods more,
    # so the sign is minus
    # the weights are determined by experiments
    return 0.8*currentGameState.getScore()+0.6/(closestGhostDist+1)+0.7*minScaredTime-0.5*remainingFoodCount+ghostScore
    # End your code

# Abbreviation
"""
If you complete this part, please replace scoreEvaluationFunction with betterEvaluationFunction ! !
"""
better = betterEvaluationFunction # betterEvaluationFunction or scoreEvaluationFunction
