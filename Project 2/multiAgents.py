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
import random
import util

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
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        # print(successorGameState)
        # print(newPos)
        # print(newFood)
        # print(newGhostStates)
        # print(newScaredTimes)

        "*** YOUR CODE HERE ***"

        food_position_list = newFood.asList()

        # ALWAYS RUN FORREST
        if action == "Stop":
            return -1000000

        num_of_ghosts = successorGameState.getNumAgents() - 1
        for i in range(num_of_ghosts):
            # If a ghost is next to you RUN AWAY FORREST
            ghost_position = successorGameState.getGhostPosition(i+1)
            if manhattanDistance(ghost_position, newPos) == 1:
                return -1000000
        # If the foodlist is empty, STOP FORREST
        if len(food_position_list) == 0:
            return 1000000

        food_distances_list = []

        for food in food_position_list:
            food_distances_list.append(manhattanDistance(food, newPos))

        # distance to closest foot (the smaller the better)
        food_distance_score_1 = min(food_distances_list)
        # average distance from every food (the smaller the better)
        food_distance_score_2 = (
            sum(food_distances_list))/len(food_distances_list)
        # print(successorGameState.getScore())
        # print(food_distance_score_1)
        return 80*successorGameState.getScore() - 15*food_distance_score_1 - 5*food_distance_score_2


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agent_index, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.maxval(gameState, 0, 0)[0]

    def minimax(self, gameState, agent_index, iteration):

        if iteration == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agent_index == 0:
            return self.maxval(gameState, agent_index, iteration)[1]
        else:
            return self.minval(gameState, agent_index, iteration)[1]

    def maxval(self, gameState, agent_index, iteration):
        all_actions = []
        for action in gameState.getLegalActions(agent_index):
            suc_state = gameState.generateSuccessor(agent_index, action)
            suc_index = (iteration + 1) % gameState.getNumAgents()
            suc_iteration = iteration + 1
            suc_action_and_value = (action, self.minimax(
                suc_state, suc_index, suc_iteration))
            all_actions.append(suc_action_and_value)
        return max(all_actions, key=lambda x: x[1])

    def minval(self, gameState, agent_index, iteration):
        all_actions = []
        for action in gameState.getLegalActions(agent_index):
            suc_state = gameState.generateSuccessor(agent_index, action)
            suc_index = (iteration + 1) % gameState.getNumAgents()
            suc_iteration = iteration + 1
            suc_action_and_value = (action, self.minimax(
                suc_state, suc_index, suc_iteration))
            all_actions.append(suc_action_and_value)
        return min(all_actions, key=lambda x: x[1])


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxval(gameState, 0, 0, float(-10000000), float(10000000))[0]

    def ab_pruning(self, gameState, agent_index, iteration, a, b):
        if iteration == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agent_index == 0:
            return self.maxval(gameState, agent_index, iteration, a, b)[1]
        else:
            return self.minval(gameState, agent_index, iteration, a, b)[1]

    def maxval(self, gameState, agent_index, iteration, a, b):
        best_action = (None, float(-10000000))
        for action in gameState.getLegalActions(agent_index):
            suc_state = gameState.generateSuccessor(agent_index, action)
            suc_index = (iteration + 1) % gameState.getNumAgents()
            suc_iteration = iteration + 1
            suc_action_and_value = (action, self.ab_pruning(
                suc_state, suc_index, suc_iteration, a, b))
            best_action = max(
                best_action, suc_action_and_value, key=lambda x: x[1])

            # Change a if needed
            if best_action[1] > b:
                return best_action
            else:
                a = max(a, best_action[1])
        return best_action

    def minval(self, gameState, agent_index, iteration, a, b):
        best_action = (None, float(10000000))
        for action in gameState.getLegalActions(agent_index):
            suc_state = gameState.generateSuccessor(agent_index, action)
            suc_index = (iteration + 1) % gameState.getNumAgents()
            suc_iteration = iteration + 1
            suc_action_and_value = (action, self.ab_pruning(
                suc_state, suc_index, suc_iteration, a, b))
            best_action = min(
                best_action, suc_action_and_value, key=lambda x: x[1])

            # Change b if needed
            if best_action[1] < a:
                return best_action
            else:
                b = min(b, best_action[1])
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # get action function
        actions = gameState.getLegalActions(0)
        all_actions_and_values = []
        for action in actions:
            all_actions_and_values.append((action, self.expval(
                gameState.generateSuccessor(0, action), 1, 1)))
        return max(all_actions_and_values, key=lambda x: x[1])[0]

    def maxval(self, gameState, agent_index, iteration):
        agent_index = 0
        legal_actions = gameState.getLegalActions(agent_index)
        all_actions = []
        # if no more new states return the evaluation function value..
        if len(legal_actions) == 0 or iteration == self.depth:
            return self.evaluationFunction(gameState)
        # ...otherwise retrun expected function value (max)
        for action in legal_actions:
            suc_state = gameState.generateSuccessor(agent_index, action)
            suc_index = agent_index + 1
            suc_iteration = iteration + 1
            suc_value = self.expval(suc_state, suc_index, suc_iteration)
            all_actions.append(suc_value)
        return(max(all_actions))

    def expval(self, gameState, agent_index, iteration):

        legal_actions = gameState.getLegalActions(agent_index)
        sum_of_agents = gameState.getNumAgents()

        # if no more new states return the evaluation function value
        if len(legal_actions) == 0:
            return self.evaluationFunction(gameState)

        exp_value = 0
        for action in legal_actions:
            # if its the pacman agent get maxval..
            if agent_index == sum_of_agents - 1:
                action_exp_value = self.maxval(gameState.generateSuccessor(agent_index, action),
                                               agent_index,  iteration)
            # ..else get expval
            else:
                action_exp_value = self.expval(gameState.generateSuccessor(agent_index, action),
                                               agent_index + 1, iteration)
            exp_value += action_exp_value

        return exp_value/len(legal_actions)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    def food_score(gameState):
        # the closer the foods are the higher the score
        food_distances_list = []
        food_list = currentGameState.getFood().asList()
        if len(food_list) > 0:
            for food in food_list:
                food_distances_list.append(
                    1.0/manhattanDistance(currentGameState.getPacmanPosition(), food))
                food_score = (sum(food_distances_list) /
                              len(food_distances_list))
                return food_score
        else:
            return 1000000

    def capsule_score(currentGameState):
        totalCapsules = len(currentGameState.getCapsules())
        return totalCapsules*20

    def ghost_score(currentGameState):
        ghost_score = 0
        # remove score the closer the ghosts are (exponential decrease the closer the ghosts are, or increase if the ghosts are scared)
        for ghost in currentGameState.getGhostStates():
            disGhost = manhattanDistance(
                currentGameState.getPacmanPosition(), ghost.getPosition())
            if ghost.scaredTimer > 1:
                # if a scared ghost is pretty close, eat it!
                if disGhost > 3:
                    ghost_score += pow(6 - disGhost, 2)
            else:
                # if a ghost is pretty close, run!
                if disGhost > 6:
                    ghost_score -= pow(6 - disGhost, 2)
        return ghost_score

    return currentGameState.getScore() + ghost_score(currentGameState) + food_score(currentGameState) + capsule_score(currentGameState)


# Abbreviation
better = betterEvaluationFunction
