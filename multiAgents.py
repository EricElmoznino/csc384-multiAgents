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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        from util import manhattanDistance

        if successorGameState.isWin(): return 1000
        if successorGameState.isLose(): return -1000

        def less_food(s0, s1):
            f0, f1 = s0.getFood().asList(), s1.getFood().asList()
            f0, f1 = len(f0), len(f1)
            return float(f1 < f0)

        def closest_food(s0, s1):
            p0, p1 = s0.getPacmanPosition(), s1.getPacmanPosition()
            f0, f1 = s0.getFood().asList(), s1.getFood().asList()
            d0, d1 = [manhattanDistance(p0, f) for f in f0], [manhattanDistance(p1, f) for f in f1]
            d0, d1 = min(d0), min(d1)
            return float(d1 < d0)

        def closest_ghost(s0, s1):
            p0, p1 = s0.getPacmanPosition(), s1.getPacmanPosition()
            g0, g1 = s0.getGhostStates(), s1.getGhostStates()
            g0, g1 = [g.getPosition() for g in g0], [g.getPosition() for g in g1]
            g0, g1 = [manhattanDistance(p0, g) for g in g0], [manhattanDistance(p1, g) for g in g1]
            if len(g1) == 0 or len(g0) == 0:
                return 1.0
            return float(g1 > g0)

        lf = 2 * less_food(currentGameState, successorGameState)
        cf = 1 * closest_food(currentGameState, successorGameState)
        cg = 0.9 * closest_ghost(currentGameState, successorGameState)

        return lf + cf + cg

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
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def terminal(s, d):
            return d == 0 or s.isWin() or s.isLose()

        def df_min_max(s, a, d):
            bm = None
            if terminal(s, d):
                return bm, self.evaluationFunction(s)
            if a == 0:
                bv = -1000000
            else:
                bv = 1000000
            an = (a + 1) % s.getNumAgents()
            dn = d - 1 if an == 0 else d
            for m in s.getLegalActions(a):
                sn = s.generateSuccessor(a, m)
                _, v = df_min_max(sn, an, dn)
                if (a == 0 and bv < v) or (a > 0 and bv > v):
                    bm, bv = m, v
            return bm, bv

        return df_min_max(gameState, 0, self.depth)[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def terminal(s, d):
            return d == 0 or s.isWin() or s.isLose()

        def df_min_max(s, a, d, alpha, beta):
            bm = None
            if terminal(s, d):
                return bm, self.evaluationFunction(s)
            if a == 0:
                bv = -1000000
            else:
                bv = 1000000
            an = (a + 1) % s.getNumAgents()
            dn = d - 1 if an == 0 else d
            for m in s.getLegalActions(a):
                sn = s.generateSuccessor(a, m)
                _, v = df_min_max(sn, an, dn, alpha, beta)
                if a == 0:
                    if bv < v:
                        bm, bv = m, v
                    if bv >= beta:
                        return bm, bv
                    alpha = max(alpha, bv)
                else:
                    if bv > v:
                        bm, bv = m, v
                    if bv <= alpha:
                        return bm, bv
                    beta = min(beta, bv)
            return bm, bv

        return df_min_max(gameState, 0, self.depth, -1000000, 1000000)[0]

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
        def terminal(s, d):
            return d == 0 or s.isWin() or s.isLose()

        def df_min_max(s, a, d):
            bm = None
            if terminal(s, d):
                return bm, self.evaluationFunction(s)
            if a == 0:
                bv = -1000000
            else:
                bv = 0
            an = (a + 1) % s.getNumAgents()
            dn = d - 1 if an == 0 else d
            for m in s.getLegalActions(a):
                sn = s.generateSuccessor(a, m)
                _, v = df_min_max(sn, an, dn)
                if a == 0 and bv < v:
                    bm, bv = m, v
                elif a > 0:
                    bv += float(v) / len(s.getLegalActions(a))
            return bm, bv

        return df_min_max(gameState, 0, self.depth)[0]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    from util import manhattanDistance

    if currentGameState.isWin(): return 1000000
    if currentGameState.isLose(): return -1000000

    def num_food(s):
        food = s.getFood().asList()
        food = len(food)
        food = 1.0 / food
        return food

    def closest_food(s):
        pos = s.getPacmanPosition()
        food = s.getFood().asList()
        dist = [manhattanDistance(pos, f) for f in food]
        dist = min(dist)
        return 1.0 / dist

    def closest_ghost(s):
        pos = s.getPacmanPosition()
        ghost = s.getGhostStates()
        ghost = [g.getPosition() for g in ghost]
        dist = [manhattanDistance(pos, g) for g in ghost] + [1000000]
        dist = min(dist)
        return 1.0 - 1.0 / dist

    score = 1.0 * currentGameState.getScore() / 1000.0
    lf = 0.0 * num_food(currentGameState)
    cf = 0.0 * closest_food(currentGameState)
    cg = 0.0 * closest_ghost(currentGameState)

    return score + lf + cf + cg

# Abbreviation
better = betterEvaluationFunction
