# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


"""
part 2-1
"""

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Begin your code
        # iterate self.iterations times
        for i in range(self.iterations):
            # get a copy from self.values because
            ########################################################################################
            # Use the “batch” version of value iteration where each vector V[K] is
            # computed from a fixed vector V[K-1], not the “online” version
            # where one single weight vector is updated in place.
            # This means that when a state’s value is updated in iteration k
            # based on the values of its successor states, the successor state values used in the
            # value update computation should be those from iteration k-1
            #########################################################################################
            copyValues = self.values.copy()
            # iterate through all states
            for state in self.mdp.getStates():
                # check for terminal
                if self.mdp.isTerminal(state):
                    copyValues[state] = 0
                    continue
                maxQValue = float("-inf")
                # iterate through all possible actions
                for action in self.mdp.getPossibleActions(state):
                    # we want the maximum QValue from all possible actions
                    maxQValue = max(maxQValue, self.getQValue(state, action))
                copyValues[state] = maxQValue
            # update values in iteration k based on the values in iteration k-1
            self.values = copyValues
        # End your code


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        QValue = 0
        # iterate through all possible nextStates, prob means the probability to transit to that nextState
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            QValue += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
        return QValue
        # End your code

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        #check for terminal
        if self.mdp.isTerminal(state):
            return None
        # initialize values to a dictionary
        values = util.Counter()
        # iterate through all possible actions and store the QValue in values
        for action in self.mdp.getPossibleActions(state):
            values[action] = self.getQValue(state, action)
        # argMax returns the key with the highest value
        # find the best action with the highest value
        return values.argMax()
        # End your code

    def getPolicy(self, state):
        """
        The policy is the best action in the given state
        according to the values computed by value iteration.
        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        """
        The q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        return self.computeQValueFromValues(state, action)
