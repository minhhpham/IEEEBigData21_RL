########################################################################
# Q-Learning Implementation
# assume: next state = actionID
########################################################################

import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, learning_rate = 0.5, discount_factor = 0.1):
        self.QTable = np.zeros(shape = (n_states, n_actions))
        self.alpha = learning_rate
        self.gamma = discount_factor
    
    def train(self, data):
        """
        data: 2d array of shape (,3)
            first column: state ID
            second column: action ID
            third column: reward
        this function updates the QTable
        """
        alpha = self.alpha
        gamma = self.gamma
        for d in data:
            stateID = d[0]
            actionID = d[1]
            reward = d[2]
            nextStateID = actionID
            maxQ_NextState = max(self.QTable[nextStateID])
            self.QTable[stateID, actionID] = self.QTable[stateID, actionID] + alpha*(reward + gamma*maxQ_NextState - self.QTable[stateID, actionID])
    
    def nextStep(self, stateID):
        """
        determine the best next step given stateID
        return: best actionID
        """
        Qvalues = self.QTable[stateID]
        return Qvalues.argmax()
    
    def nextNSteps(self, stateID, N):
        """
        find N non-repeated next step
        return: array of N
        """
        steps = []
        for _ in range(N):
            Qvalues = self.QTable[stateID]
            negQvalues = [-x for x in Qvalues]
            order = np.argsort(negQvalues)
            for action in order:
                if action not in steps:
                    steps.append(action)
                    break
        return steps    
        
        
        
    def printQTable(self):
        """ for debugging """
        print(self.QTable)
        