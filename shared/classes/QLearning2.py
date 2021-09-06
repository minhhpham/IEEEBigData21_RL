########################################################################
# Q-Learning Implementation
# train on an array of tuple (stateID, actionID, reward, nextStateID
#      where nextStateID = 0, 1, 2, ... if there is a next state
#           or nextStateID = -1 if there is no next state (game terminated)
########################################################################

import numpy as np
from classes.mpSharedMem import SharedArray
import multiprocessing as mp
from tqdm import tqdm

class QLearning:
    def __init__(self, n_states, n_actions, learning_rate = 0.5, discount_factor = 0.1):
        # create shared memory for QTable (for multiprocessing)
        sharedMemSize = np.dtype('float32').itemsize * n_states * n_actions
        self.sharedMemBuffer = mp.shared_memory.SharedMemory(create = True, size = sharedMemSize)
        # QTable as np.array object on the created shared memory region
        self.QTable = np.ndarray(shape = (n_states, n_actions), dtype = 'float32', buffer = self.sharedMemBuffer.buf)
#         self.QTable = np.ndarray(shape = (n_states, n_actions), dtype = 'float32')
        # initialize QTable to 0
        self.QTable[:] = 0
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.n_states = n_states
        self.n_actions = n_actions
        self.predCache = None    # prediction cache
    
    def train(self, data):
        """
        data: 2d array of shape (,4)
            first column: state ID
            second column: action ID
            third column: reward
            4th column: next state ID
        this function updates the QTable
        """
        alpha = self.alpha
        gamma = self.gamma
        for d in tqdm(data):
            stateID = int(d[0])
            actionID = int(d[1])
            reward = d[2]
            nextStateID = int(d[3])
            if nextStateID>0:
                maxQ_NextState = max(self.QTable[nextStateID])
            else:
                maxQ_NextState = 0
            self.QTable[stateID, actionID] = self.QTable[stateID, actionID] + alpha*(reward + gamma*maxQ_NextState - self.QTable[stateID, actionID])
    
    @staticmethod
    def train_kernel(args):
        """ kernel function for trainParallel 
        args: (sharedBuffer, stateID, actionID, reward, nextStateID, alpha, gamma)
        sharedBuffer: memory region that stores the QTable (self.sharedBuffer)
        """
        sharedBuffer, stateID, actionID, reward, nextStateID, alpha, gamma, n_states, n_actions = args
        QTable = np.ndarray(shape = (n_states, n_actions), dtype = 'float32', buffer = sharedBuffer.buf)
        if nextStateID>0:
            maxQ_NextState = max(QTable[nextStateID])
        else:
            maxQ_NextState = 0
        # calculate new Q Value for QTable[stateID, actionID] and update table
        newQValue = QTable[stateID, actionID] + alpha*(reward + gamma*maxQ_NextState - QTable[stateID, actionID])
        QTable[stateID, actionID] = newQValue

    def trainParallel(self, data):
        """ same as self.train(), but train in parallel 
        by sharing self.QTable as a shared array
        """
        # construct args for parallel processing
        args = [ ( self.sharedMemBuffer, int(d[0]), int(d[1]), int(d[2]), int(d[3]), self.alpha, self.gamma, self.n_states, self.n_actions ) for d in data ]
        #   vars:      sharedQTable        state     action     reward    nextState   alpha       gamma
        # invoke kernel:
        with mp.Pool(32) as pool:
            pool.map(self.train_kernel, args)
    
    def predict(self, stateID):
        """
        determine the best action given stateID
        return: best actionID
        """
        Qvalues = self.QTable[stateID]
        return Qvalues.argmax()
    
    def initPredCache(self):
        """
        init cache for faster predictions
        """
        self.predCache = {}
    
    def predict_value(self, stateID, actionID):
        """
        calculate the value of an action in the given state
        return: value of the actionID
        """
        Qvalues = self.QTable[stateID]
        return Qvalues[actionID]

    def predictBestK(self, stateID, K):
        """
        find the best K actions for the given stateID
        return: list of K actionIDs, ordered from highest value to lowest value
        """
        # check if we already cached the result
        if stateID in self.predCache:
            return self.predCache.get(stateID)
        # if not, calculate results and save to cache
        Qvalues = self.QTable[stateID]
        negQvalues = np.array([-v for v in Qvalues])
        order = np.argsort(negQvalues)
        result = order[:K]
        self.predCache[stateID] = result
        return result
        
    def printQTable(self):
        """ for debugging """
        print(self.QTable)
        