###############################################################################
# Wrapper functions of d3rlpy
# implement functions to train and test all models at once
# Models:
#     DQN: Deep Q Network
#     DoubleDQN: Double Deep Q Network
#     DiscreteBCQ: Discrete Batched-Constrained Q Learning
#     DiscreteCQL: Discrete Conservative Q Learning
###############################################################################
from d3rlpy.algos import DQN, DoubleDQN, DiscreteBCQ, DiscreteCQL
from d3rlpy.metrics.scorer import discrete_action_match_scorer
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
import numpy as np
import multiprocessing as mp
from multiprocessing import sharedctypes
from tqdm import tqdm
MP_THREADS = 32
N_ITEMS = 381

class RLModelWrapper:
    def __init__(self, datasetTrain, datasetVal = None):
        """
        datasetTrain: MDPDataset created from states (observations), actions, rewards, terminalFlag
        """
        self.datasetTrain = datasetTrain
        self.datasetVal = datasetVal
        # initialize models
        ## DQN
        self.DQN = DQN(use_gpu = True)
        self.DQN.build_with_dataset(datasetTrain)
        ## DoubleDQN
        self.DoubleDQN = DoubleDQN(use_gpu = True)
        self.DoubleDQN.build_with_dataset(datasetTrain)
        ## DiscreteBCQ
        self.DiscreteBCQ = DiscreteBCQ(use_gpu = True)
        self.DiscreteBCQ.build_with_dataset(datasetTrain)
        ## DiscreteCQL
        self.DiscreteCQL = DiscreteCQL(use_gpu = True)
        self.DiscreteCQL.build_with_dataset(datasetTrain)

    def trainAllModels(self, n_epochs = 10):
        """ train all 4 models (can't do in parallel because memory limit) """
        if self.datasetVal is not None:
#             self.DQN.fit(self.datasetTrain, 
#                          eval_episodes = self.datasetVal,
#                          n_epochs=n_epochs, verbose=False, show_progress=False)
#             self.DoubleDQN.fit(self.datasetTrain, 
#                                eval_episodes = self.datasetVal,
#                                n_epochs=n_epochs, verbose=False, show_progress=False)
            self.DiscreteBCQ.fit(self.datasetTrain, 
                                 eval_episodes = self.datasetVal,
                                 n_epochs=n_epochs, verbose=False, show_progress=False)
#             self.DiscreteCQL.fit(self.datasetTrain, 
#                                  eval_episodes = self.datasetVal,
#                                  n_epochs=n_epochs, verbose=False, show_progress=False)
        else:
#             self.DQN.fit(self.datasetTrain, n_epochs=n_epochs, verbose=False, show_progress=False)
#             self.DoubleDQN.fit(self.datasetTrain, n_epochs=n_epochs, verbose=False, show_progress=False)
            self.DiscreteBCQ.fit(self.datasetTrain, n_epochs=n_epochs, verbose=False, show_progress=False)
#             self.DiscreteCQL.fit(self.datasetTrain, n_epochs=n_epochs, verbose=False, show_progress=False)
    
    @staticmethod
    def calculateItemValue(args_, newStates_shared):
        """ By using a model, calculate value of the itemID given the states in newStates_shared (defined in predict9ItemsbyModel)
        This is used as kernel function for parallel processing in predict9ItemsbyModel
        newStates: np.array or pd.DataFrame
        Return: array of size newDataSet.shape[0], showing values of item
        """
        model, itemID = args_
        tmp = np.ctypeslib.as_array(newStates_shared)
        return list(model.predict_value(tmp, np.array([itemID]*tmp.shape[0])))
    
    @staticmethod
    def best9Items(values):
        """ return 9 itemID that have the highest values 
        itemID = index+1 because itemID is 1-based array
        """
        negValues = [-x for x in values]
        order = np.argsort(negValues)
        highest9 = order[:9]
        return [index+1 for index in highest9]
    
    def predict9ItemsbyModel(self, model, newStates):
        """ predict 9 items by using a single model 
        This is done by calculating values of all items for each sample in newStates. 
        Then for each sample, pick 9 items with the highest values.
        newDataSet: np.array or pd.DataFrame
        """
        # calculate value array for each item, each array has length newStates.shape[0]
        tmp = np.ctypeslib.as_ctypes(newStates)
        newStates_shared = sharedctypes.RawArray(tmp._type_, tmp)
        args = [(model, itemID) for itemID in range(1, N_ITEMS + 1)]
        allItemsValues = []
        print('calculate values for each item ... ')
        for arg in tqdm(args):
            itemValues = self.calculateItemValue(arg, newStates_shared)
            allItemsValues.append(itemValues)
#         pool = mp.Pool(MP_THREADS)
#         allItemsValues = pool.map(self.calculateItemValue, args)
        # now for each sample, find the best 9 items.
        print('for each sample, find best 9 items ...')
        output = []
        for i in tqdm(range(newStates.shape[0])):  # loop thru sample
            values_allItems = [v[i] for v in allItemsValues] # array of 380 item values for this sample
            bestItems = self.best9Items(values_allItems)
            output.append(bestItems)
        return np.array(output)

    def predict9ItemsAllModels(self, newStates):
        res1 = []; res2 = []; res3 = []; res4 = []
#         res1 = self.predict9ItemsbyModel(self.DQN, newStates)
#         res2 = self.predict9ItemsbyModel(self.DoubleDQN, newStates)
        res3 = self.predict9ItemsbyModel(self.DiscreteBCQ, newStates)
#         res4 = self.predict9ItemsbyModel(self.DiscreteCQL, newStates)
        return res1, res2, res3, res4


def predictBestK(model, newStates, K):
    """
    input: model of a class in package d3rlpy
           newStates: np.array or pd.DataFrame
           K: number of best actions to output
    return: 2-D array:
           1st D: sample = newStates.shape[0]
           2nd D: best K items, size = K
    """
    N_ITEMS = 381
    allItemsValues = []
    for itemID in tqdm(range(1, N_ITEMS+1)):
        itemValues = model.predict_value(newStates, np.array([itemID]*newStates.shape[0]))  # values of itemID in all states
        allItemsValues.append(itemValues)
    # now for each sample, find the best K items
    output = []
    for i in tqdm(range(newStates.shape[0])):  # loop thru sample
        negValues_allItems = [-v[i] for v in allItemsValues] # array of 380 item values for this sample
        order = np.argsort(negValues_allItems)
        bestK = [index+1 for index in order[:K]]
        output.append(bestK)
    return np.array(output)


# function to concatenate multiple rows of itemSet into a single set for each user sample
# for each sample:
#     finalSet = []
#     for each step in sample:
#          iterate thru recommended items and add to finalSet if that item is not already in finalSet
def finalizeItemSetsTestSet(statesInput, itemSet, K):
    output = []
    for i in tqdm(range(statesInput.shape[0])):
        # loop through expanded samples
        state = list(statesInput[i])
        step = state[len(state)-1]
        if step==0: # init new finalItemSet
            finalItemSet = []
        # try to add new item to finalItemSet, based on their highest value
        for item in itemSet[i]:
            if item not in finalItemSet:
                finalItemSet.append(item)
                break
        # export finalItemSet once reaching step K-1
        if step==(K-1):
            assert len(finalItemSet)==K
            output.append(finalItemSet)
    return output
