########################################################################
# Classes for generating prediction, given a model, for different setups of the problem
########################################################################
from classes.ItemSet import ItemSet3
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import pickle
# from classes.mpSharedMem import SharedArray

def predictValue(args):
    """
    calculate values of action given all states.
    kernel function for multiprocessing
    action: single integer
    """
    statesShared, action = args
    states = statesShared.getSharedArray()
    with open('/tmp/model.pkl', 'rb') as file:
        model = pickle.load(file)
    action = np.array([action]*states.shape[0])
    values = list(model.predict_value(states, action))
    return values

class Model3D:
    def __init__(self, model):
        """
        model: has to be one of the model classes from package d3rlpy
        required methods: model.predict_value()
        """
        self.model = model
        itemSets = ItemSet3()
        self.itemSets = itemSets
        self.candidateSets = itemSets.getCandidateItemSets()   # candidate sets for prediction

    def predictValue(self, states, action):
        """
        calculate values of AN action given all states.
        kernel function for multiprocessing
        action: single integer
        """
        action = np.array([action]*states.shape[0])
        values = list(self.model.predict_value(states, action))
        return values

    @staticmethod
    def orderDec9(array):
        """ sort an array decreasing, return order index, up to the highest 9 indices
        array: numpy array
        """
        negArray = np.array([-x for x in array])
        return negArray.argsort()[:9]

    def predict(self, states):
        """
        create recommended items.
        input states: numpy array, grouped by every 3 rows according to step 0, 1, 2
        for each samples in states:
            calculate value of each candidate item sets. Select top item sets so that no item is duplicated
        output: list (length states.shape[0]/3) of list of itemID (of length 9)
        """
        output = []
        values_allActions = []    # values of entire sample by action. row: action, col: sample
        print('calculating values for each action ... ')
        for action in tqdm(self.candidateSets):
            values_allActions.append(self.predictValue(states, action))
#         # save model to disk to share between processes
#         with open('/tmp/model.pkl', 'wb') as file:
#             pickle.dump(self.model, file, protocol=pickle.HIGHEST_PROTOCOL)
#         statesShared = SharedArray(states)
#         args = [(statesShared, action) for action in self.candidateSets]
#         pool = mp.Pool(32)
#         values_allActions = pool.map(predictValue, args)
        # find optimal 3 actions
        # get set of 3 actions such that no item are duplicated
        print('find optimal item set for each sample')
        for i in tqdm(range(states.shape[0])):
        # loop thru samples
            if i%3==0:
                recItems = []    # output
                # first step 0
                valuesByAction = [v[i] for v in values_allActions]
                indices9  = self.orderDec9(valuesByAction)    # index of 9 best actions
                itemSetID = self.candidateSets[indices9[0]]      # best action
                itemID1, itemID2, itemID3 = self.itemSets.getItemSet(itemSetID)
                recItems.extend([itemID1, itemID2, itemID3])
                # second step 1
                valuesByAction = [v[i+1] for v in values_allActions]
                indices9  = self.orderDec9(valuesByAction)    # index of 9 best actions
                for j in range(9):
                    itemSetID = self.candidateSets[indices9[j]]
                    itemID1, itemID2, itemID3 = self.itemSets.getItemSet(itemSetID)
                    if (itemID1 not in recItems) and (itemID2 not in recItems) and (itemID3 not in recItems):
                    # make sure no duplicated item
                        recItems.extend([itemID1, itemID2, itemID3])
                        break
                # third step 2
                valuesByAction = [v[i+2] for v in values_allActions]
                indices9  = self.orderDec9(valuesByAction)    # index of 9 best actions
                for j in range(9):
                    itemSetID = self.candidateSets[indices9[j]]
                    itemID1, itemID2, itemID3 = self.itemSets.getItemSet(itemSetID)
                    if (itemID1 not in recItems) and (itemID2 not in recItems) and (itemID3 not in recItems):
                    # make sure no duplicated item
                        recItems.extend([itemID1, itemID2, itemID3])
                        break
                # append to output
                assert len(recItems)==9
                output.append(recItems)
        # done looping thru samples
        assert len(output) == states.shape[0]/3
        return output
                
    def predictPartial(self, states):
        """
        create recommended items.
        input states: numpy array
        for each samples in states:
            calculate value of each candidate item sets. 
        For each step in a sample: Select top item sets so that no item is duplicated
        output: list (length states.shape[0]) of list (of length 9)
        """
        output = []
        values_allActions = []    # values of entire sample by action. row: action, col: sample 
        print('calculating values for each action ... ')
        for action in tqdm(self.candidateSets):
            values_allActions.append(self.predictValue(states, action))
        # find optimal 3 actions
        # get set of 3 actions such that no item are duplicated
        print('find optimal item set for each sample')
        recItems = []
        for i in tqdm(range(states.shape[0])):
        # loop thru samples
            # find step
            ncol = states.shape[1]
            if states[i][ncol-3]==1:
                step = 0
            elif states[i][ncol-2]==1:
                step = 1
            else:
                step = 2
            if step==0:  # start collecting recommended item set
                recItems = []
            # values of all actions
            valueByAction = [v[i] for v in values_allActions]
            indices9  = self.orderDec9(valueByAction)    # index of 9 best actions
            for j in range(9):
                itemSetID = self.candidateSets[indices9[j]]
                itemID1, itemID2, itemID3 = self.itemSets.getItemSet(itemSetID)
                if (itemID1 not in recItems) and (itemID2 not in recItems) and (itemID3 not in recItems):
                    output.append(list(self.itemSets.getItemSet(itemSetID)))
                    recItems.extend([itemID1, itemID2, itemID3])
                    break
        # done
        assert len(output)==states.shape[0]
        return output
    

    def findCandidateItemSets(self, sample_states, NCandidates = 1000):
        """ calculate and save best actions from a small sample states"""
        allItemSets = self.itemSets.getAllItemSets()  # all actions possible
        # first calculate average value of each action
        avgValues_allActions = []  # average value of all actions (array of len(allItemSets))
        n_samples = sample_states.shape[0]
        for action in tqdm(allItemSets):
            if action==112367:
                continue
            values = self.predictValue(sample_states, action)  # array of size n_samples
            avgValues_allActions.append(sum(values)/n_samples)
        # find top NCandidates actions
        negValues = np.array([-v for v in avgValues_allActions])
        order = np.argsort(negValues)
        topIndices = order[:NCandidates]
        # get top itemSetID
        bestItemSets = []
        for i in topIndices:
            bestItemSets.append(allItemSets[i])
        # save these top itemSetID     
        self.candidateSets = bestItemSets
        print('total candidate actions : ' + str(len(self.candidateSets)))

                        
                        