########################################################################
# Module to create and index item sets
# Usage: itemSet = ItemSet3(N_ITEMS) # default N_ITEMS = 381
#        itemSet.getSetID((1,2,3)) # translate this set to a single setID
#        itemSet.getItemSet(1)     # retrieve the itemset of setID 1
########################################################################

import sys
sys.path.append('/tf/shared/')
import numpy as np
from itertools import combinations
from tqdm import tqdm
from random import randrange
from DataPrep import getExposedItemsTrainSet
import pickle, os
from classes.Items import Items

class ItemSet3:
    """ Sets of 3 items """
    def __init__(self, N_ITEMS = 381):
        """
        Create hash tables of all possible combinations.
        One table for (item1, item2, item3) --> setID lookup
        One table for setID --> (item1, item2, item3) lookup
        itemID are 1-based index
        """
        self.hashItemIDs = {}   # hash table 1, (item1, item2, item3) --> setID
        self.hashSetID = {}     # hash table 2, setID --> (item1, item2, item3)
        self.candidateItemSets = None  # candidates of item sets for prediction. dict: {itemSetID: count}
        self.items = Items()
        allItemIDs = list(range(1, N_ITEMS+1))
        allItemSets = list(combinations(allItemIDs, 3))
        itemSetID = 0
        # all itemSet hash maps
        if os.path.isfile('/tf/shared/data/ItemSets3Dict.pkl'):
            with open('/tf/shared/data/ItemSets3Dict.pkl', 'rb') as file:
                self.hashItemIDs, self.hashSetID = pickle.load(file)
        # retrieve candidate itemSets
        if os.path.isfile('/tf/shared/data/commonItemSets3.pkl'):
            with open('/tf/shared/data/commonItemSets3.pkl', 'rb') as file:
                self.candidateItemSets = pickle.load(file)

    def getLocationBySetID(setID):
        item1, item2, item3 = self.getItemSet(setID)
        return self.items.getItemLocation(item1)
    
    def getLocationByItem(itemID):
        return self.items.getItemLocation(itemID)
    
    def getSetID(self, itemSet):
        """ Input: itemSet, for example (1,2,3)
        Return: setID
        """
        itemSet_ = list(itemSet)
        itemSet_.sort()
        itemSet_ = tuple(itemSet_)
        return self.hashItemIDs[itemSet_]

    def getItemSet(self, setID):
        """ Input: setID
        Return: (itemID1, itemID2, itemID3)
        """
        return self.hashSetID[setID]

    def getNSets(self):
        """return number of itemSets"""
        return len(self.hashSetID)
    
    def getCandidateItemSets(self):
        """ return list of candidate itemSetID whose frequency in trainset are high """
        return list(self.candidateItemSets.keys())
    
    def getAllItemSets(self):
        """ return list of all itemSetIDS observed from trainset """
        return list(self.hashSetID.keys())

def testItemSet3():
    recItems, _ = getExposedItemsTrainSet()
    itemSet3 = ItemSet3()
    print('testing ...')
    for items9 in tqdm(recItems):
        items = [items9[0], items9[1], items9[2]]
        setID = itemSet3.getSetID(items)
        items_out = itemSet3.getItemSet(setID)
        if items[0] not in items_out or items[1] not in items_out or items[2] not in items_out:
            print('test failed: ' + str(items))
            return
        items = [items9[3], items9[4], items9[5]]
        setID = itemSet3.getSetID(items)
        items_out = itemSet3.getItemSet(setID)
        if items[0] not in items_out or items[1] not in items_out or items[2] not in items_out:
            print('test failed: ' + str(items))
            return
        items = [items9[6], items9[7], items9[8]]
        setID = itemSet3.getSetID(items)
        items_out = itemSet3.getItemSet(setID)
        if items[0] not in items_out or items[1] not in items_out or items[2] not in items_out:
            print('test failed: ' + str(items))
            return
    print('test passed')
    candidates = itemSet3.getCandidateItemSets()
    print('number of candidate item sets: ' + str(len(candidates)))

def findAllItemSet3():
    """
    find  all Item sets from trainset
    store these sets at /tf/shared/data/ItemSets3Dict.pkl in the form of item hash and setID hash
    """
    recItems, purLabels = getExposedItemsTrainSet()
    hashItemIDs = {}  # output
    hashSetID = {}    # output
    itemSetID = 0     # setID counter
    sanityFails = 0   # sanity check for location
    itemsInfo = Items()
    for i in tqdm(range(len(recItems))):
        items = recItems[i]
        label = purLabels[i]
        # process first 3 items
        itemSet = [items[0], items[1], items[2]]
        itemSet.sort()
        itemSet = tuple(itemSet)
        # sanity check location
        if not (itemsInfo.getItemLocation(itemSet[0])==itemsInfo.getItemLocation(itemSet[1]) and itemsInfo.getItemLocation(itemSet[0])==itemsInfo.getItemLocation(itemSet[2])):
            # failed
            sanityFails += 1
        else:
            if (itemSet) not in hashItemIDs:
                hashItemIDs[itemSet] = itemSetID
                hashSetID[itemSetID] = itemSet
                itemSetID = itemSetID + 1
        # process second 3 items
        itemSet = [items[3], items[4], items[5]]
        itemSet.sort()
        itemSet = tuple(itemSet)
        # sanity check location
        if not (itemsInfo.getItemLocation(itemSet[0])==itemsInfo.getItemLocation(itemSet[1]) and itemsInfo.getItemLocation(itemSet[0])==itemsInfo.getItemLocation(itemSet[2])):
            # failed
            sanityFails += 1
        else:
            if (itemSet) not in hashItemIDs:
                hashItemIDs[itemSet] = itemSetID
                hashSetID[itemSetID] = itemSet
                itemSetID = itemSetID + 1
        # process last 3 items
        itemSet = [items[6], items[7], items[8]]
        itemSet.sort()
        itemSet = tuple(itemSet)
        # sanity check location
        if not (itemsInfo.getItemLocation(itemSet[0])==itemsInfo.getItemLocation(itemSet[1]) and itemsInfo.getItemLocation(itemSet[0])==itemsInfo.getItemLocation(itemSet[2])):
            # failed
            sanityFails += 1
        else:
            if (itemSet) not in hashItemIDs:
                hashItemIDs[itemSet] = itemSetID
                hashSetID[itemSetID] = itemSet
                itemSetID = itemSetID + 1
    print('total number of sets: ' + str(itemSetID))
    print('total fails of location check: ' + str(sanityFails))
    # save output
    with open('/tf/shared/data/ItemSets3Dict.pkl', 'wb') as file:
        pickle.dump((hashItemIDs, hashSetID), file, protocol=pickle.HIGHEST_PROTOCOL)
    
def findCommonItemSet3():
    """
    find the most common Item Sets
    store these sets at /tf/shared/data/commonItemSets3.pkl
    """
    CUTOFF_FREQ = 10    # lowest frequency to be kept in the common set
    # get Train Data
    recItems, purLabels = getExposedItemsTrainSet()
    # use hash table to count frequencies of Item Sets
    hashTable = {}
    itemSets = ItemSet3()
    for i in tqdm(range(len(recItems))):
        items = recItems[i]
        label = purLabels[i]
        # count first set
        setID = itemSets.getSetID((items[0], items[1], items[2]))
        if setID in hashTable:
            hashTable[setID] = hashTable[setID] + 1
        else:
            hashTable[setID] = 1
        # count second set
        setID = itemSets.getSetID((items[3], items[4], items[5]))
        if setID in hashTable:
            hashTable[setID] = hashTable[setID] + 1
        else:
            hashTable[setID] = 1
        # count third set
        setID = itemSets.getSetID((items[6], items[7], items[8]))
        if setID in hashTable:
            hashTable[setID] = hashTable[setID] + 1
        else:
            hashTable[setID] = 1
    # remove sets with low counts
    staticKeys = list(hashTable.keys())
    for setID in tqdm(staticKeys):
        if hashTable[setID]<CUTOFF_FREQ:
            del hashTable[setID]
    # output
    with open('/tf/shared/data/commonItemSets3.pkl', 'wb') as file:
        pickle.dump(hashTable, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    findAllItemSet3()
    findCommonItemSet3()
    testItemSet3()