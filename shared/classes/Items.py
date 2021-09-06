#####################################################################
# Class for storing all Item Info
####################################################################
import numpy as np
import pandas as pd

class Items:
    def __init__(self):
        # attributes
        self.itemVectors = None     # Item Vectors, 5 dimensional, np array of float, shape = (381, 4)
        self.itemPrices  = None     # Item Prices
        self.itemLocations = None   # Item Locations
        # read data
        itemInfo = pd.read_csv('/tf/shared/item_info.csv', ' ')
        itemInfo = itemInfo.sort_values(by = 'item_id')
        # parse item Vectors
        itemVectors = []
        for i in range(itemInfo.shape[0]):
            vec = itemInfo.item_vec[i]
            vec = [float(x) for x in vec.split(',')]
            itemVectors.append(vec)
        self.itemVectors = np.array(itemVectors)
        # prices
        self.itemPrices = itemInfo.price
        # locations
        self.itemLocations = itemInfo.location

    def getItemVector(self, itemID):
        """ return item vector of the given itemID """
        if itemID<1 or itemID>len(self.itemPrices):
            raise ValueError('itemID ' + str(itemID) + ' out of range')
        return self.itemVectors[itemID-1]
    
    def getItemPrice(self, itemID):
        """ return the price of the given itemID """
        if itemID<1 or itemID>len(self.itemPrices):
            raise ValueError('itemID ' + str(itemID) + ' out of range')
        return self.itemPrices[itemID-1]

    def getItemLocation(self, itemID):
        """ return location of the given itemID """
        if itemID<1 or itemID>len(self.itemPrices):
            raise ValueError('itemID ' + str(itemID) + ' out of range')
        return self.itemLocations[itemID-1]

