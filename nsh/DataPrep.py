######################################################################################
# Script to prepare data objects for training and testing
#    Usage: from DataPrep import getUserFeaturesTrainSet, getPurchasedItemsTrainSet
#    1. getUserFeaturesTrainSet():
#         return: DataFrame with N_ITEMS+N_USER_PORTRAITS columns
#             first N_ITEMS cols: one hot encoding of clicked items
#             last N_USER_PORTRAITS cols: normalized user portraits to [0,1] range
#
#    2. getPurchasedItemsTrainSet():
#         return: a list, each element is a list of purchased itemIDs by a user
#             list length is same as PurchasedItemsTrainSet's nrow
######################################################################################
import pandas as pd
import numpy as np
import multiprocessing as mp
import functools, pickle
N_ITEMS = 380
N_USER_PORTRAITS = 10
N_THREADS = mp.cpu_count() - 1
print('Number of Multiprocessing threads: ' + str(N_THREADS))
DATA_PATH = '/tf/shared/data/'
##################################################################

def prepareUserFeaturesTrainSet():
    """
    return: nothing, save to UserFeaturesTrainSet.pkl
        data frame with N_ITEMS+N_USER_PORTRAITS columns
        first N_ITEMS cols: one hot encoding of clicked items
        last N_USER_PORTRAITS cols: normalized user portraits
    Data source: ../shared/trainset.csv
    """
    # read data to pd dataframe
    rawTrainSet = pd.read_csv('../shared/trainset.csv',' ')
    # create output frame
    colNames = ['clickedItem'+str(i+1) for i in range(N_ITEMS)] + ['userPortrait'+str(i+1) for i in range(N_USER_PORTRAITS)]
    out = pd.DataFrame(data = np.zeros(shape = (rawTrainSet.shape[0], N_ITEMS+N_USER_PORTRAITS)), columns = colNames)
    # parse each line in parallel
    kernelFunc = functools.partial(parseUserFeaturesOneLine, rawDataSet = rawTrainSet, output = out) # set default values for input and output of the kernel function
    with mp.Pool(N_THREADS) as p:
        # now we map row index to the kernel function
        p.map(kernelFunc, list(range(rawTrainSet.shape[0])))
    # save to pickle file
    out.to_pickle('/tf/shared/data/UserFeaturesTrainSet.pkl')

def parseUserFeaturesOneLine(rowIndex, rawDataSet, output):
    """
    Kernel function
    Return: Nothing, only modify output[rowIndex]. DataFrames are passed by reference
    Input:
        rawDataSet: trainset or testset DataFrames
        output: Output DataFrame, expect N_ITEMS+N_USER_PORTRAITS columns
    """
    # parse click history
    clickSeries = rawDataSet.user_click_history[rowIndex].split(',')
    clickedItems = [item.partition(':')[0] for item in clickSeries]
    # add clicked items to output
    for itemID in clickedItems:
        if int(itemID)<=0 or int(itemID)>=N_ITEMS:  # ignore if itemID invalid
            continue
        colName = 'clickedItem' +  itemID
        output[colName][rowIndex] = 1
    # parse user portraits
    portraits = rawDataSet.user_protrait[rowIndex].split(',')
    if len(portraits)!=N_USER_PORTRAITS:
        raise Exception("row "+rowIndex+" of data set does not have the expected number of portrait features")
    # add portrait features to output
    for i in range(N_USER_PORTRAITS):
        colName = 'userPortrait' + str(i+1)
        output[colName][rowIndex] = int(portraits[i])

def preparePurchasedItemsTrainSet():
    """
    return: nothing, write to PurchasedItemsTrainSet.pkl
    Data source: /tf/shared/trainset.csv
    """
    # read data to pd dataframe
    rawTrainSet = pd.read_csv('/tf/shared/trainset.csv',' ')
    output = []
    for i in range(rawTrainSet.shape[0]):
        # parse each line
        exposedItems = rawTrainSet.exposed_items[i]
        labels = rawTrainSet.labels[i]
        exposedItems = exposedItems.split(',')
        labels = labels.split(',')
        purchasedItems = []
        for j in range(len(labels)):
            if int(labels[j])==1:
                # item is purchased, append it to the purchasedItems list
                purchasedItems.append(int(exposedItems[j]))
        # append the list of this row to output
        output.append(purchasedItems)
    # write to pkl file
    with open(DATA_PATH + 'PurchasedItemsTrainSet.pkl', 'wb') as f:
        pickle.dump(output, f)


def getUserFeaturesTrainSet():
    """
    return: DataFrame with N_ITEMS+N_USER_PORTRAITS columns
        first N_ITEMS cols: one hot encoding of clicked items
        last N_USER_PORTRAITS cols: normalized user portraits
    """
    return pd.read_pickle(DATA_PATH + 'UserFeaturesTrainSet.pkl')

def getPurchasedItemsTrainSet():
    """
    return: a list, each element is a list of purchased itemID by a user
    list length is same as PurchasedItemsTrainSet's nrow
    """
    file = open(DATA_PATH + 'PurchasedItemsTrainSet.pkl', 'rb')
    data = pickle.load(file)
    file.close()
    return data


if __name__ == "__main__":
    prepareUserFeaturesTrainSet()