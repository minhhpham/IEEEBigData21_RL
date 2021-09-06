######################################################################################
# Script to prepare data objects for training and testing
#    Usage: from DataPrep import getUserFeaturesTrainSet, getPurchasedItemsTrainSet
#    1. getUserFeaturesTrainSet():
#         return: DataFrame with N_ITEMS+N_USER_PORTRAITS columns
#             first N_ITEMS cols: one hot encoding of clicked items
#             last N_USER_PORTRAITS cols: normalized user portraits to [0,1] range
#         DataFrame shape: (260087, 380+10)
#    2. getPurchasedItemsTrainSet():
#         return: a list, each element is a list of purchased itemIDs by a user
#             each element i of the list corresponds to a user in row i of getUserFeaturesTrainSet()
#         list length: 260087
######################################################################################
import pandas as pd
import numpy as np
import multiprocessing as mp
import functools, pickle
from tqdm import tqdm
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
    Data source: /tf/shared/trainset.csv
    """
    # read data to pd dataframe
    print('reading raw data file ...')
    rawTrainSet = pd.read_csv('../shared/trainset.csv',' ')
    # create output frame
    colNames = ['clickedItem'+str(i+1) for i in range(N_ITEMS)] + ['userPortrait'+str(i+1) for i in range(N_USER_PORTRAITS)]
    output = pd.DataFrame(data = np.zeros(shape = (rawTrainSet.shape[0], N_ITEMS+N_USER_PORTRAITS)), columns = colNames)
    # parse each line in parallel
    # first objects in shared memory for input and output
    print('creating shared memory objects ... ')
    mpManager = mp.Manager()
    inputSharedList = mpManager.list(rawTrainSet.values.tolist())  # for memory efficiency
    outputSharedList = mpManager.list(output.values.tolist())  # shared output as a list (because DataFrame can't)
    p = mp.Pool(N_THREADS)
    print('multiprocessing ... ')
    for i in tqdm(range(rawTrainSet.shape[0])):
        p.apply_async(parseUserFeaturesOneLine, [i, inputSharedList, outputSharedList])
    p.close()
    p.join()
    # convert outputSharedList back to DataFrame
    print('convert to DataFrame ...')
    outputList = [x for x in outputSharedList]
    output = pd.DataFrame(data = outputList, columns = colNames)
#     for i in range(rawTrainSet.shape[0]):
#         parseUserFeaturesOneLine(i, rawTrainSet, out)
    # save to pickle file
    return out
#     output.to_pickle('./data/UserFeaturesTrainSet.pkl')

def parseUserFeaturesOneLine(rowIndex, inputSharedList, outputSharedList):
    """
    Kernel function
    Return: Nothing, only modify outputSharedList[rowIndex]. DataFrames are passed by reference
    Input:
        inputSharedList: trainset or testset DataFrames converted to list
        outputSharedList: Output as mp.Manager.list, each element of list is a list that expects N_ITEMS+N_USER_PORTRAITS element
    ASSUMPTIONS:
        user_click_history is on column index  1 of inputSharedList
        user_portrait is on column index 2 of inputSharedList
    """
    CLICKHIST_INDEX = 1
    PORTRAIT_INDEX = 2
    # parse click history, assuming 
    clickSeries = inputSharedList[rowIndex][CLICKHIST_INDEX].split(',')
    clickedItems = [item.partition(':')[0] for item in clickSeries]
    # add clicked items to output
    for itemID in clickedItems:
        if int(itemID)<=0 or int(itemID)>=N_ITEMS:  # ignore if itemID invalid
            continue
        colIndex = int(itemID) - 1  # index of clicked item on an element of outputSharedList
        outputSharedList[rowIndex][colIndex] = 1
    # parse user portraits
    portraits = inputSharedList[rowIndex][PORTRAIT_INDEX].split(',')
    if len(portraits)!=N_USER_PORTRAITS:
        raise Exception("row "+rowIndex+" of data set does not have the expected number of portrait features")
    # add portrait features to output
    for i in range(N_USER_PORTRAITS):
        colIndex = N_ITEMS + i  # index of feature on an element of outputSharedList
        outputSharedList[rowIndex][colIndex] = int(portraits[i])

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

def prepareUserFeaturesTestSet():
    """
    return: nothing, write to PurchasedItemsTestSet.pkl
    Data source: /tf/shared/track1_testset.csv
    """
    # read data to pd dataframe
    print('reading raw data file ...')
    rawTestSet = pd.read_csv('/tf/shared/track1_testset.csv',' ')
    # create output frame
    colNames = ['clickedItem'+str(i+1) for i in range(N_ITEMS)] + ['userPortrait'+str(i+1) for i in range(N_USER_PORTRAITS)]
    output = pd.DataFrame(data = np.zeros(shape = (rawTestSet.shape[0], N_ITEMS+N_USER_PORTRAITS)), columns = colNames)
    # parse each line in parallel
    # first objects in shared memory for input and output
    print('creating shared memory objects ... ')
    mpManager = mp.Manager()
    inputSharedList = mpManager.list(rawTestSet.values.tolist())  # for memory efficiency
    outputSharedList = mpManager.list(output.values.tolist())  # shared output as a list (because DataFrame can't)
    p = mp.Pool(N_THREADS)
    print('multiprocessing ... ')
    for i in tqdm(range(rawTestSet.shape[0])):
        p.apply_async(parseUserFeaturesOneLine, [i, inputSharedList, outputSharedList])
    p.close()
    p.join()
    # convert outputSharedList back to DataFrame
    output = pd.DataFrame(data = outputSharedList, columns = colNames)
    # write to pkl file
    output.to_pickle('/tf/shared/data/UserFeaturesTestSet.pkl')

        
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

def getUserFeaturesTestSet():
    """
    return: DataFrame with N_ITEMS+N_USER_PORTRAITS columns
        first N_ITEMS cols: one hot encoding of clicked items
        last N_USER_PORTRAITS cols: normalized user portraits
    """
    return pd.read_pickle(DATA_PATH + 'UserFeaturesTestSet.pkl')


if __name__ == "__main__":
    test = prepareUserFeaturesTrainSet()
    print(test)
#     preparePurchasedItemsTrainSet()
#     prepareUserFeaturesTestSet()
