######################################################################################
# Script to prepare data objects for training and testing
#    Usage: from DataPrep import getUserFeaturesTrainSet, getPurchasedItemsTrainSet, getUserFeaturesTestSet
#    1. getUserFeaturesTrainSet():
#         return: DataFrame with N_ITEMS+N_USER_PORTRAITS columns
#             first N_ITEMS cols: one hot encoding of clicked items
#             last N_USER_PORTRAITS cols: normalized user portraits to [0,1] range
#         DataFrame shape: (260087, 380+10)
#    2. getPurchasedItemsTrainSet():
#         return: a list, each element is a list of purchased itemIDs by a user
#             each element i of the list corresponds to a user in row i of getUserFeaturesTrainSet()
#         list length: 260087
#    3. getUserFeaturesTestSet():
#         return: DataFrame with N_ITEMS+N_USER_PORTRAITS columns
#             first N_ITEMS cols: one hot encoding of clicked items
#             last N_USER_PORTRAITS cols: normalized user portraits to [0,1] range
#    4. getClusterLabels():
#       return: (model, labels)
#             model : model for testset prediction
#             labels: numpy array of labels of clusters from the trainset
######################################################################################
import pandas as pd
import numpy as np
import multiprocessing as mp
import functools, pickle
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
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
    rawTrainSet = pd.read_csv('/tf/shared/trainset.csv',' ')
    # create output frame
    colNames = ['clickedItem'+str(i+1) for i in range(N_ITEMS)] + ['userPortrait'+str(i+1) for i in range(N_USER_PORTRAITS)]
    output = pd.DataFrame(data = np.zeros(shape = (rawTrainSet.shape[0], N_ITEMS+N_USER_PORTRAITS)), columns = colNames)
    # parse each line in parallel
    # first objects in shared memory for input and output
    print('creating shared memory objects ... ')
    inputList = rawTrainSet.values.tolist()  # for memory efficiency
    p = mp.Pool(N_THREADS)
    print('multiprocessing ... ')
    outputList = p.map(parseUserFeaturesOneLine, inputList)
    # convert outputSharedList back to DataFrame
    print('convert to DataFrame ...')
    output = pd.DataFrame(data = outputList, columns = colNames)
    # normalize the portraits columns
    for i in range(N_USER_PORTRAITS):
        colName = 'userPortrait' + str(i+1)
        scaler = MinMaxScaler()
        output[colName] = scaler.fit_transform(output[colName].values.reshape(-1,1))
    # save to pickle file
    output.to_pickle('/tf/shared/data/UserFeaturesTrainSet.pkl')

def parseUserFeaturesOneLine(inputArray):
    """
    Kernel function
    Return: list of length N_ITEMS + N_USER_PORTRAITS 
    Input:
        inputArray: an array as a row of trainset or testset raw data
    ASSUMPTIONS:
        user_click_history is on column index  1 of inputArray
        user_portrait is on column index 2 of inputArray
    """
    CLICKHIST_INDEX = 1
    PORTRAIT_INDEX = 2
    output = [0]*(N_ITEMS + N_USER_PORTRAITS)
    # parse click history, assuming 
    clickSeries = inputArray[CLICKHIST_INDEX].split(',')
    clickedItems = [item.partition(':')[0] for item in clickSeries]
    # add clicked items to output
    for itemID in clickedItems:
        if int(itemID)<=0 or int(itemID)>=N_ITEMS:  # ignore if itemID invalid
            continue
        colIndex = int(itemID) - 1  # index of clicked item on an element of outputSharedList
        output[colIndex] = 1
    # parse user portraits
    portraits = inputArray[PORTRAIT_INDEX].split(',')
    if len(portraits)!=N_USER_PORTRAITS:
        raise Exception("row "+rowIndex+" of data set does not have the expected number of portrait features")
    # add portrait features to output
    for i in range(N_USER_PORTRAITS):
        colIndex = N_ITEMS + i  # index of feature on an element of outputSharedList
        output[colIndex] = int(portraits[i])
    return output

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
        write content: userIDs, UserFeaturesTestSet
            userIDs: array of user ids
            UserFeaturesTestSet: data frame with N_ITEMS+N_USER_PORTRAITS columns
    Data source: /tf/shared/track1_testset.csv
    """
    # read data to pd dataframe
    print('reading raw data file ...')
    rawTestSet = pd.read_csv('/tf/shared/track2_testset.csv',' ')
    # create output frame
    colNames = ['clickedItem'+str(i+1) for i in range(N_ITEMS)] + ['userPortrait'+str(i+1) for i in range(N_USER_PORTRAITS)]
    output = pd.DataFrame(data = np.zeros(shape = (rawTestSet.shape[0], N_ITEMS+N_USER_PORTRAITS)), columns = colNames)
    # parse each line in parallel
    # first objects in shared memory for input and output
    print('creating shared memory objects ... ')
    inputList = rawTestSet.values.tolist()  # for memory efficiency
    p = mp.Pool(N_THREADS)
    print('multiprocessing ... ')
    outputList = p.map(parseUserFeaturesOneLine, inputList)
    # convert outputSharedList back to DataFrame
    print('convert to DataFrame ...')
    output = pd.DataFrame(data = outputList, columns = colNames)
    # normalize the portraits columns
    for i in range(N_USER_PORTRAITS):
        colName = 'userPortrait' + str(i+1)
        scaler = MinMaxScaler()
        output[colName] = scaler.fit_transform(output[colName].values.reshape(-1,1))
    # create userIDs array
    userIDs = rawTestSet['user_id'].tolist()
    # save to pickle file
    f = open('/tf/shared/data/UserFeaturesTestSet.pkl', 'wb')
    pickle.dump((userIDs, output), f)

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
    return: (userIDs, UserFeaturesTestSet)
        userIDs: list of user ids
        UserFeaturesTestSet: DataFrame with N_ITEMS+N_USER_PORTRAITS columns
            first N_ITEMS cols: one hot encoding of clicked items
            last N_USER_PORTRAITS cols: normalized user portraits
    """
    file = open(DATA_PATH + 'UserFeaturesTestSet.pkl', 'rb')
    userIDs, output = pickle.load(file)
    return (userIDs, output)


def getClusterLabels():
    """
    return: (model, labels)
        model : model for testset prediction
        labels: numpy array of labels of clusters from the trainset
    """
    with open('/tf/shared/data/ClusterLabels.pkl', 'rb') as f:
        model, labels = pickle.load(f)
    return (model, labels)

def splitTrainSet(percentageTrain = 0.8):
    """
    return (userFeaturesTrain, recItemsTrain, purchaseLabelTrain, userFeaturesTest, recItemsTest, purchaseLabelTest)
    userFeaturesTrain: DataFrame
    recItemsTrain:numpy array
    purchaseLabelTrain: numpy array
    userFeaturesTest: DataFrame
    purchaseLabelTest: numpy array
    recItemsTest: numpy array
    """
    # read raw data
    userFeatures = getUserFeaturesTrainSet()
    rawTrainSet = pd.read_csv('/tf/shared/trainset.csv',' ')
    purchaseLabels1 = rawTrainSet.labels
    recItems1 = rawTrainSet.exposed_items
    N = len(purchaseLabels1)
    # create permutation index
    permutedIndex = np.random.permutation(N)
    trainIndex = permutedIndex[:int(N*percentageTrain)]
    testIndex = permutedIndex[int(N*percentageTrain):]
    # split user features
    userFeaturesTrain = userFeatures.iloc[trainIndex]
    userFeaturesTest = userFeatures.iloc[testIndex]
    # convert recItems to integer
    recItems = []
    for i, s in enumerate(recItems1):
    # loop thru samples
        recItems.append([int(x) for x in s.split(',')])
    recItems = np.array(recItems)
    # convert purchaseLabels to integer
    purchaseLabels = []
    for i, s in enumerate(purchaseLabels1):
    # loop thru samples
        purchaseLabels.append([int(x) for x in s.split(',')])
    purchaseLabels = np.array(purchaseLabels)
    # split recItems
    recItemsTrain = recItems[trainIndex]
    recItemsTest = recItems[testIndex]
    # split purchaseLabels
    purchaseLabelTrain = purchaseLabels[trainIndex]
    purchaseLabelTest = purchaseLabels[testIndex]
    return (userFeaturesTrain, recItemsTrain, purchaseLabelTrain, userFeaturesTest, recItemsTest, purchaseLabelTest)
    
    
def getItemPrice():
    """return: array of item prices"""
    itemInfo = pd.read_csv('/tf/shared/item_info.csv', ' ')
    itemInfo = itemInfo.sort_values(by = 'item_id')
    itemPrice = itemInfo.price
    return itemPrice


def getExposedItemsTrainSet():
    """return list of exposed items in trainset and whether they are purchased
    (exposedItems, purchaseLabels)
    both are list of list
    """
    rawTrainSet = pd.read_csv('/tf/shared/trainset.csv', ' ')
    exposedItems = rawTrainSet.exposed_items
    purchaseLabels = rawTrainSet.labels
    exposedItems_out = []
    purchaseLabels_out = []
    for i in range(len(exposedItems)):
        items = exposedItems[i]
        labels = purchaseLabels[i]
        items = [int(x) for x in items.split(',')]
        labels = [int(x) for x in labels.split(',')]
        exposedItems_out.append(items)
        purchaseLabels_out.append(labels)
    return (exposedItems_out, purchaseLabels_out)
    



if __name__ == "__main__":
    userFeaturesTrain, recItemsTrain, purchaseLabelTrain, userFeaturesTest, recItemsTest, purchaseLabelTest = splitTrainSet(0.8)
    print(userFeaturesTrain.shape)
    print(purchaseLabelTrain.shape)
    print(userFeaturesTest.shape)
    print(purchaseLabelTest.shape)
    
#     prepareUserFeaturesTrainSet()
#     preparePurchasedItemsTrainSet()
#     prepareUserFeaturesTestSet()
