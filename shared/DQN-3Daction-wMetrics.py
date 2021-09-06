#!/usr/bin/env python
# coding: utf-8

# In[1]:


####################################################################
# Implementation 2 of DQN by using sets of 3 items.
#     State: (Click History, User Portraits) (note: purchase timestamp is not available in testset)
#     Action: sets of 3 itemID (each sample is split into 3 steps)
#     Rewards: Total price of purchased items
# 0. split train data into training set and validation set
# 1. prepare data for DQN from training set
# 2. prepare data for DQN from validation set
# 3. train DQN
# 4. make suggestions for validation set
# 5. Calculate Metrics 1 for our suggestions
# 6. Generate suggestions for provided testset for true scoring
#####################################################################


# In[1]:


# 0. Split Train into training set and validation set
from DataPrep import *
from tqdm import tqdm
userFeaturesTrain, recItemsTrain, purchaseLabelTrain, userFeaturesVal, recItemsVal, purchaseLabelVal = splitTrainSet()
# when training, userFeaturesTrain represent state
N_ITEMS = 381
# load item info
from classes.Items import Items
itemInfo = Items()
# translator from (ID1, ID2, ID3) to setID
from classes.ItemSet import ItemSet3
itemSet3 = ItemSet3()


# In[2]:


# 1. prepare data for training set
import numpy as np
from d3rlpy.dataset import MDPDataset

statesTrain = []
actionsTrain = []
rewardsTrain = []
terminalTrain = []  # terminal flag: 0 = game continue, 1 = game stop

for i in tqdm(range(userFeaturesTrain.shape[0])):
# loop through samples
    state = list(userFeaturesTrain.iloc[i])
    itemList = recItemsTrain[i]
    purchase = purchaseLabelTrain[i]
    for step in range(3):
        # check if game is still running
        if step>0 and purchase[0]*purchase[1]*purchase[2]==0:
            # stop adding to data set if game stopped
            break
        if step>1 and purchase[3]*purchase[4]*purchase[5]==0:
            # stop adding to data set if game stopped
            break
        # after passing check, we can add new record to train set
        # append step to state
        if step==0:
            step_OneHot = [1, 0, 0]
        elif step==1:
            step_OneHot = [0, 1, 0]
        else:
            step_OneHot = [0, 0, 1]
        statesTrain.append(state + step_OneHot)
        # action = itemSetID
        itemIDs = (itemList[step*3], itemList[step*3+1], itemList[step*3+2])
        itemSetID = itemSet3.getSetID(itemIDs)
        actionsTrain.append(itemSetID)
        # calculate reward
        price0 = itemInfo.getItemPrice(itemIDs[0])
        price1 = itemInfo.getItemPrice(itemIDs[1])
        price2 = itemInfo.getItemPrice(itemIDs[2])
        purch0 = purchase[step*3]
        purch1 = purchase[step*3+1]
        purch2 = purchase[step*3+2]
        reward = price0*purch0 + price1*purch1 + price2*purch2
        rewardsTrain.append(reward)
        # terminal flag: determine by looking at previous purchase flags
        if step!=2:
            if purch0*purch1*purch2 == 1: # game continue if all is 1
                terminalTrain.append(0)
            else:
                terminalTrain.append(1) # game stop
        else: # game stop at step 2
            terminalTrain.append(1)

# ### terminal flags: all 1
statesTrain = np.array(statesTrain)
actionsTrain = np.array(actionsTrain)
rewardsTrain = np.array(rewardsTrain)
terminalTrain = np.array(terminalTrain)
datasetTrain = MDPDataset(statesTrain, actionsTrain, rewardsTrain, terminalTrain, discrete_action = True)


# In[3]:


# 1. prepare data for validation set
statesVal = []
actionsVal = []
rewardsVal = []
terminalVal = []  # terminal flag: 0 = game continue, 1 = game stop

for i in tqdm(range(userFeaturesTrain.shape[0])):
# loop through samples
    state = list(userFeaturesTrain.iloc[i])
    itemList = recItemsTrain[i]
    purchase = purchaseLabelTrain[i]
    for step in range(3):
        # check if game is still running
        if step>0 and purchase[0]*purchase[1]*purchase[2]==0:
            # stop adding to data set if game stopped
            break
        if step>1 and purchase[3]*purchase[4]*purchase[5]==0:
            # stop adding to data set if game stopped
            break
        # after passing check, we can add new record to train set
        # append step to state
        if step==0:
            step_OneHot = [1, 0, 0]
        elif step==1:
            step_OneHot = [0, 1, 0]
        else:
            step_OneHot = [0, 0, 1]
        statesVal.append(state + step_OneHot)
        # action = itemSetID
        itemIDs = (itemList[step*3], itemList[step*3+1], itemList[step*3+2])
        itemSetID = itemSet3.getSetID(itemIDs)
        actionsVal.append(itemSetID)
        # calculate reward
        price0 = itemInfo.getItemPrice(itemIDs[0])
        price1 = itemInfo.getItemPrice(itemIDs[1])
        price2 = itemInfo.getItemPrice(itemIDs[2])
        purch0 = purchase[step*3]
        purch1 = purchase[step*3+1]
        purch2 = purchase[step*3+2]
        reward = price0*purch0 + price1*purch1 + price2*purch2
        rewardsVal.append(reward)
        # terminal flag: determine by looking at previous purchase flags
        if step!=2:
            if purch0*purch1*purch2 == 1: # game continue if all is 1
                terminalVal.append(0)
            else:
                terminalVal.append(1) # game stop
        else: # game stop at step 2
            terminalVal.append(1)

# ### terminal flags: all 1
statesVal = np.array(statesVal)
actionsVal = np.array(actionsVal)
rewardsVal = np.array(rewardsVal)
terminalVal = np.array(terminalVal)
datasetVal = MDPDataset(statesVal, actionsVal, rewardsVal, terminalVal, discrete_action = True)


# In[4]:

# train deep learning models 
from classes import d3rlpy_wrapper
from importlib import reload
d3rlpy_wrapper = reload(d3rlpy_wrapper)

wrapper = d3rlpy_wrapper.RLModelWrapper(datasetTrain, datasetVal)
# wrapper.trainAllModels(n_epochs = 1)


# In[4]: save checkpoints
# save checkpoint
# import pickle
# with open('/tf/shared/checkpoints/models-3D.pkl', 'wb') as file:
#     pickle.dump((wrapper.DQN, wrapper.DoubleDQN, wrapper.DiscreteBCQ, wrapper.DiscreteCQL),
#                 file, protocol=pickle.HIGHEST_PROTOCOL)
# with open('/tf/shared/checkpoints/data-3D.pkl', 'wb') as file:
#     pickle.dump(statesVal,
#                 file, protocol=pickle.HIGHEST_PROTOCOL)

# reload checkpoint
# reload checkpoint
import pickle
from classes import d3rlpy_wrapper
wrapper = d3rlpy_wrapper.RLModelWrapper(datasetTrain, datasetVal)
with open('/tf/shared/checkpoints/models-3D.pkl', 'rb') as file:
    wrapper.DQN, wrapper.DoubleDQN, wrapper.DiscreteBCQ, wrapper.DiscreteCQL = pickle.load(file)
with open('/tf/shared/checkpoints/data-3D.pkl', 'rb') as file:
    statesVal = pickle.load(file)

# In[5]:


# make predictions
from classes import ModelPrediction, ItemSet
from importlib import reload  
ModelPrediction = reload(ModelPrediction)
ItemSet = reload(ItemSet)

# modelDQN = ModelPrediction.Model3D(wrapper.DQN)
# modelDoubleDQN = ModelPrediction.Model3D(wrapper.DoubleDQN)
modelDiscreteBCQ = ModelPrediction.Model3D(wrapper.DiscreteBCQ)
# modelDiscreteCQL = ModelPrediction.Model3D(wrapper.DiscreteCQL)


# In[ ]:


pred_DiscreteBCQ = modelDiscreteBCQ.predict(statesVal[:9000])


# In[ ]:


print(pred_DiscreteBCQ)


# 
