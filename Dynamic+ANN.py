
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
import math
from math import exp
from operator import mul


# In[2]:
#remove nulls from dataset
def removeNulls(dataFrame):
    updated_dataFrame = dataFrame
    for i in range(len(dataFrame)):
        for j in range(len(dataFrame.iloc[i, :])):
            if(dataFrame.iloc[i,j] == '?'):
                updated_dataFrame = updated_dataFrame.drop(dataFrame.index[i])
                break

    return updated_dataFrame


# In[3]:
#normalize the dataset
def meanNormalize(array):
    mean = np.average(array)
    std_dev = np.std(array)
    normalizedArray = (array - mean)/std_dev
    return normalizedArray


# In[4]:
#enter input path of the dataset
url = input("Enter the URL of the dataset:")
output_path = input("Enter the output path for processed data:")
dataFrame = pd.read_csv(url, skipinitialspace=True, na_values='.', header = None)


# In[5]:

dataFrame_without_nulls = removeNulls(dataFrame)


# In[6]:

normalised_dataFrame = dataFrame_without_nulls
for i in range(len(dataFrame_without_nulls.columns)):
    data_type = dataFrame_without_nulls.dtypes[i]
    #print("TYPE:", data_type)
    if(data_type == np.int64 or data_type == np.float64):
        #print(data_type, "We'll need to normalize", i)
        #print(dataFrame_without_nulls[i])
        normalised_dataFrame[i] = meanNormalize(dataFrame_without_nulls[i])
        #print()
    #print()


# In[7]:

preprocessed_dataFrame = normalised_dataFrame
for j in range(len(preprocessed_dataFrame.columns)):
    data_type = preprocessed_dataFrame.dtypes[j]
    if(data_type == np.object):
        #print("Column:", j) 
        list_unique = preprocessed_dataFrame[j].unique().tolist()
        
        #print(list_unique)
        for i in range(len(preprocessed_dataFrame)):
            value = preprocessed_dataFrame.iloc[i, j]
            #print(value)
            label = list_unique.index(value)
            preprocessed_dataFrame.iloc[i, j] = label
        
        #print()


# In[8]:

j = len(preprocessed_dataFrame.columns)-1
preprocessed_dataFrame[j]
list_unique = preprocessed_dataFrame[j].unique().tolist()

a = list(list_unique)
n = len(a)
interval = 1/(n-1)
b = a
value = 0
for i in range(len(a)):
    b[i] = value
    value = value + interval

b = np.array(b)


for i in range(len(preprocessed_dataFrame)):
    value = preprocessed_dataFrame.iloc[i, j]
    #print(value)
    label = b[list_unique.index(value)]
    preprocessed_dataFrame.iloc[i, j] = label



# In[9]:

normalised_dataFrame.to_csv(output_path, sep=',',index= None, header=None)


#sigmoid funtion
def sigmoidActivationFunction(netValue):
    if(netValue > 20):
        netValue = 20
    elif(netValue<-20):
        netValue = -20
        
    return 1.0 / (1.0 + exp(-netValue))

#train neural network
def trainNeuralNetwork(maximumIterations, dataFrame, allWeights, numberOfNodes, learningRate):
    #for n in range(100):
    for n in range(maximumIterations):    
        #print("\n\nIteration :", n)
        #for i in range(5):
        for i in range(len(dataFrame)):
            #print("Example :", i)
            inputArray = dataFrame.iloc[i,:-1]
            inputArray = list(inputArray)
            inputArray.insert(0, 1)
            inputArray = np.array(inputArray)
        
            targetOutput = dataFrame.iloc[i,-1]
            #print(inputArray)
            #print("targetOutput",targetOutput)

            sigmoidList, output = forwardPropagation(allWeights, inputArray, numberOfNodes)
            #print("Output", output)
            #print()
            allWeights = backPropagationFunction(allWeights, inputArray, sigmoidList, numberOfNodes, learningRate, targetOutput)
            
    return allWeights


# In[12]:

def calculateWeights(inputNodeCount, hiddenLayersCount, hiddenNodesCount):
    
    #print("Total Nodes in Input Layer:", inputNodeCount)
    #print("Total Hidden Layers:", hiddenLayersCount)

    #for i in range(hiddenLayersCount):
        #print("Weights in Hidden Layer", i, ":", hiddenNodesCount[i])

    hiddenNodesCount.insert(0, inputNodeCount)
    numberOfNodes = hiddenNodesCount
    #print(numberOfNodes)

    #print("Weight Matrices Required:", count_weight_matrices)
    #print()

    allWeights = list()
    
    for i in range(hiddenLayersCount):
        j = i + 1
        #print("\nBetween layer", i, "and layer",j)
        #print("Dimension of Wt Mat.:", numberOfNodes[j]-1, "X", numberOfNodes[i])
        W = np.matrix(np.random.uniform(-1,1, size=(numberOfNodes[j]-1, numberOfNodes[i])))
        #print(W)
        allWeights.append(W.tolist())

    i = i + 1
    j = i + 1
    #print("\nBetween layer", i, "and layer", j)
    #print("Dimension of Wt Mat.:", outputNodesCount, "X", numberOfNodes[i])
    W = np.matrix(np.random.uniform(-1,1, size=(outputNodesCount, numberOfNodes[i])))
    #print(W)
    allWeights.append(W.tolist())
    #print("\n\nWeights Matrix")
    #print(allWeights)
    return allWeights, numberOfNodes


# In[13]:

def findAccuracyOfNetwork(dataFrame, allWeights, numberOfNodes):
    
    j = len(dataFrame.columns)-1
    dataFrame[j]
    list_unique = dataFrame[j].unique().tolist()
    list_unique.sort()
    #print(list_unique)
    num_corrects = 0
    for i in range(len(dataFrame)):
        #print("Example :", i)
        inputArray = dataFrame.iloc[i,:-1]
        inputArray = list(inputArray)
        inputArray.insert(0, 1)
        inputArray = np.array(inputArray)
        
        targetOutput = dataFrame.iloc[i,-1]
        #print(inputArray)
        #print("Target output",targetOutput)

        sigmoidList, output = forwardPropagation(allWeights, inputArray, numberOfNodes)
        #print("Output", output[0])
        
        label  = findLabel(list_unique, output[0])
        #print(label[0], "\n")
        if(label == targetOutput):
           num_corrects = num_corrects+1
    #print(num_corrects)
    accuracy = num_corrects/len(dataFrame)
    return accuracy


# In[14]:

#forward propagation function

def forwardPropagation(allWeights, inputArray, numberOfNodes):
    
    #print(inputArray)

    sigmoidList = list()
    
    #print(len(numberOfNodes))
    for i in range(len(numberOfNodes)):
        temp = 0
        sigmoid_at_level = list()
        sigmoid_at_level.append(1)
        #print("\n\nAt layer", i)
        if(i==0):
            vector = inputArray
        else:
            vector
            
        temp = np.array(vector)*allWeights[i]
        #print(temp)
        
        netValue = list()
        for j in range(len(temp)):
            netValue.append(sum(temp[j]))
        #print(netValue)
        
        
        for j in range(len(netValue)):
                sigmoid_at_level.append(sigmoidActivationFunction(netValue[j]))
            
        #print("Sigmoid at level", sigmoid_at_level)
        sigmoidList.append(sigmoid_at_level)
        #print(sigmoidList)
        vector = sigmoid_at_level
    
    #print the output
    output = sigmoid_at_level[1:]
    #print("\n\nOutput:", output)
    return sigmoidList, output


# In[15]:

def findLabel(array,value):
    index = np.searchsorted(array, value, side="left")
    if index > 0 and (index == len(array) or math.fabs(value - array[index-1]) < math.fabs(value - array[index])):
        return array[index-1]
    else:
        return array[index]


# In[16]:

#back prop

def backPropagationFunction(allWeights, inputArray, sigmoidList, numberOfNodes, learningRate, targetOutput):
    #print("Back Prop")
    output = sigmoidList[-1][1]
    #print("Output:", output)
    #print("Target:", targetOutput)
    x = len(numberOfNodes)
    
    list_delta = list()
    
    for i in range(len(numberOfNodes), 0, -1):
        #print("\n\nAt Layer:", i)
        delta_at_level = list()
        delta_weights_level = 0
        if(i==len(numberOfNodes)):
            temp = (targetOutput - output)*output*(1-output)
            delta_at_level.insert(0, temp)
            #print(delta_at_level)
            list_delta.insert(0, delta_at_level)
            #print(list_delta)
        else:
            #print("List Delta", list_delta)
            #print("\n\n")
            #print("List Sigmoid", sigmoidList[i-1])
            #print("\nList Delta:", list_delta[x-i-1])
            #print("\nList Weights:", allWeights[i])
            
            delta_at_level = np.array(sigmoidList[i-1]) * (1 - np.array(sigmoidList[i-1])) * np.array(list_delta[x-i-1]) * np.array(allWeights[i])
            #print("Delta at level:", delta_at_level)
            
            #print("\n\n\n")
            delta_weights_level = learningRate * list_delta[x-i-1][0] * np.array(sigmoidList[i-1])
            #print(delta_weights_level)
            #print(allWeights[i][0])
            list_delta.insert(0, delta_at_level.tolist())
        
            #allWeights[1][0] 
            mytemp = allWeights[i][0] + delta_weights_level
            #print(list(mytemp))
            allWeights[i][0] = list(mytemp)
            #print(allWeights)
            
           
    #print("\n\nNow at input layer")
    #print(list_delta[0][0][1:])
    #print(inputArray)
    
    delta_weights_level = learningRate * np.matrix(list_delta[0][0][1:]).T * np.matrix(inputArray)
    delta_weights_level = delta_weights_level.tolist()
    #print(delta_weights_level)
    
    a = np.matrix(delta_weights_level)
    b = np.matrix(allWeights[0])
    c = np.add(a,b)
    temp = c.tolist()
    allWeights[0] = temp
    #print("Final Weights:", allWeights)
    
    return allWeights


# In[17]:

#input from user

path_input_dataset = input("Enter the input Dataset to form the Neural Network : ")
dataFrame = pd.read_csv(path_input_dataset, header=None)
dataFrame = dataFrame.sample(frac=1)
training_percent = int(input("Enter the percent of training data to be used : "))

maximumIterations = int(input("Enter the Maximum iterations :"))
learningRate = 0.9
count_features = len(dataFrame.columns)-1+1 # -1 for the class label, +1 for bias

'''hiddenLayersCount = 2
hiddenNodesCount = [2 + 1, 1+1]'''

hiddenLayersCount = int(input("Enter the total number of hidden layers:"))

hiddenNodesCount = list()
for i in range(0, hiddenLayersCount):
    value = int(input("Enter the nodes in each of the hidden layers : "))
    hiddenNodesCount.append(value+1)
    
 

inputNodeCount = count_features
count_weight_matrices = hiddenLayersCount + 1
outputNodesCount = 1

num_examples = len(dataFrame)
index = int(training_percent * num_examples / 100)
training_dataset = dataFrame[0:index]
testing_dataset = dataFrame[index:]

allWeights, numberOfNodes = calculateWeights(inputNodeCount, hiddenLayersCount, hiddenNodesCount)
#print("Weight Matrix", allWeights)
#print("Count of Nodes in Layers", numberOfNodes)


# In[18]:

allWeights = trainNeuralNetwork(maximumIterations, training_dataset, allWeights, numberOfNodes, learningRate)
        
#print("Training Done")
#print(allWeights)


# In[19]:

training_accuracy = findAccuracyOfNetwork(training_dataset, allWeights, numberOfNodes)
#print("Traing Accuracy:", training_accuracy*100)


# In[20]:

testing_accuracy = findAccuracyOfNetwork(testing_dataset, allWeights, numberOfNodes)
#print("Testing Accuracy:", testing_accuracy*100)


# In[21]:

training_error = 1 - training_accuracy
testing_error = 1 - testing_accuracy


# In[ ]:




# In[22]:

num_layers = len(allWeights)


# In[23]:

for i in range(num_layers):
    print("\n\nLayer ", i)
    allWeights_layer = allWeights[i]
    #print(len(allWeights_layer))
    #print(allWeights_layer)
    
    for k in range(len(allWeights_layer[0])):
        #print(j)
        print("\n\tNeuron", k, "weights:\n")
        for j in range(len(allWeights_layer)):
            print("\t\t",allWeights_layer[j][k])


# In[24]:

print("Total Training Error:", training_error*100)
print("Total Testing Error:", testing_error*100)


# In[ ]:



