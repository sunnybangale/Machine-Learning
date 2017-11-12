
# coding: utf-8

# In[1]:
#import libraries
from math import log
import operator


# In[2]:
#Get input of the datasets required for the program from the user
trainingPath = input("Enter Training Dataset file path: ")
validationPath = input("Enter Validation Dataset file path: ")
testingPath = input("Enter Testing Dataset file path: ")


#Reading the dataset
def getData():
    import csv
    # opening the training dataset file
    with open(trainingPath, 'r') as myTrainingFile:
        read = csv.reader(myTrainingFile)
        head = next(read, None)  # read the headers and go to next
        data =[list(map(int,record)) for record in csv.reader(myTrainingFile, delimiter=',')]
    label = head[:-1]
    #print(label)
    #print(data)
    return data, label


#Read the training dataset
def getTrainDataSet():
    import csv

    #Open the training dataset file from the given path
    with open(trainingPath, 'r') as myTrainingFile:
        read = csv.reader(myTrainingFile)
        head = next(read, None)  # read the headers of the file and then increment
        data =[list(map(int,record)) for record in csv.reader(myTrainingFile, delimiter=',')]
    label = head[:-1]
    #print(label)
    # print(data)
    return data, label

#Read the test dataset
def getTestDataSet():
    import csv

    #Open the test dataset file from the given path

    with open(testingPath, 'r') as myTrainingFile:
        read = csv.reader(myTrainingFile)
        head = next(read, None)  # read the headers of the file and then increment
        data =[list(map(int,record)) for record in csv.reader(myTrainingFile, delimiter=',')]
    label = head[:-1]
    #print(label)
    # print(data)
    return data, label


#Read the validation dataset
def getValidDataSet():
    import csv

    #Open the validation dataset file from the given path

    with open(validationPath, 'r') as myTrainingFile:
        read = csv.reader(myTrainingFile)
        head = next(read, None)  # read the headers of the file and then increment
        data =[list(map(int,record)) for record in csv.reader(myTrainingFile, delimiter=',')]
    label = head[:-1]
    #print(labels)
    #print(data)
    return data, label


#Calculate entropy from the data

def calcEntropy(data):
    numEntries = len(data)
    countOfLabel = {}
    for feature_vector in data:  # this calculates the number of unique attributes and their occurance
        splittingLabel = feature_vector[-1]
        if splittingLabel not in countOfLabel.keys(): countOfLabel[splittingLabel] = 0
        countOfLabel[splittingLabel] += 1

    #print("count of label: ", countOfLabel)

    entropy = 0.0
    for key_value in countOfLabel:
        prob = float(countOfLabel[key_value]) / numEntries
        entropy -= prob * log(prob, 2)
    return entropy


#We need to split the data set based on the parent attribute
def dataSplitting(data, ref_var, val):
    retrieveData = []
    for feature_vector in data:
        if feature_vector[ref_var] == val:
            reducedFeatVec = feature_vector[:ref_var]  # chop out based on reference variable used for splitting
            reducedFeatVec.extend(feature_vector[ref_var + 1:])
            retrieveData.append(reducedFeatVec)
    return retrieveData


#Choosing the best attribute based on the max info gain
def selectBestFeature(data):
    numberOfFeatures = len(data[0]) - 1  #The last column of the dataset represents labels
    baseEntropy = calcEntropy(data)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numberOfFeatures):  #Increment over all the features
        featList = [example[i] for example in data]  # create a list of all the examples of this feature
        uniqueValues = set(featList)  #Get a set of unique values from feature list
        #print("Feature list ", featList)
        #print("set: ", uniqueValues)

        newEntropy = 0.0
        
        for val in uniqueValues:
            subDataSet = dataSplitting(data, i, val
                                     )
            prob = len(subDataSet) / float(len(data))
            newEntropy += prob * calcEntropy(subDataSet)


        infoGain = baseEntropy - newEntropy  # calculate the infogain
        
        #print("feature : " + str(i))
        #print("baseEntropy : "+str(baseEntropy))
        #print("AvgEntropy : " + str(newEntropy))
        #print("infoGain : " + str(infoGain))
        
        if (infoGain > bestInfoGain):  # compare this to the best infogaingain
            bestInfoGain = infoGain
            bestFeature = i

    #print("Best Feature: ", bestFeature)
    return bestFeature



#Get the majority class

def majorityClassCount(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]



#Creating a decision tree

def createDecisionTree(data, label):

    classList = [example[-1] for example in data]

    #print(classList)
    #print("Data ", data[0])
    
    if classList.count(classList[0]) == len(classList):
        return classList[0]     #Stop splitting when all classes are equal
    if len(data[0]) == 1:       #Stop splitting when no more features exist in dataSet
        return majorityClassCount(classList)
    
    #Use Information Gain
    bestFeat = selectBestFeature(data)
    bestFeatLabel = label[bestFeat]

    #Use recursion to build the best tree

    myTree = {bestFeatLabel: {}}

    #print("myTree : "+labels[bestFeat])
    del (label[bestFeat])
    
    featValues = [example[bestFeat] for example in data]
    #print("featValues: "+str(featValues))
    
    uniqueValues = set(featValues)
    #print("uniqueValues: " + str(uniqueValues))
    
    for val in uniqueValues:
        subLabels = label[:]    #Copy all the labels

        #print("subLabels"+str(subLabels))

        myTree[bestFeatLabel][val] = createDecisionTree(dataSplitting(data, bestFeat, val), subLabels)

        #print("myTree : " + str(myTree))

    return myTree


#Classify

def classify(ipTree, featureLabels, instanceVector):

    # Find the attribute on which split has been performed
    for key in ipTree:
        innerpart = ipTree[key]
        

    index = featureLabels.index(key)
    val = instanceVector[index]
    
    
    innervalue = innerpart[val]
    
    ## Find the value for inner node and values
    
    # If we get a leaf node, apply recusive fuction till we reach the root node.
    if isinstance(innervalue, dict):
        
        classLabel = classify(innervalue, featureLabels, instanceVector)
    else:
    # If it is a class label, we have reached the root node and hence ready for prediction
    # The class label is the prediction, return
        classLabel = innervalue
    
    return classLabel


#Stores the tree in a file
def storeTree(ipTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(ipTree, fw)
    fw.close()


#Retrieves a tree from the stored file
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)



#Get the accuracy on the test data
def calculateTestingAccuracy():
    testingAccuracy = 0

    #load testing data
    
    import csv
    with open(testingPath, 'r') as myTestingFile:
        read = csv.reader(myTestingFile)
        head = next(read, None)  # read the headers and go to next
        data =[list(map(int,record)) for record in csv.reader(myTestingFile, delimiter=',')]
    label = head[:-1]
    #print(data)
    #print(label)
    

    totInstances = len(data)
    numberOfCorrectPredictions = 0
    
    for instance in data:
        predictedClassLabel = classify(mytree, label, instance[:-1])
        if(predictedClassLabel == instance[-1]):
            numberOfCorrectPredictions += 1
            
    accuracy = numberOfCorrectPredictions/totInstances
    return accuracy



#Get accuracy on Training data

def calculateTrainingAccuracy():
    trainingAccuracy = 0

    #load testing data
    
    import csv
    with open(trainingPath, 'r') as myTrainingFile:
        read = csv.reader(myTrainingFile)
        head = next(read, None)  # read the headers and go to next
        data =[list(map(int,record)) for record in csv.reader(myTrainingFile, delimiter=',')]
    label = head[:-1]

    #print(data)
    #print(label)
    
    
    totInstances = len(data)
    numberOfCorrectPredictions = 0
    
    for instance in data:
        predictedClassLabel = classify(mytree, label, instance[:-1])
         #print("Actual: ", instance[-1], "Predicted: ", predictedClassLabel)

        if(predictedClassLabel == instance[-1]):
            numberOfCorrectPredictions += 1
            
    accuracy = numberOfCorrectPredictions/totInstances
    return accuracy



#Get accuracy on validation data

def calculateValidationAccuracy():
    ValidationAccuracy = 0

    #load testing data
    
    import csv
    
    with open(validationPath, 'r') as myValidationFile:
        read = csv.reader(myValidationFile)
        head = next(read, None)  # read the headers and go to next
        data =[list(map(int,record)) for record in csv.reader(myValidationFile, delimiter=',')]
    label = head[:-1]

    #print(data)
    #print(label)
    
    
    
    
    
    totInstances = len(data)
    numberOfCorrectPredictions = 0
    
    for instance in data:
        predictedClassLabel = classify(mytree, label, instance[:-1])
        if(predictedClassLabel == instance[-1]):
            numberOfCorrectPredictions += 1
            
    accuracy = numberOfCorrectPredictions/totInstances
    return accuracy


#Print the decision tree

def printTree(tree, d=0):
    for attribute in tree:
        options = tree[attribute]
        for val in options:
            print(" |" * d,attribute, "=", val, ":")
            subtree = options[val]
            if(isinstance(subtree, dict)):
                printTree(subtree, d+1)
            else:
                print(" |" * d,attribute, "=", val, ":", subtree)

#printTree(mytree)



#Collect data from the files
myDat, label = getData()

#Build the decision tree
mytree = createDecisionTree(myDat, label)

#print(mytree);

#print the decision tree
printTree(mytree)




#Get accuracy on data
def calculateAccuracy(data, label):
    totInstances = len(data)
    numberOfCorrectPredictions = 0
    
    for instance in data:
        predictedClassLabel = classify(mytree, label, instance[:-1])
        if(predictedClassLabel == instance[-1]):
            numberOfCorrectPredictions += 1
            
    accuracy = numberOfCorrectPredictions/totInstances
    return accuracy


#Get total nodes
def getTotalNodes(tree):
    str_tree = str(tree)
    return str_tree.count("{") + 1


#Get total number of leaves
def getNumLeaves(tree):
    N = getTotalNodes(tree)
    return (N+1)/2


#We are yet to implement pruning of the tree
def getPrunedTree(tree):

    return tree




#############################################################################

training_dataset, training_headers = getTrainDataSet()
training_accuracy = calculateAccuracy(training_dataset, training_headers)

testing_dataset, testing_headers = getTestDataSet()
testing_accuracy = calculateAccuracy(testing_dataset, testing_headers)

validation_dataset, validation_headers = getValidDataSet()
validation_accuracy = calculateAccuracy(validation_dataset, validation_headers)

print(" ")
print("Pre Pruned Accuracy")
print('-'*20)
print("Number of training instances = ", len(training_dataset))
print("Number of training attributes = ", len(training_headers))

print("Total number of nodes in the tree = ", getTotalNodes(mytree))
print("Total number of leaf nodes in the tree = ", getNumLeaves(mytree))
print("Accuracy of the model on the training dataset = ", training_accuracy*100)

print("\nNumber of validation instances = ", len(validation_dataset))
print("Number of validation attributes = ", len(validation_headers))
print("Accuracy of the model on the validation dataset before pruning = ", validation_accuracy*100)

print("\nNumber of testing instances = ", len(testing_dataset))
print("Number of testing attributes = ", len(testing_headers))
print("Accuracy of the model on the testing dataset = ", testing_accuracy*100)

###################
##### We are yet to implement post pruning activity
###################

#print(getPrunedTree(mytree))

print(" ")
print("Post Pruned Accuracy")
print('-'*20)
print("Number of training instances = ", len(training_dataset))
print("Number of training attributes = ", len(training_headers))

print("Total number of nodes in the tree = ", getTotalNodes(mytree))
print("Total number of leaf nodes in the tree = ", getNumLeaves(mytree))
print("Accuracy of the model on the training dataset = ", training_accuracy*100)

print("\nNumber of validation instances = ", len(validation_dataset))
print("Number of validation attributes = ", len(validation_headers))
print("Accuracy of the model on the validation dataset before pruning = ", validation_accuracy*100)

print("\nNumber of testing instances = ", len(testing_dataset))
print("Number of testing attributes = ", len(testing_headers))
print("Accuracy of the model on the testing dataset = ", testing_accuracy*100)


