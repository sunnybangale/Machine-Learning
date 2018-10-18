

import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import sys


#function to calculate eucledian distance
def calculateEuclideanDistance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1-x2, 2) + math.pow(y1 - y2, 2))


#function to get the sum of squared error
def getSumOfSquaredError(assign, xList, yList, clusterIdList, listCentroidX, listCentroidY):
    sumSquaredError = 0
    #print(assign)
    #print(ClusterIdList)
    for cId in range(len(clusterIdList)):
        for point in range(len(assign)):
            if(assign[point] == clusterIdList[cId]):
                sumSquaredError = sumSquaredError + calculateEuclideanDistance(listCentroidX[cId], listCentroidY[cId], xList[point], yList[point]) * calculateEuclideanDistance(listCentroidX[cId], listCentroidY[cId], xList[point], yList[point])
    return sumSquaredError



#command line arguments as inputs
k = int(sys.argv[1])
inputFile = sys.argv[2]
outputFile = sys.argv[3]

#read data from the input file
dataFrame = pd.read_csv(inputFile, delimiter="\t")

#create lists for id,x and y
idList = []
xList = []
yList = []

#separate dataframe into x, y and id
for i in range(len(dataFrame)):
    idList.append(int(dataFrame.iloc[i][0]))
    xList.append(dataFrame.iloc[i][1])
    yList.append(dataFrame.iloc[i][2])

# plt.scatter(x, y)
# plt.show()

#print the value of k
print("\n\nValue of k is:", k)


minOfX = min(xList)
maxOfX = max(xList)
minOfY = min(yList)
maxOfY = max(yList)

# print("Range of x: [" + str(minX) + ",", str(maxX) + "]")
# print("Range of y: [" + str(minY) + ",", str(maxY) + "]")

#create lists for centroids of x,y and id
ClusterIdList = []
CentroidXList = []
CentroidYList = []
CentroidXOldList = None
CentroidYOldList = None


for i in range(k):
    ClusterIdList.append(i + 1)
    CentroidXList.append(random.uniform(minOfX + 0.2, maxOfX - 0.2))
    CentroidYList.append(random.uniform(minOfY + 0.2, maxOfY - 0.2))

#print the cluster centroids
print("Cluster Centroids:")
for i in range(len(ClusterIdList)):
    print(ClusterIdList[i], CentroidXList[i], CentroidYList[i])
    

#create list for distances
DistanceList = [[] for i in range(len(ClusterIdList))]
assign = [-1 for i in range(len(idList))]

#calculate eucledian distance between every data point
for i in range(len(idList)):
    minimumDistance = 100
    for j in range(len(ClusterIdList)):
        eucledianDistance = calculateEuclideanDistance(xList[i], yList[i], CentroidXList[j], CentroidYList[j])
        # print("Instance:", i,", Cluster:", m+1, "Distance:", distance)
        if(minimumDistance > eucledianDistance):
            minimumDistance = eucledianDistance
            assign[i] = j + 1


# plt.show()
# print("SSE", getSSE(assign, xList, yList, ClusterIdList, CentroidXList, CentroidYList))

for t in range(1,25):
    for num in range(len(ClusterIdList)):
        #print("Updating centroid", k+1)
        sumOfX = 0
        totalOfX = 0.00001
        sumOfY = 0
        totalOfY = 0.00001
        for i in range(len(idList)):
            if(assign[i] == num+1):
                sumOfX = sumOfX + xList[i]
                totalOfX = totalOfX + 1
                sumOfY = sumOfY + yList[i]
                totalOfY = totalOfY + 1
        CentroidXList[num] = sumOfX / totalOfX
        CentroidYList[num] = sumOfY / totalOfY
        
        
#assignment of points to clusters
    for i in range(len(idList)):
        minimumDistance = 100
        for j in range(len(ClusterIdList)):
            eucledianDistance = calculateEuclideanDistance(xList[i], yList[i], CentroidXList[j], CentroidYList[j])
            #print("Instance:", i,", Cluster:", j+1, "Distance:", distance)
            if(minimumDistance > eucledianDistance):
                minimumDistance = eucledianDistance
                assign[i] = j + 1
   
    if(CentroidXList == CentroidXOldList and CentroidYList == CentroidYOldList):
        break
    else:
        CentroidXOldList = CentroidXList
        CentroidYOldList = CentroidYList


#write cluster id and points in every cluster to output file
outputTextFile = open(outputFile, "w")
for j in range(len(ClusterIdList)):
    string = str(j + 1) + '\t\t'
    for i in range(len(idList)):
        if(assign[i] == ClusterIdList[j]):
            string = string + str(idList[i]) + ','
    string = string + '\n\n'
    outputTextFile.write(string)


#calculate the mean squared error and write to output file
sumOfSquaredError = "SSE: " + str(getSumOfSquaredError(assign, xList, yList, ClusterIdList, CentroidXList, CentroidYList))
print("\n\n" + sumOfSquaredError)
outputTextFile.write(sumOfSquaredError)
outputTextFile.close()