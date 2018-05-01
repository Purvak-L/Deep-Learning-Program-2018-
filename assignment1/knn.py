import numpy as np
import math
import operator
import sys
from sklearn.decomposition import PCA

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


K = int(sys.argv[1])
D = int(sys.argv[2])
N = int(sys.argv[3])
fileName = sys.argv[4]
data = unpickle(fileName)



def RGB2Grey(R,G,B):
    return 0.299*R + 0.587*G + 0.114*B

def seperateChannels(imageVec):
    R = imageVec[:1024]
    G = imageVec[1024:2048]
    B = imageVec[2048:]
    return R,G,B

def preprocessData(traindata,testdata):
    preprocess_train=[]
    preprocess_test=[]
    for vec in traindata:
        R,G,B = seperateChannels(vec)
        greyscaleImage = RGB2Grey(R,G,B)
        preprocess_train.append(greyscaleImage)
    for vec in testdata:
        R,G,B = seperateChannels(vec)
        greyscaleImage = RGB2Grey(R,G,B)
        preprocess_test.append(greyscaleImage)
    return preprocess_train,preprocess_test

def getDistance(x,y):
    distance = np.sum(np.square(x-y))
    distance = math.sqrt(distance)
    return 1.0/distance

def findNeighbours(train_data,test_data, train_labels, K):
    distances=[]
    for i in range(len(train_data)):
        dist = getDistance(train_data[i],test_data)
        distances.append((dist,train_labels[i],train_data[i]))
    distances.sort(key=operator.itemgetter(0),reverse=True)
    neighbours = distances[:K]
    return neighbours

def getClass(neighbours):
    votes = {}
    for i in range(len(neighbours)):
        classLabel = neighbours[i][1]
        if classLabel in votes:
            votes[classLabel] += neighbours[i][0]
        else:
            votes[classLabel] = neighbours[i][0]
    sorted_votes = sorted(votes, key=votes.get, reverse=True)
    return sorted_votes[0]


labels = data[b'labels'][:1000]
image_data = data[b'data'][:1000]

testlabels = labels[:N]
testdata = image_data[:N]

trainlabels = labels[N:]
traindata = image_data[N:]

traindata,testdata = preprocessData(traindata,testdata)

pca = PCA(n_components=D,svd_solver='full')
train_data_transformed = pca.fit_transform(traindata)
testdata_transformed = pca.transform(testdata)


result = []
for i in range(len(testdata_transformed)):
    neighbours = findNeighbours(train_data_transformed,testdata_transformed[i],trainlabels,K)
    className = getClass(neighbours)
    #print(str(className)+" "+str(testlabels[i]))
    result.append(str(className)+" "+str(testlabels[i]))

with open("knn_output.txt",'wt') as f:
    f.write("\n".join(result))
