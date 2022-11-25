import numpy as np
import matplotlib.pyplot as plt
import time

#load data
print("loading data")
from loadData import load_mnist
[traindata, trainlabel] = load_mnist()


#collect data
print("collecting data")
image_size = 28 # width and length
numClasses = len(np.unique(trainlabel))
numSamplesPerClass = 2
image_pixels = image_size * image_size
inputVectorSize = len(traindata[0,:])
td = []
tl = []
tls = np.zeros(numClasses)
i = -1
while i < len(traindata) and np.sum(tls) < numSamplesPerClass*numClasses:
    i += 1
    itemlabel = int(trainlabel[i])
    if tls[itemlabel] >= 10:
        continue
    tls[itemlabel] += 1
    td.append(traindata[i,:])
    tl.append(itemlabel)
inputdata = np.asarray(td)
inputdata = inputdata.astype(float)
#inputdata = inputdata / 255
for i in range(len(inputdata)):
	inputdata[i,:] = inputdata[i,:] / np.linalg.norm(inputdata[i,:])


#train Kmeans , and test it
from AEBPObject import AEBP
from AEBPObject import filterType
from AEBPObject import activationFunctionType
learningrate = 0.2
totalerrors = []
totalepoches = []
numHiddenNodes_range = [200]
times = []
numepoches = []
allerrs = []


#train AutoEncoder with BackPropagation method with different hidden layer size
totalerrors_i = []
times_i = []
numepoches_i = []
for numHiddenNodes in numHiddenNodes_range:
    print("\tnum hidden layer nodes = " + str(numHiddenNodes))
    t1 = time.time()
    ae = AEBP(inputVectorSize, numHiddenNodes, learningrate)
    ae.setActivationFunction(activationFunctionType.sigmoid, -1)
#    ae.setFilters([filterType.Normalize])  #add normalization filte
#    ae.setNormalize(1)
    print("\t\ttraining...")
    [errs, eps, totallearningerrores] = ae.learn(inputdata.copy())
    print("\t\t\tnum epocs = " + str(eps))
    print("\t\t\terr = " + str(totallearningerrores))
    totalerrors_i.append([numHiddenNodes, totallearningerrores])
    times_i.append([numHiddenNodes, int(np.round((time.time() - t1) / 60, 0))])
    allerrs.append([numHiddenNodes, errs])
    numepoches_i.append([numHiddenNodes, eps])
#        plot one sample
    if numHiddenNodes == 200:
        plt.figure()
        plt.imshow(np.reshape(inputdata[1,:],[28,28]),'Greys')
    y = ae.predict(inputdata[1,:])
    plt.figure()
    plt.imshow(np.reshape(y,[28,28]),'Greys')
totalerrors_i = np.asarray(totalerrors_i)
times_i = np.asarray(times_i)
numepoches_i = np.asarray(numepoches_i)
plt.figure()
plt.scatter(totalerrors_i[:,0], totalerrors_i[:,1])
plt.figure()
plt.scatter(numepoches_i[:,0], numepoches_i[:,1])
