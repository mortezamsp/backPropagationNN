import numpy as np

from enum import Enum
class activationFunctionType(Enum):
    bipolarStepFunction = 0
    sigmoid = 1

class filterType(Enum):
    Normalize = 0
    addNoise = 1
    loosLessImResize = 2
    loosyImResize = 3
    Threshold = 4
    convolution = 5

import cv2 as cv
    
class AEBP:
    def __init__(self, inputsize, numHiddenNodes, learningrate):
        self.learningRate = learningrate
        self.filters = []
        self.numHiddenNodes = numHiddenNodes
        self.inputsize = inputsize
    #for step-function, params must be array of 1 element : [teta]
    def setActivationFunction(self, functiontype, params):#, valueType):
        self.activationFunction = functiontype
        self.activationFunctionParams = params
        #self.activationFunctionValueType = valueType
    def f_activationFunction(self, inputdata):
        outputdata = np.zeros(len(inputdata))
        for i in range(len(inputdata)):
            if self.activationFunction == activationFunctionType.bipolarStepFunction:
                if inputdata[i] <= -self.activationFunctionParams[0]:
                    outputdata[i] = -1
                else:
                    if inputdata[i] >= self.activationFunctionParams[0]:
                        outputdata[i] = 1
                    else:
                        outputdata[i] = 0
                continue
            if self.activationFunction == activationFunctionType.sigmoid:
                outputdata[i] = 1 / (1 + np.power(np.e, -inputdata[i]))
                continue
        return outputdata
    def fprim_activationFunction(self, inputdata):
        outputdata = np.zeros(len(inputdata))
        for i in range(len(inputdata)):
            if self.activationFunction == activationFunctionType.bipolarStepFunction:
                outputdata[i] = 0
                continue
            if self.activationFunction == activationFunctionType.sigmoid:
                si = 1 / (1 + np.power(np.e, -inputdata[i]))
                outputdata[i] = si * (1 - si)
                continue
        return outputdata
    def setFilters(self, filtersType):
        for f in filtersType:
            self.filters.append(f)
            if f == filterType.addNoise:
                self.noiseProbabilty = 0.02
                self.noiseValue = -1
            if f == filterType.convolution:
                self.convolutionWinSize = 3
            if f == filterType.loosLessImResize or f == filterType.loosyImResize:
                self.newsize = 28
            if f == filterType.Normalize:
                self.NormalizeMaxValue = 1
            if f == filterType.Threshold:
                self.binarizationLowerThreshold = 63
                self.binarizationUpperThreshold = 255
    def learn(self, inputdata, maxnumepoches = 10000, minchange=1):
        #phase 1 : train net
        x_ = np.array(inputdata)
        numsamples = len(x_)
#        for ii in range(0, numsamples):
#            x_[ii][:] = self.applyFilters(x_[ii][:])
        errors = []
        self.w1 = np.random.random([self.inputsize, self.numHiddenNodes]) - 0.5
        self.w2 = np.random.random([self.numHiddenNodes, self.inputsize]) - 0.5
        numepoches = 0
        for i in range(0, maxnumepoches):
            errs = 0
            for ii in range(0, numsamples):
                #forward phase
                x =  x_[ii][:]
                zin = (np.transpose(self.w1)).dot(x)
                z = self.f_activationFunction(zin)
                yin = (np.transpose(self.w2)).dot(z)
                y = self.f_activationFunction(yin)
                #backward phase
                delta2 = (y - x)*(y * (1 - y))
                deltaw2 = np.reshape(self.learningRate * z,[self.numHiddenNodes ,1]).dot(np.reshape(delta2,[1, self.inputsize]))
                delta1 = self.w2.dot(np.reshape(delta2,[self.inputsize, 1]))*np.reshape(z*(1-z), [self.numHiddenNodes,1])
                deltaw1 = (self.learningRate * np.reshape(x, [self.inputsize, 1])).dot(np.transpose(delta1))
                #update
                self.w2 -= deltaw2
                self.w1 -= deltaw1
                #stop condition
                errs += np.abs(0.5 * np.sum(np.power(y-x ,2)))
            errors.append(errs)
            numepoches = i
            if i > 0 and i % 50 == 0:
                print(i)
                print(errs)
            if errs <= minchange:
                break
#            self.learningRate *= 0.999
        #phase 2: test on input data

        self.setNoise(0.5, 1)
        err = 0
        for i in range(0, numsamples):
            x =  x_[i][:]
#            x = self.applyFilters(x)
#            x = self.addNoise(x)
            y = self.predict(x)
            err += (0.5 * np.sum(np.power(y - x, 2)))
        return [errors, numepoches, err]
    def predict(self, inputsample):
        x = self.applyFilters(inputsample)
        x = np.reshape(np.asarray(x),[self.inputsize ,1])
        zin = (np.transpose(self.w1)).dot(x)
        z = self.f_activationFunction(zin)
        yin = (np.transpose(self.w2)).dot(z)
        y = self.f_activationFunction(yin)
        return y
    
    def applyFilters(self, x):
        for f in self.filters:
            if f == filterType.addNoise:
                x = self.addNoise(x)
                continue
            if f == filterType.convolution:
                x = self.convolution(x)
                continue
            if f == filterType.loosLessImResize:
                x = self.loosLessImResize(x)
                continue
            if f == filterType.loosyImResize:
                x = self.loosyImResize(x)
                continue
            if f == filterType.Normalize:
                x = self.normalizeIm(x)
                continue
            if f == filterType.Threshold:
                x = self.ImBinaryFilter(x)
                continue
        return x
    def setBinarytFilter(self, lowthresh, upthresh):
        self.binarizationUpperThreshold = upthresh
        self.binarizationLowerThreshold = lowthresh
    def ImBinaryFilter(self, x):
        x[x<self.binarizationLowerThreshold] = 0
        #x[x>=self.binarizationLowerThreshold] = 1
        x[x>self.binarizationUpperThreshold] = 0
        return x
    #noise value = -1: random
    def setNoise(self, noiseProbabilty, noiseValue=-1):
        self.noiseProbabilty = noiseProbabilty
        self.noiseValue = noiseValue
    def addNoise(self, x):
        numnoises = int(len(x) * self.noiseProbabilty)
        for i in range(0, numnoises):
            if self.noiseValue == -1:
                x[int(np.random.random() * len(x))] = int(np.random.random() * 255)
            else:
                x[int(np.random.random() * len(x))] = self.noiseValue
        return x
    def setImResize(self, newSize):
        self.newSize = newSize
    def loosyImResize(self, x):
        insize = len(x)
        bsize = int(insize / self.newSize)
        newx = np.zeros(self.newSize)
        i = 0
        while i+ bsize < len(newx):
            newx[i] = np.average(x[i*bsize:(i+1)*bsize])
            i += 1
        #return np.reshape(newx, [newsize, 1])
        return newx
#        oldsize = np.shape(self.w)[0] - 1
#        oldsize = int(np.sqrt(len(x)))
#        x = np.reshape(x, [oldsize, oldsize])
##        return np.array(Image.fromarray(x).resize())
#        return scipy.misc.imresize(x, [oldsize, oldsize])
    def loosLessImResize(self, x):
        lx = len(x)
        oldsize = int(np.sqrt(lx))
        x = np.reshape(x, [oldsize, oldsize])
#        return cv.resize(src=x, dim=(newsize, newsize), interpolation = cv.INTER_AREA)
        newx = cv.resize(x, dsize=(self.newsize, self.newsize))
        #return np.reshape(newx, [newsize*newsize,1])
        return np.reshape(newx, [self.newsize*self.newsize])
    def setNormalize(self, maxValue):
        self.NormalizeMaxValue = maxValue
    def normalizeIm(self, x):
#        x = (x - np.mean(x)) / np.sqrt(np.var(x))
#        return x / self.NormalizeMaxValue
#        su = np.sum(np.power(np.e, np.asarray(x)))
#        return np.power(np.e, np.asarray(x)) / su
#        return (np.asarray(x) + 0.0) / np.max(x)
        return x / np.linalg.norm(x)
    def setConvlution(self, winsize):
        self.convolutionWinSize = winsize
    def convolution(self, x):
        lx = len(x)
        oldsize = int(np.sqrt(lx))
        A = np.reshape(x, [oldsize, oldsize])
        w = self.convolutionWinSize
        B = (1.0/(w*w))*np.ones((w,w)) # Specify kernel here
        C = cv.filter2D(A, -1, B) # Convolve
        #H = np.floor(np.array(B.shape)/2).astype(np.int) # Find half dims of kernel
        #C = C[H[0]:-H[0],H[1]:-H[1]] # Cut away unwanted information
        return np.reshape(C,[oldsize*oldsize])