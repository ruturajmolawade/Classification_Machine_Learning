import pickle
import gzip
from PIL import Image
import os
import numpy as np

import tensorflow as tf
from tqdm import tqdm_notebook
import pandas as pd
from keras.utils import np_utils
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
# MNIST data
filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()
trainingData = np.array(training_data[0][:2000])
print("Training Data size ",trainingData.shape)
trainingTarget = np.array(training_data[1][:2000])
print("Training Target size ",trainingTarget.shape)
print(trainingTarget)

# USPS Data
USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)

# defining weight matrix
W = np.ones((784,10))

Z = np.dot(W.T,trainingData.T)

def softmax(y_linear):
    exp = np.exp(y_linear-np.max(y_linear, axis=1).reshape((-1,1)))
    norms = np.sum(exp, axis=1).reshape((-1,1))
    return exp / norms

#a = softmax(Z)
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))


def oneHotVector(trainingTarget):
    V = np.zeros((trainingData.shape[0],10),dtype='uint8')
    for i in range(trainingTarget.shape[0]):
        x = np.int8(trainingTarget[i])
        V[i][x] = 1
        #V[i] = temp
    return V

y = oneHotVector(trainingTarget)
print(y)

def calculateWeights(X, y):
    W = np.zeros((X.shape[1],10))
    lr = 0.05
    for i in range(1500):
        z = np.dot(X,W)
        h = softmax(z)
        gradient = np.dot(X.T, (h - y)) / y.shape[0]
        W -= lr * gradient   

    return W

def calculateWeightsSGD(X, y):
    W = np.zeros((X.shape[1],10))
    BATCH_SIZE = 128
    lr = 0.05
    for i in range(1500):
        for start in range(0, len(X), BATCH_SIZE):
            end = start + BATCH_SIZE
           # y = y[start:end]    
            z = np.dot(X[start:end],W)
            h = softmax(z)
            # print("y shape",y.shape)
            #print(W[start:end].shape)
            #print(h.shape)
            gradient = np.dot(X[start:end].T, (h - y[start:end])) / y[start:end].shape[0]
            W -= lr * gradient   

    return W    

W = calculateWeights(trainingData,y)

def createConfusionMatrix(predictedValues, targetValues,classifier):
    confusion_matrix = np.array([[0 for x in range(10)] for y in range(10)])
    for i in range(0,predictedValues.shape[0]):
        if classifier == "SOFTMAX":
            x = np.argmax(predictedValues[i])
        else:
            x = predictedValues[i]
        y = targetValues[i]
        confusion_matrix[x][y] += 1
    return confusion_matrix

def getAccuracy(predictedValues, targetValues):
    accuracy = 0
    for i in range(0,predictedValues.shape[0]):
        if np.argmax(predictedValues[i]) == targetValues[i]:
            accuracy = accuracy + 1
    return float(accuracy*100)/predictedValues.shape[0]

def getAccuracyAnother(predictedValues, targetValues):
    accuracy = 0
    for i in range(0,predictedValues.shape[0]):
        if predictedValues[i] == targetValues[i]:
            accuracy = accuracy + 1
    return float(accuracy*100)/predictedValues.shape[0]

def finalConfusionMatrix(confusion_matrix):
    true_positive = 0;
    false_positive = 0;
    false_negative = 0;
    true_negative = 0;
    for i in range(10):
        for j in range(10):
            if i == j:
                true_positive += confusion_matrix[i][j]
            # else:
            #     false_positive += 

def defineModel(M): 
    inputTensor  = tf.placeholder(tf.float32, [None, M])
    outputTensor = tf.placeholder(tf.float32, [None, 10])

    NUM_HIDDEN_NEURONS_LAYER_1 = 200
    NUM_HIDDEN_NEURONS_LAYER_2 = 200
    LEARNING_RATE = 0.05

    #input_hidden_weights  = init_weights([M, NUM_HIDDEN_NEURONS_LAYER_1])
    input_hidden_weights_layer1  = init_weights([M, NUM_HIDDEN_NEURONS_LAYER_1])
    input_hidden_weights_layer2  = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, NUM_HIDDEN_NEURONS_LAYER_2])
    hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_2, 10])

    hidden_layer_1 = tf.nn.sigmoid(tf.matmul(inputTensor, input_hidden_weights_layer1))
    hidden_layer_2 = tf.nn.sigmoid(tf.matmul(hidden_layer_1, input_hidden_weights_layer2))
    output_layer = tf.matmul(hidden_layer_2, hidden_output_weights)

    # Defining Error Function
    error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))
    training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(error_function)
    # Prediction Function
    prediction = tf.argmax(output_layer, 1)

    return prediction, training, inputTensor, outputTensor

def trainModel(prediction,training, inputTensor, outputTensor,trainingData,trainingTarget,validationData,validationTarget,testingData,testingTarget,USPS_Data,USPS_Tar):
    NUM_OF_EPOCHS = 400
    BATCH_SIZE = 128

    training_accuracy = []

    with tf.Session() as sess:
    
        tf.global_variables_initializer().run()
    
        for epoch in tqdm_notebook(range(NUM_OF_EPOCHS)):
            
            #Shuffle the Training Dataset at each epoch
            p = np.random.permutation(range(len(trainingData)))
            trainingData  = trainingData[p]
            trainingTarget = trainingTarget[p]
            
            # Start batch training
            for start in range(0, len(trainingData), BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(training, feed_dict={inputTensor: trainingData[start:end], 
                                          outputTensor: trainingTarget[start:end]})
            # Training accuracy for an epoch
            training_accuracy.append(np.mean(np.argmax(trainingTarget, axis=1) ==
                             sess.run(prediction, feed_dict={inputTensor: trainingData,
                                                             outputTensor: trainingTarget})))
        # Testing
        # this line of code gives predicted output for testing data based on above trained model
        validationPredictValues = sess.run(prediction, feed_dict={inputTensor: validationData})
        validation_accuracy = getAccuracyAnother(validationPredictValues, validationTarget)
        print("Validation Accuracy: ",validation_accuracy)
        testingPredictValues = sess.run(prediction, feed_dict={inputTensor: testingData})
        testing_accuracy = getAccuracyAnother(testingPredictValues, testingTarget) 
        print("Testing Accuracy: ",testing_accuracy)
        USPSPredictValues = sess.run(prediction, feed_dict={inputTensor: USPS_Data})
        USPS_accuracy = getAccuracyAnother(USPSPredictValues, USPS_Tar)
        print("USPS Accuracy: ",USPS_accuracy) 
        return testingPredictValues,USPSPredictValues

def defineSVMModel(trainingData,trainingTarget,kernelType):
    clf = svm.SVC(gamma=0.05,kernel=kernelType)
    clf.fit(trainingData, trainingTarget)
    return clf

def predictedValues(clf,data):
    predictedValues = clf.predict(data)
    return predictedValues

def ensembleResult(softmaxRegressionPredictedValues,softmaxRegressionPredictedValuesSGD,neuralNertworksPredictedValues,SVM_RBFPredictedValues,SVM_LinearPredictedValues,randomForestPredictedValues):
    enesemblePredictedValues =[]
    for i in range(neuralNertworksPredictedValues.shape[0]):
        temp_list = [np.argmax(softmaxRegressionPredictedValues[i]),np.argmax(softmaxRegressionPredictedValuesSGD[i]),neuralNertworksPredictedValues[i],SVM_RBFPredictedValues[i],SVM_LinearPredictedValues[i],randomForestPredictedValues[i]]
        data = Counter(temp_list)
        x = data.most_common(1)[0][0]
        enesemblePredictedValues.append(x)
    return enesemblePredictedValues

validationData = np.array(validation_data[0][:2000])
validationTarget = np.array(validation_data[1][:2000])
testingData = np.array(test_data[0][:2000])
testingTarget = np.array(test_data[1][:2000])
print("______________________ SOFTMAX REGRESSION ________________________________________")

validationPredictValues = softmax(np.dot(validationData,W))
validation_accuracy = getAccuracy(validationPredictValues,validationTarget)
print("validation_accuracy - ",validation_accuracy)

testingPredictValuesSoftmax = softmax(np.dot(testingData,W))
testing_accuracy = getAccuracy(testingPredictValuesSoftmax,testingTarget)
print("testing_accuracy - ",testing_accuracy)
print("Confusion Matrix MNIST Testing Data Softmax Regression- ")
confusion_matrix = createConfusionMatrix(testingPredictValuesSoftmax,testingTarget,'SOFTMAX')
print(confusion_matrix)

USPS_predictedValuesSoftmax = softmax(np.dot(USPSMat,W))
USPS_accuracy  = getAccuracy(USPS_predictedValuesSoftmax,USPSTar)
print("USPS_accuracy - ",USPS_accuracy)
print("Confusion Matrix USPS Data Softmax Regression- ")
confusion_matrix = createConfusionMatrix(USPS_predictedValuesSoftmax,USPSTar,'SOFTMAX')
print(confusion_matrix)

print("_______________________ NEURAL NETWORKS ___________________________________________")
prediction_model,training, inputTensor, outputTensor = defineModel(trainingData.shape[1])
trainingTargetOneHotVector = oneHotVector(trainingTarget)
testingPredictValuesNeuralNetworks,USPSPredictValuesNeuralNetworks = trainModel(prediction_model,training, inputTensor, outputTensor,trainingData,trainingTargetOneHotVector,validationData,validationTarget,testingData,testingTarget,USPSMat,USPSTar)
print("Confusion Matrix MNIST Testing Data Neural Networks- ")
confusion_matrix = createConfusionMatrix(testingPredictValuesNeuralNetworks,testingTarget,"")
print(confusion_matrix)
print("Confusion Matrix USPS Data Neural Networks- ")
confusion_matrix = createConfusionMatrix(USPSPredictValuesNeuralNetworks,USPSTar,"")
print(confusion_matrix)

print("_________________________ SVM _______________________________________________________")
print("Kernel - RBF ")
clf = defineSVMModel(trainingData,trainingTarget,'rbf')
validationPredictValues = predictedValues(clf,validationData)
testingPredictedValuesSVMRBF = predictedValues(clf,testingData)
validation_accuracy = getAccuracyAnother(validationPredictValues,validationTarget)
print("validation_accuracy - ",validation_accuracy)
testing_accuracy = getAccuracyAnother(testingPredictedValuesSVMRBF,testingTarget)
print("testing_accuracy - ",testing_accuracy)
USPS_predictedValuesSVMRBF = predictedValues(clf,USPSMat)
USPS_accuracy = getAccuracyAnother(USPS_predictedValuesSVMRBF,USPSTar)
print("USPS_accuracy - ",USPS_accuracy)
print("Confusion Matrix MNIST Testing Data SVM RBF- ")
confusion_matrix = createConfusionMatrix(testingPredictedValuesSVMRBF,testingTarget,"")
print(confusion_matrix)
print("Confusion Matrix USPS Data SVM RBF- ")
confusion_matrix = createConfusionMatrix(USPS_predictedValuesSVMRBF,USPSTar,"")
print(confusion_matrix) 

print("\nKernel - Linear ")
clf = defineSVMModel(trainingData,trainingTarget,'linear')
validationPredictValues = predictedValues(clf,validationData)
testingPredictedValuesSVMLinear = predictedValues(clf,testingData)
validation_accuracy = getAccuracyAnother(validationPredictValues,validationTarget)
print("validation_accuracy - ",validation_accuracy)
testing_accuracy = getAccuracyAnother(testingPredictedValuesSVMLinear,testingTarget)
print("testing_accuracy - ",testing_accuracy)
USPS_predictedValuesSVMLinear = predictedValues(clf,USPSMat)
USPS_accuracy = getAccuracyAnother(USPS_predictedValuesSVMLinear,USPSTar)
print("USPS_accuracy - ",USPS_accuracy)
print("Confusion Matrix MNIST Testing Data SVM Linear- ")
confusion_matrix = createConfusionMatrix(testingPredictedValuesSVMLinear,testingTarget,"")
print(confusion_matrix)
print("Confusion Matrix USPS Data SVM Linear- ")
confusion_matrix = createConfusionMatrix(USPS_predictedValuesSVMLinear,USPSTar,"")
print(confusion_matrix)

print("______________________ RANDOM FOREST __________________________________________________") 

clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(trainingData, trainingTarget)
validationPredictValues = predictedValues(clf,validationData)
testingPredictedValuesRandomForest = predictedValues(clf,testingData)
USPS_predictedValuesRandomForest = predictedValues(clf,USPSMat)
validation_accuracy = getAccuracyAnother(validationPredictValues,validationTarget)
print("validation_accuracy - ",validation_accuracy)
testing_accuracy = getAccuracyAnother(testingPredictedValuesRandomForest,testingTarget)
print("testing_accuracy - ",testing_accuracy)
USPS_accuracy = getAccuracyAnother(USPS_predictedValuesRandomForest,USPSTar) 
print("USPS_accuracy - ",USPS_accuracy)
print("Confusion Matrix MNIST Testing Data Random Forest- ")
confusion_matrix = createConfusionMatrix(testingPredictedValuesRandomForest,testingTarget,"")
print(confusion_matrix)
print("Confusion Matrix USPS Data Random Forest- ")
confusion_matrix = createConfusionMatrix(USPS_predictedValuesRandomForest,USPSTar,"")
print(confusion_matrix)

print("___________________________ SGD ______________________________________________________________")
W = calculateWeightsSGD(trainingData,y)
validationPredictValues = softmax(np.dot(validationData,W))
validation_accuracy = getAccuracy(validationPredictValues,validationTarget)
print("validation_accuracy - ",validation_accuracy)

testingPredictValuesSoftmaxSGD = softmax(np.dot(testingData,W))
testing_accuracy = getAccuracy(testingPredictValuesSoftmaxSGD,testingTarget)
print("testing_accuracy - ",testing_accuracy)
print("Confusion Matrix MNIST Testing Data Softmax Regression- ")
confusion_matrix = createConfusionMatrix(testingPredictValuesSoftmaxSGD,testingTarget,'SOFTMAX')
print(confusion_matrix)

USPS_predictedValuesSoftmaxSGD = softmax(np.dot(USPSMat,W))
USPS_accuracy  = getAccuracy(USPS_predictedValuesSoftmaxSGD,USPSTar)
print("USPS_accuracy - ",USPS_accuracy)
print("Confusion Matrix USPS Data Softmax Regression- ")
confusion_matrix = createConfusionMatrix(USPS_predictedValuesSoftmaxSGD,USPSTar,'SOFTMAX')
print(confusion_matrix)

print("___________________ COMBINED RESULT ___________________________________________________")

enesemblePredictedValuesMNISTTesting = ensembleResult(testingPredictValuesSoftmax,testingPredictValuesSoftmaxSGD,testingPredictValuesNeuralNetworks,testingPredictedValuesSVMRBF,testingPredictedValuesSVMLinear,testingPredictedValuesRandomForest)
#print(enesemblePredictedValuesMNISTTesting)
enesemblePredictedValuesMNISTTesting = np.array(enesemblePredictedValuesMNISTTesting)
print("shape1",enesemblePredictedValuesMNISTTesting.shape)
print(enesemblePredictedValuesMNISTTesting)
print(testingData)
print("shape2",testingData.shape)
print(type(enesemblePredictedValuesMNISTTesting[0]))
print(type(testingData[0]))
enesemblePredictedValues_accuracy =getAccuracyAnother(enesemblePredictedValuesMNISTTesting,testingTarget)
print(enesemblePredictedValues_accuracy)

enesemblePredictedValuesUSPS = np.array(ensembleResult(USPS_predictedValuesSoftmax,USPS_predictedValuesSoftmaxSGD,USPSPredictValuesNeuralNetworks,USPS_predictedValuesSVMRBF,USPS_predictedValuesSVMLinear,USPS_predictedValuesRandomForest))
enesemblePredictedValuesUSPS_accuracy =getAccuracyAnother(enesemblePredictedValuesUSPS,USPSTar)
print(enesemblePredictedValuesUSPS_accuracy)

