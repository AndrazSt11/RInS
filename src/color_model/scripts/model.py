import os
import math
import numpy as np
from skimage import io
from skimage.color import rgb2hsv
from matplotlib import pyplot as plt 

import sklearn.metrics as metrics
from sklearn import neighbors
from sklearn import ensemble 
from sklearn import tree 
from sklearn import svm 
from sklearn import neural_network
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import joblib
import texttable

def getNumericalLabel(color):
    if color == "Black":
        return 0
    elif color == "Blue":
        return 1
    elif color == "Green":
        return 2
    elif color == "Red":
        return 3
    elif color == "White":
        return 4
    elif color == "Yellow":
        return 5

def getStringLabel(numLabel):
    if numLabel == 0:
        return "Black"
    elif numLabel == 1:
        return "Blue"
    elif numLabel == 2:
        return "Green"
    elif numLabel == 3:
        return "Red"
    elif numLabel == 4:
        return "White"
    elif numLabel == 5:
        return "Yellow"

def printConfusionMatrix(file, confusionMatrix, colors):
    file.write("--Confusion matrix--" + "\n")

    table = texttable.Texttable()
    table.add_row(["true / predicted"] + colors)
    for i in range(0, len(colors)): 
        confusionRow = confusionMatrix[i, :]
        confusionRow = [str(item) for item in confusionRow]
        table.add_row([colors[i]] + confusionRow)

    file.write(table.draw() + "\n")

def computeHistogram(image, numBins, mode):
    if mode == "HSV":
        image = rgb2hsv(image)

    Channel1 = image[:, :, 0].flatten()
    Channel2 = image[:, :, 1].flatten()
    Channel3 = image[:, :, 2].flatten()

    Hist1 = np.histogram(Channel1, numBins, range=(0, 255), density=True)[0] 
    Hist2 = np.histogram(Channel2, numBins, range=(0, 255), density=True)[0]
    Hist3 = np.histogram(Channel3, numBins, range=(0, 255), density=True)[0]

    # plt.plot(range(0, 256, 1), Hist1, 'r-')
    # plt.plot(range(0, 256, 1), Hist2, 'g-')
    # plt.plot(range(0, 256, 1), Hist3, 'b-')
    # plt.show()

    return np.concatenate((Hist1, Hist2, Hist3))   

def generateDataset(numBins, colorMode, testSize, colors):
    dataSetPath = "../DataSet/"

    dataSetHist = []
    dataSetLabels = []

    # Compute dataset - histograms
    for color in colors:
        print("Preprocessing " + color)

        path = dataSetPath + color + "/"
        sourceImageNames = os.listdir(path)
        label = getNumericalLabel(color)

        for sourceImageName in sourceImageNames:
            sourceImage = io.imread(path + sourceImageName)  # uint8 
            histogram = computeHistogram(sourceImage, numBins, "RGB")
    

            dataSetHist.append(histogram)
            dataSetLabels.append(label)

    return train_test_split(dataSetHist, dataSetLabels, test_size=testSize, shuffle=True)

def printResults(modelName, label_test, label_predicted, colors):
    file = open("../Results/modelResults.txt", "a")
    file.write("--------------------" + modelName + "------------------\n")
    file.write("Classification accuracy: " + np.array2string(metrics.accuracy_score(label_test, label_predicted, normalize=True)) + "\n")
    file.write("Percision: " + np.array2string(metrics.precision_score(label_test, label_predicted, labels=range(0, len(colors), 1), average='macro')) + "\n")
    file.write("Recall: " + np.array2string(metrics.recall_score(label_test, label_predicted, labels=range(0, len(colors), 1), average='macro')) + "\n")
   
    confusionMatrix = metrics.confusion_matrix(label_test, label_predicted, labels=range(0, len(colors), 1))
    printConfusionMatrix(file, confusionMatrix, colors)

    file.write("\n")
    file.close()

    # Save confusion matrix as image
    disp = ConfusionMatrixDisplay(confusionMatrix, display_labels=colors).plot()
    plt.savefig("confusion_matrix_" + modelName + ".pdf")

def trainKnn(numNeighbours, hist_train, hist_test, label_train, label_test, colors, colorMode):
    modelName = "Knn" + str(numNeighbours) + "neighbours" + colorMode
    
    # Train
    knnClf = neighbors.KNeighborsClassifier(numNeighbours, weights="distance")
    knnClf.fit(hist_train, label_train)
    label_predicted = knnClf.predict(hist_test)

    # Report results
    printResults(modelName, label_test, label_predicted, colors)

    # Save model
    joblib.dump(knnClf, "../Models/" + modelName + ".pkl")

def trainDecisionTree(hist_train, hist_test, label_train, label_test, colors, colorMode):
    modelName = "DecisionTree" + colorMode
    
    # Train
    decisionTreeClf = tree.DecisionTreeClassifier()
    decisionTreeClf.fit(hist_train, label_train)
    label_predicted = decisionTreeClf.predict(hist_test)

    # Report results
    printResults(modelName, label_test, label_predicted, colors)

    # Save model
    joblib.dump(decisionTreeClf, "../Models/" + modelName + ".pkl")


def trainRandomForest(numTrees, hist_train, hist_test, label_train, label_test, colors, colorMode):
    modelName = "RandomForest" + str(numTrees) + "trees" + colorMode
    
    # Train
    randomForestClf = ensemble.RandomForestClassifier(n_estimators=numTrees)
    randomForestClf.fit(hist_train, label_train)
    label_predicted = randomForestClf.predict(hist_test)

    # Report results
    printResults(modelName, label_test, label_predicted, colors)

    # Save model
    joblib.dump(randomForestClf, "../Models/" + modelName + ".pkl")

def trainSVM(hist_train, hist_test, label_train, label_test, colors, colorMode):
    modelName = "SVM" + colorMode
    
    # Train
    svmClf = svm.SVC(kernel="rbf")
    svmClf.fit(hist_train, label_train)
    label_predicted = svmClf.predict(hist_test)

    # Report results
    printResults(modelName, label_test, label_predicted, colors)

    # Save model
    joblib.dump(svmClf, "../Models/" + modelName + ".pkl")

def trainMLP(hist_train, hist_test, label_train, label_test, colors, colorMode):
    modelName = "MLP" + colorMode
    
    # Train
    mlpClf = neural_network.MLPClassifier(hidden_layer_sizes=(128, 128), activation="relu", solver="adam", alpha=0.0001,
                                          max_iter=300, n_iter_no_change=20)
    mlpClf.fit(hist_train, label_train)
    label_predicted = mlpClf.predict(hist_test)

    # Report results
    printResults(modelName, label_test, label_predicted, colors)

    # Save model
    joblib.dump(mlpClf, "../Models/" + modelName + ".pkl")


# 1) Load images
# 2) Prepare train and test dataset -> Compute histograms and normalize them
# 3) Train different classificiers
# 4) Test classificiers
def main():
    # Prepare dataset
    numBins = 256
    testSize = 0.3
    colors = ["Black", "Blue", "Green", "Red", "White", "Yellow"]

    hist_train_RGB, hist_test_RGB, label_train_RGB, label_test_RGB = generateDataset(numBins, "RGB", testSize, colors)
    hist_train_HSV, hist_test_HSV, label_train_HSV, label_test_HSV = generateDataset(numBins, "HSV", testSize, colors)

    # KNN multiclass clasification
    print("KNN classification-RGB")
    numNeighbours = 12
    trainKnn(numNeighbours, hist_train_RGB, hist_test_RGB, label_train_RGB, label_test_RGB, colors, "RGB")

    print("KNN classification-HSV")
    numNeighbours = 12
    trainKnn(numNeighbours, hist_train_HSV, hist_test_HSV, label_train_HSV, label_test_HSV, colors, "HSV")
    
    # Decision tree
    print("Decision tree-RGB")
    trainDecisionTree(hist_train_RGB, hist_test_RGB, label_train_RGB, label_test_RGB, colors, "RGB")

    print("Decision tree-HSV")
    trainDecisionTree(hist_train_HSV, hist_test_HSV, label_train_HSV, label_test_HSV, colors, "HSV")

    # Random forest
    print("Random forest-RGB")
    numTrees = 200
    trainRandomForest(numTrees, hist_train_RGB, hist_test_RGB, label_train_RGB, label_test_RGB, colors, "RGB")

    print("Random forest-HSV")
    numTrees = 200
    trainRandomForest(numTrees, hist_train_HSV, hist_test_HSV, label_train_HSV, label_test_HSV, colors, "HSV")

    # Support vector machine
    print("SVM-RGB")
    trainSVM(hist_train_RGB, hist_test_RGB, label_train_RGB, label_test_RGB, colors, "RGB")

    print("SVM-HSV")
    trainSVM(hist_train_HSV, hist_test_HSV, label_train_HSV, label_test_HSV, colors, "HSV")

    # MLP
    print("MLP-RGB")
    trainMLP(hist_train_RGB, hist_test_RGB, label_train_RGB, label_test_RGB, colors, "RGB")

    print("MLP-HSV")
    trainMLP(hist_train_HSV, hist_test_HSV, label_train_HSV, label_test_HSV, colors, "HSV")


if __name__ == '__main__':
    main()