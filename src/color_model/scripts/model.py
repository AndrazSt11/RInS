import os
import math
import numpy as np
from skimage import io
from matplotlib import pyplot as plt 

import sklearn.metrics as metrics
from sklearn import neighbors
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
        # TODO: convert to HSV color space
        return

    elif mode == "RGB":
        RChannel = image[:, :, 0].flatten()
        GChannel = image[:, :, 1].flatten()
        BChannel = image[:, :, 2].flatten()

        RHist = np.histogram(RChannel, numBins, range=(0, 255), density=True)[0] 
        GHist = np.histogram(GChannel, numBins, range=(0, 255), density=True)[0]
        BHist = np.histogram(BChannel, numBins, range=(0, 255), density=True)[0]

        # plt.plot(range(0, 256, 1), RHist, 'r-')
        # plt.plot(range(0, 256, 1), GHist, 'g-')
        # plt.plot(range(0, 256, 1), BHist, 'b-')
        # plt.show()

        return np.concatenate((RHist, GHist, BHist))   

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

def trainKnn(numNeighbours, hist_train, hist_test, label_train, label_test, colors):
    modelName = "Knn" + str(numNeighbours)
    
    # Train
    knnClf = neighbors.KNeighborsClassifier(numNeighbours, weights="distance")
    knnClf.fit(hist_train, label_train)
    label_predicted = knnClf.predict(hist_test)

    # Report results
    file = open("../Results/" + modelName + ".txt", "a")
    file.write("--------------------Results------------------\n")
    file.write("Number of neighbours: " + str(numNeighbours) + "\n")
    file.write("Classification accuracy: " + np.array2string(metrics.accuracy_score(label_test, label_predicted, normalize=True)) + "\n")
    file.write("Percision: " + np.array2string(metrics.precision_score(label_test, label_predicted, labels=range(0, len(colors), 1), average='micro')) + "\n")
    file.write("Recall: " + np.array2string(metrics.recall_score(label_test, label_predicted, labels=range(0, len(colors), 1), average='micro')) + "\n")
   
    confusionMatrix = metrics.confusion_matrix(label_test, label_predicted, labels=range(0, len(colors), 1))
    printConfusionMatrix(file, confusionMatrix, colors)

    file.close()

    # Save model
    joblib.dump(knnClf, "../Models/" + modelName + ".pkl")


# 1) Load images
# 2) Prepare train and test dataset -> Compute histograms and normalize them
# 3) Train different classificiers
# 4) Test classificiers
def main():
    # Prepare dataset
    numBins = 256
    testSize = 0.3
    colorMode = "RGB"
    colors = ["Black", "Blue", "Green", "Red", "White", "Yellow"]

    hist_train, hist_test, label_train, label_test = generateDataset(numBins, colorMode, testSize, colors)

    # KNN multiclass clasification
    print("KNN classification")
    numNeighbours = 12
    trainKnn(numNeighbours, hist_train, hist_test, label_train, label_test, colors)
    

    # TODO: 
    # Train different classificiers
    # Test on HSV
    # Report
    
                   
if __name__ == '__main__':
    main()