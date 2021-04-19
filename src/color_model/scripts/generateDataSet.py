import os
import math
import random
import numpy as np
from skimage import io
from skimage import util

def colorPreprocess(image, brightness, contrast):
    image = np.int16(image)
    image = image * (contrast/127+1) - contrast + brightness
    image = np.clip(image, 0, 255)
    image = np.uint8(image)
    return image # uint8

def noisePreprocess(image, noiseType):
    image = util.img_as_float(image);

    if noiseType == "gauss":
        mean = random.uniform(-0.05, 0.05)
        std_deviation = random.uniform(0.01, 0.05)
        gauss = np.random.normal(mean, std_deviation, image.shape)
        image = image + gauss
        image = np.clip(image, -1.0, 1.0)
        return util.img_as_ubyte(image) # uint8
    
    elif noiseType == "speckle":
        mean = random.uniform(-0.05, 0.05)
        std_deviation = random.uniform(0.05, 0.1)
        gauss = np.random.normal(mean, std_deviation, image.shape)   
        image = image * gauss + image
        image = np.clip(image, -1.0, 1.0)
        return util.img_as_ubyte(image) # uint8

    elif noiseType == "salt_and_pepper":
        row,col,ch = image.shape
        s_vs_p = random.uniform(0.2, 0.8)
        amount = random.uniform(0.003, 0.02)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        image[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        image[coords] = 0

        image = np.clip(image, -1.0, 1.0)
        return util.img_as_ubyte(image)

def imagePreprocessPipeline(sourceImage, num):
    patchSize = 100;
    noiseType = ["gauss", "speckle", "salt_and_pepper"]
    
    processedImages = []
    width, height, col = sourceImage.shape 

    for i in range(0, num):
        brightness = random.uniform(-20, 20)
        contrast = random.uniform(-30, 30)

        xPatch = math.trunc(random.uniform(0 + patchSize, width - patchSize))
        yPatch = math.trunc(random.uniform(0 + patchSize, height - patchSize))

        newImage = np.copy(sourceImage)
        newImage = newImage[xPatch:xPatch + patchSize, yPatch:yPatch + patchSize] # 50 * 50 image
        newImage = colorPreprocess(newImage, brightness, contrast)
        newImage = noisePreprocess(newImage, noiseType[i % 3])

        processedImages.append(newImage)        

    return processedImages

def saveImages(processedImages, destinationPath, color, sourceImageName):
    i = 0
    fullPath = destinationPath + color + "/"

    for processedImage in processedImages:
        io.imsave(fullPath + sourceImageName + "_" + str(i) + ".png", processedImage)
        i += 1


# 1) Load images from original dataset
# 2) Disect each image --> or sample values!! and then construct image
# 3) Change brightness, contrast and add noise
# 4) imageoput
def main():
    random.seed(1)

    num = 100
    sourcePath = "../DataSetOriginal/"
    destinationPath = "../DataSet/"
    colors = ["Black", "Blue", "Green", "Red", "White", "Yellow"]
    
    for color in colors:
        print("Generating" + color)

        path = sourcePath + color + "/"
        sourceImageNames = os.listdir(path)

        for sourceImageName in sourceImageNames:
            sourceImage = io.imread(path + sourceImageName)  # uint8 
            processedImages = imagePreprocessPipeline(sourceImage, num)
            saveImages(processedImages, destinationPath, color, sourceImageName)
                    

if __name__ == '__main__':
    main()