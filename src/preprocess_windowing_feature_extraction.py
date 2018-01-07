

# Description: Extracts the features from the .dat file given

## Importing python modules

import pandas as pd
from os.path import join, exists
import os
from scipy import stats
import numpy as np
from sklearn import preprocessing

win_size = int(sys.argv[1])
columnArr = ["activityID",
              "handT", "handAx_16", "handAy_16", "handAz_16",
              "handAx_6", "handAy_6", "handAz_6",
              "handGx", "handGy", "handGz",
              "handMx", "handMy", "handMz",
              "chestT", "chestAx_16", "chestAy_16", "chestAz_16",
              "chestAx_6", "chestAy_6", "chestAz_6",
              "chestGx", "chestGy", "chestGz",
              "chestMx", "chestMy", "chestMz",
              "ankleT", "ankleAx_16", "ankleAy_16", "ankleAz_16",
              "ankleAx_6", "ankleAy_6", "ankleAz_6",
              "ankleGx", "ankleGy", "ankleGz",
              "ankleMx", "ankleMy", "ankleMz"]

#calculating the MAV
def calculateMAV(dataF):
    absoluteDataFrame = dataF.abs()
    MAVFrame = absoluteDataFrame.mean()
    return MAVFrame

# calculating harmonic mean
def calculateHarmonicMean(dataF):
    dataF.fillna(1)
    hMeanFrame = pd.DataFrame(stats.hmean(dataF.abs(), axis=0))
    return hMeanFrame

# calculating RMS
def calculateRMS(dataF):
    squaredFrame = dataF ** 2
    meanFrame = squaredFrame.mean()
    RMSFrame = meanFrame.apply(np.sqrt)
    return RMSFrame

#calculating variance
def calculateVariance(dataF):
    varFrame = dataF.var()
    return varFrame

# calculating cummulative length
def calculateCumLength(dataF):
    diffFrame = dataF.diff()
    cumLengthFrame = diffFrame.sum()
    return cumLengthFrame

# calcualting skew
def calculateSkew(dataF):
    skewFrame = dataF.skew()
    return skewFrame

# calculating energy
def calculateEnergy(dataF):
    squaredFrame = dataF ** 2
    energyFrame = squaredFrame.sum()
    return energyFrame

# reading the data from the file. window size is configurable
def readDataFromFile(filePath, windowSize = win_size):

    print('Reading data from file ' + filePath)
    data = pd.read_csv(filePath, sep=' ', header=None)

    print(data.shape)

    print("Removing unused columns from the data")

    # removing orientation columns from the data
    data = data.drop(data.columns[[0, 2, 16, 17, 18, 19, 33, 34, 35, 36, 50, 51, 52, 53]], axis=1)
    print(data.shape)

    # creating of copy of the entire dataFrame
    df = data.copy()
    df.columns = columnArr
    (dataRows, dataCols) = df.shape

    # replacing the Nan values
    df.fillna(method='ffill')
    df.fillna(0)

    # extracting features from each column
    # MAV - Mean Absolute Value
    # Harmonic mean
    # RMS
    # Variance
    # Cumulative length
    # Skewness
    # Energy of the signal


    destDf = pd.DataFrame()  # destination data frame
    startIndex = 0
    endIndex = startIndex + windowSize
    shouldLoopContinue = True
    while (shouldLoopContinue):
        window = df.iloc[startIndex:endIndex, :]

        # dividing data based on the label
        labelList = window['activityID'].unique().tolist()
        #print(labelList)

        for label in labelList:
                window_div = window[window["activityID"] == label]
                sensorFrame = window_div.iloc[:, 1:40]

                rowFrame = pd.DataFrame([window_div.iloc[0, 0]])
                MAVFrame = calculateMAV(sensorFrame)
                # HMeanFrame = calculateHarmonicMean(sensorFrame)
                RMSFrame = calculateRMS(sensorFrame)
                varFrame = calculateVariance(sensorFrame)
                cumLengthFrame = calculateCumLength(sensorFrame)
                skewFrame = calculateSkew(sensorFrame)
                energyFrame = calculateEnergy(sensorFrame)
                rowFrame = pd.concat([rowFrame, MAVFrame, RMSFrame, varFrame, cumLengthFrame, skewFrame, energyFrame], ignore_index=True)
                destDf = destDf.append(rowFrame.transpose(), ignore_index=True)
               

        startIndex = endIndex
        endIndex = startIndex + windowSize

        if (startIndex < dataRows and endIndex < dataRows):
            shouldLoopContinue = True
        elif (startIndex < dataRows and endIndex > dataRows):
            endIndex = dataRows
            shouldLoopContinue = True
        else:
            shouldLoopContinue = False
    print(destDf.shape)
    return destDf

def dumpFeaturesToFile(fileArr, sourceDir, dataPath, labelPath):

    combinedDf = pd.DataFrame()
    for fileName in fileArr:
        filePath = join(sourceDir, fileName)
        retreivedDf = readDataFromFile(filePath)
        if ((retreivedDf.shape)[0] > 0):
            combinedDf = combinedDf.append(retreivedDf, ignore_index=True)
            print(combinedDf.shape)

    combinedDf.iloc[:, 1:235].to_csv(dataPath, index=False)
    combinedDf.iloc[:,0].to_csv(labelPath, index=False)


def main():
    # initializing the directories
    sourceDir = "../raw_data/"
    destinationDir = "../"
    preprocessedDirName = "PreProccessed_Data_Window_Sizewise/W"+str(win_size)
    preprocessedDirPath = join(destinationDir, preprocessedDirName)
    train_data = "Train_Data.csv"
    train_label = "Train_Label.csv"
    validation_data = "Validation_Data.csv"
    validation_label = "Validation_Label.csv"
    test_data = "Test_Data.csv"
    test_label = "Test_Label.csv"

    #creating destination directories if not present
    if not exists(preprocessedDirPath):
        os.makedirs(preprocessedDirPath)

    trainingFiles = ["subject101.dat", "subject102.dat", "subject103.dat", "subject104.dat", "subject105.dat", "subject106.dat","subject107.dat","subject108.dat","subject109.dat"]
    #validationFiles = ["subject108.dat"]
    #testingFiles = ["subject109.dat"]

    # Generating the features for the training data
    dumpFeaturesToFile(trainingFiles, sourceDir, join(preprocessedDirPath, train_data), join(preprocessedDirPath, train_label))

    # Generating the features for the validation data
    #dumpFeaturesToFile(validationFiles, sourceDir, join(preprocessedDirPath, validation_data), join(preprocessedDirPath, validation_label), False, True)

    # Generating the features for the test data
    #dumpFeaturesToFile(testingFiles, sourceDir, join(preprocessedDirPath, test_data), join(preprocessedDirPath, test_label), False, True)




if __name__== "__main__":
    main()
