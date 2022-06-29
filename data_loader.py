from csv import list_dialects
import pandas as pd
import tensorflow as tf
import os

#At this point basically just a shortcut for some pandas splicing

#Given a csv file, load in the specified categories, as well as allow grouping of said categories
#And seperate the data out into those categories. Time seperation can then be done by simple splicing
class DataFileLoader:

    #Groupings = { Name: ["categoryName1", "categoryName2"] }
    def __init__(self, fname, label = None):
        self.fname = fname
        self.df = pd.read_csv(fname)
        self.colNames = list(self.df.columns)
        self.label = label
    
    #Get a dataset of (windows, labels) of windowSize and with data from groups, shifting by shift
    def getWindow(self, windowSize, groups = None, shift=1):
        if groups == None: #Getting all columns
            groups = self.colNames

        relevantData = self.df[groups]
        #Load in the data and window it
        dataset = tf.data.Dataset.from_tensor_slices(relevantData).window(windowSize, shift=shift, drop_remainder=True)
        #Convert from dataset of datasets to dataset of windows
        dataset = dataset.flat_map(lambda ds: ds.batch(windowSize, drop_remainder=True))
        #reshape so that each individual feature is in its own tensor and add in label
        dataset = dataset.map(lambda v: (tf.reshape(v, (windowSize, len(groups), 1)), int(self.label)))

        return dataset
    
    def getData(self, groups = None):
        if groups == None:
            groups = self.colNames
        
        relevantData = self.df[groups]
        dataset = tf.data.Dataset.from_tensor_slices(relevantData)
        dataset = dataset.map(lambda v: (v, self.fname))

        return dataset
        
class DataLoader:

    #groupings = list of relevant 
    def __init__(self, labeledFiles):
        """labeledFiles should be a dictionary of structure: { StringLabel : { value : num0, files: [fname1, fname2, ...] }, ... }"""
        #Load in all the dataFiles
        self.dataFiles = []
        for sLabel, data in labeledFiles.items():
            numericLabel = data["value"]
            for fname in data["files"]:
                self.dataFiles.append(DataFileLoader(fname, numericLabel))

    #Using all loaded files, get windows from them and return a dataset
    def getWindow(self, windowSize, groups = None, shift=1):
        dataSets = []
        for dataFile in self.dataFiles:
            dataSets.append(dataFile.getWindow(windowSize, groups=groups, shift=shift))
        
        finalSet = dataSets[0]
        for i in range(1, len(dataSets)):
            finalSet = finalSet.concatenate(dataSets[i])
        
        return finalSet
    
    def getData(self, groups = None):
        datasets = []
        for dataFile in self.dataFiles:
            datasets.append(dataFile.getData(groups))
        
        finalSet = datasets[0]
        for i in range(1, len(datasets)):
            finalSet = finalSet.concatenate(datasets[i])
        
        return finalSet


files = {
        "Walking" : { "value": 0, "files": [] },
        "Running" : { "value": 1, "files": [] },
        "Crawling" : { "value": 2, "files": [] },
        "StairsUp" : { "value": 3, "files": [] },
        "StairsDown" : { "value": 4, "files": [] }
}


def checkForBadValues(loader):
    allData = loader.getData(["accel_x", "accel_y", "accel_z"])
    print("Datalines:", len(list(allData)))

    badFiles = set()

    import numpy as np
    for data, label in allData:
        m = np.mean(data.numpy())
        if np.isnan(m) or np.abs(m) > 2:
            print("Label:", label.numpy())
            print("Data:", data.numpy())
            badFiles.add(label.numpy().decode())

    print(f"Corrupt files: {badFiles}")
    print("Removing bad files")

    for fileName in badFiles:
        print(fileName)
        if os.path.exists(fileName):
            os.remove(fileName)
        else:
            print("No file")

if __name__ == "__main__":

    files = {
        "Walking" : { "value": 0, "files": [] },
        "Running" : { "value": 1, "files": [] },
        "Crawling" : { "value": 2, "files": [] },
        "StairsUp" : { "value": 3, "files": [] },
        "StairsDown" : { "value": 4, "files": [] }
    }

    speeds = ["Average", "Fast", "Slow"]
    
    
    for action in files.keys():
        for speed in speeds:
            folderPath = os.path.join("DataResults", f"{action}Results", speed)
            filenames = [f for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f))]
            filenames = [os.path.join(folderPath, f) for f in filenames]
            files[action]["files"] = filenames
    
    loader = DataLoader(files)

    checkForBadValues(loader)






    