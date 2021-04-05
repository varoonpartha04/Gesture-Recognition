import task2
import math
import re
import sys
from sklearn.neighbors import NearestNeighbors
import scipy
import numpy as np

if __name__ == "__main__":
    gesture_file = sys.argv[1]
    col = int(sys.argv[2])

file = open(gesture_file, "r")
directory = "output/"



def getCells(filenumber, column,vectorFile):
    vectorFile = open("output/vectors.txt", "r") 
    lines=vectorFile.readlines() 
    line = lines[filenumber]
    a = re.findall(r'\<(.*?)\>', line)
    i = a[column]  
    j = i.split(",")
    res = [string.strip() for string in j]
    return res

def getIDF(totalSensors, termSensors):

    if(termSensors == 0):
        return 0.0
    else:
        return math.log(totalSensors/termSensors)

def getTfIdf(tf, idf, maximumtf):
    return (0.5+ 0.5*tf/maximumtf)*idf




bands = task2.getTerms(directory)
bands = sorted(bands)
idfVec = []
idf2Vec = []
file.seek(0)
mydict = task2.getTF(file)


def getTfIdfList(file,i, resTf):
    file.seek(0)

    termSensor = task2.getTermSensors(file, i)
    file.seek(0)

    resIdf = getIDF(no_of_sensors, termSensor)
    resMaxTf = float(max(mydict, key=mydict.get))

    res_tfIdf = getTfIdf(resTf,resIdf, resMaxTf)
 
    return res_tfIdf

def getTfDdf2List(file, i, resTf):
    file.seek(0)

    resIdf2 = getIDF(task2.getTotalFiles(directory), task2.getTermFilesCount(directory, i))
    resMaxTf2 = float(max(mydict, key=mydict.get))
    resTfIdf2 = task2.getTfIdf2(resTf, resIdf2, resMaxTf2)
    return resTfIdf2

for i in bands:
    resTf = mydict.get(i,0)
    file.seek(0)
    no_of_sensors = task2.getTotalSensors(file)
    file.seek(0)
    idfVec.append(getTfIdfList(file,i, resTf))
    idf2Vec.append(getTfDdf2List(file, i, resTf))
    file.seek(0)

my_vectorFile = open("output/vectors.txt", "r")
counter = 0

content = my_vectorFile.read() 
columns = content.split("\n")
for i in columns:
    if i:
        counter += 1

my_vectorFile.seek(0)
allLines=my_vectorFile.readlines()

n=sorted(task2.getTerms(directory))

tfVec = []
for i in n:
    if i in mydict:
        tfVec.append(mydict[i])
    else:
        tfVec.append(0.0)





idfVec = [float(i) for i in idfVec]
idf2Vec = [float(i) for i in idf2Vec]

distances = []
for i in range(counter):
    tF = getCells(i,0,my_vectorFile)
    iDf = getCells(i,1,my_vectorFile)
    iDf2 = getCells(i,2,my_vectorFile)
    
    tF = [float(i) for i in tF]
    iDf = [float(i) for i in iDf]
    iDf2 = [float(i) for i in iDf2]
    if(col==0):
        dist = scipy.spatial.distance.euclidean(tF, tfVec)
        distances.append(dist)
    if(col==1):
        dist = scipy.spatial.distance.euclidean(iDf, idfVec)
        distances.append(dist)
    if(col==2):
        dist = scipy.spatial.distance.euclidean(iDf2, idf2Vec)
        distances.append(dist)

top10_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:10]
nearestNeighbors = [i for i in top10_indices]
print(nearestNeighbors)