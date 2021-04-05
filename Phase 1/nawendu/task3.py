import re
import task2
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import pandas as pd
import sys

def greyscalePlotHeatmap(file):
    cmap=sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0)
    _axes = sns.heatmap(file,cmap=cmap)
    plt.show()
def getCells(filenumber, column):
    vectorFile = open("output/vectors.txt", "r") 
    lines=vectorFile.readlines() 
    line = lines[filenumber]
    a = re.findall(r'\<(.*?)\>', line)
    # print(a)
    i = a[column]  
    j = i.split(",")
    res = [string.strip() for string in j]
    return res


    # -------------------------------------------------------------------------------------------
def get_tf(file):
    output  = []
    for i in file:
        if(i != "\n"):
            a = re.findall(r'\<(.*?)\>', i)
            k = a[0]
            arr = k.split(',')
            arr = np.array(arr).astype(np.float)
            output.append(arr)
    return output

def get_tfidf(file):
    output  = []
    for i in file:
        if(i != "\n"):
            a = re.findall(r'\<(.*?)\>', i)
            k = a[1]
            arr = k.split(',')
            arr = np.array(arr).astype(np.float)
            output.append(arr)
    return output

def get_tfidf2(file):
    output  = []
    for i in file:
        if(i != "\n"):
            a = re.findall(r'\<(.*?)\>', i)
            k = a[2]
            arr = k.split(',')
            arr = np.array(arr).astype(np.float)
            output.append(arr)
    return output


# -------------------------------------------------------------------------------------------

def getHeatmapTf(wordFile, filenum, vectorfile):
    wordFile.seek(0)
    res = pd.DataFrame()
    lines = 0
    for i in wordFile:
        lines+=1
    
    wordFile.seek(0)
    mydict = task2.getTF(wordFile)
    
    wordFile.seek(0)
    for i in range(lines):
        wordFile.seek(0)
        j = wordFile.readlines()[i]
        k = j.split(" ")
        sensor = int(k[1].strip())
        time = int(k[2].strip())
        word = k[3].strip()
        tfValue = mydict.get(word)
        
        col = 0 
        tfValues = getCells(filenum, col)
        # print(tfValues)
        dict2 = {}        
        n=0
        for i in mydict.keys():
            dict2[i] = tfValues[n]
            n+=1
        tfValue = dict2[word]
        res.loc[sensor,time] = float(tfValue)
    return res


def getHeatmapTfIdf(wordFile, filenum, vectorfile):
    wordFile.seek(0)
    res = pd.DataFrame()
    lines = 0
    for i in wordFile:
        lines+=1
    wordFile.seek(0)
    mydict = task2.getTF(wordFile)
    wordFile.seek(0)
    for i in range(lines):
        wordFile.seek(0)
        j = wordFile.readlines()[i]
        k = j.split(" ")
        sensor = int(k[1].strip())
        time = int(k[2].strip())
        word = k[3].strip()
        tfValue = mydict.get(word)
        col = 1 
        tfValues = getCells(filenum, col)
        dict2 = {}        
        n=0
        for i in mydict.keys():
            dict2[i] = tfValues[n]
            n+=1
        tfValue = dict2[word]
        res.loc[sensor,time] = float(tfValue)
    return res

def getHeatmapTfIdf2(wordFile, filenum, vectorfile):
    wordFile.seek(0)
    res = pd.DataFrame()
    lines = 0
    for i in wordFile:
        lines+=1
    wordFile.seek(0)
    mydict = task2.getTF(wordFile)
    wordFile.seek(0)
    for i in range(lines):
        wordFile.seek(0)
        j = wordFile.readlines()[i]
        k = j.split(" ")
        sensor = int(k[1].strip())
        time = int(k[2].strip())
        word = k[3].strip()
        tfValue = mydict.get(word)
        col = 2 
        tfValues = getCells(filenum, col)
        dict2 = {}        
        n=0
        for i in mydict.keys():
            dict2[i] = tfValues[n]
            n+=1
        tfValue = dict2[word]
        res.loc[sensor,time] = float(tfValue)
    return res


def main(gesture_file, value):
    wordFile = open("output/"+gesture_file,"r")
    vectorfile = open("output/vectors.txt", "r")
    fileNumber = int(gesture_file.replace('.wrd',''))
    if(value == 1):
        result = getHeatmapTf(wordFile, fileNumber, vectorfile)
        greyscalePlotHeatmap(result)
    elif(value == 2):
        result = getHeatmapTfIdf(wordFile, fileNumber, vectorfile) 
        greyscalePlotHeatmap(result)
    elif(value == 3):
        result = getHeatmapTfIdf2(wordFile, fileNumber, vectorfile)
        greyscalePlotHeatmap(result)


if __name__ == "__main__":
    gesture_file = sys.argv[1]
    values = int(sys.argv[2])
    main(gesture_file, values)