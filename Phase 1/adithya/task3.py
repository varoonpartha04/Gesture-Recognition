import seaborn as sns
sns.set_theme()
import numpy as np; np.random.seed(0)
import matplotlib.pyplot as plt

import re
import sys
import task2
import task2
import pandas as pd

def heatmap(data):
    # uniform_data = np.random.rand(10, 12)
    cmap=sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0)
    ax = sns.heatmap(data, xticklabels=True, yticklabels=True,cmap=cmap)
    # plt.gray()
    # plt.style.use('grayscale')
    plt.show()

def get_tf_getures(Openedfile):
    output  = []
    for i in Openedfile:
        if(i != "\n"):
        # print(i)
        # a = i[i.find(char1)+1 : i.find(char2)]
            a = re.findall(r'\<(.*?)\>', i)
            k = a[0]
            arr = k.split(',')
            # print(arr)
            arr = np.array(arr).astype(np.float)
            output.append(arr)
    return output

def get_tfidf1_getures(Openedfile):
    output  = []
    for i in Openedfile:
        # print(i)
        # a = i[i.find(char1)+1 : i.find(char2)]
        if(i != "\n"):
            a = re.findall(r'\<(.*?)\>', i)
            k = a[1]
            arr = k.split(',')
            # print(arr)
            arr = np.array(arr).astype(np.float)
            output.append(arr)
    return output
def get_tfidf2_getures(Openedfile):
    output  = []
    for i in Openedfile:
        # print(i)
        # a = i[i.find(char1)+1 : i.find(char2)]
        if(i != "\n"):
            a = re.findall(r'\<(.*?)\>', i)
            k = a[2]
            arr = k.split(',')
            # print(arr)
            arr = np.array(arr).astype(np.float)
            output.append(arr)
    return output

def get_termfreqs(sens_id, elem, Openedfile):
    Openedfile.seek(0)
    map1 = {}
    for i in Openedfile:
        k = i.split(" ")
        term = k[3]
        term = term.strip()
        map1[term]+= map1.get(term,0)
    return map1

def getcells(filenum, col, vectorfile):
    Openedfile = open("output/vectors.txt", "r")
    lines=Openedfile.readlines()
    line = lines[filenum]
    # print(filenum)
    # line = Openedfile.seek(filenum)
    # print("line is")
    # print(line)
    a = re.findall(r'\<(.*?)\>', line)
    k = a[col]
    # print(k)
    b = k.split(",")
    ans = [string.strip() for string in b]
    return ans

def heatmap_matrix_tf(gesture, filenum, vectorfile):
    # rows = task2.find_no_of_sensors(gesture)
    # gesture.seek(0)
    # cols = max(task2.find_all_levels(gesture))
    # cols = int(cols)
    gesture.seek(0)
    matrix = pd.DataFrame()
    num_lines = 0
    for k in gesture: num_lines+=1
    gesture.seek(0)
    map1 = task2.get_termfreqs(gesture)
    # print("map1 is")
    # print(map1)
    gesture.seek(0)
    for p in range(num_lines):
        # print(p)
        gesture.seek(0)
        k = gesture.readlines()[p]
        i = k.split(" ")
        sensor = int(i[1].strip())
        time = int(i[2].strip())
        word = i[3].strip()
        # tfval = task2.termfreq(str(sensor), str(word), gesture)
        # matrix[sensor-1][word-1] = tfval
        tfval = map1.get(word)
        col = 0 # for tf
        tfvals = getcells(filenum, col, vectorfile)
        # print(tfvals)
        # print(type(tfvals))
        map2 = {}        
        n=0
        for i in map1.keys():
            map2[i] = tfvals[n]
            n+=1
        # print("map2 is")
        # print(map2)
        # break
        tfval = map2[word]
        # print(type(tfval))
        matrix.loc[sensor,time] = float(tfval)
    print(matrix)
    return matrix

def heatmap_matrix_tfidf(gesture, filenum, vectorfile):
    # rows = task2.find_no_of_sensors(gesture)
    # gesture.seek(0)
    # cols = max(task2.find_all_levels(gesture))
    # cols = int(cols)
    gesture.seek(0)
    matrix = pd.DataFrame()
    num_lines = 0
    for k in gesture: num_lines+=1
    gesture.seek(0)
    map1 = task2.get_termfreqs(gesture)
    # print(map1)
    gesture.seek(0)
    for p in range(num_lines):
        # print(p)
        gesture.seek(0)
        k = gesture.readlines()[p]
        i = k.split(" ")
        sensor = int(i[1].strip())
        time = int(i[2].strip())
        word = i[3].strip()
        # tfval = task2.termfreq(str(sensor), str(word), gesture)
        # matrix[sensor-1][word-1] = tfval
        tfval = map1.get(word)
        col = 1 # for tfidf
        tfvals = getcells(filenum, col, vectorfile)
        # print(tfvals)
        # print(type(tfvals))
        map2 = {}        
        n=0
        for i in map1.keys():
            map2[i] = tfvals[n]
            n+=1
        # print(map2)
        tfval = map2[word]
        # print(type(tfval))
        matrix.loc[sensor,time] = float(tfval)
    print(matrix)
    return matrix


def heatmap_matrix_tfidf2(gesture, filenum, vectorfile):
    # rows = task2.find_no_of_sensors(gesture)
    # gesture.seek(0)
    # cols = max(task2.find_all_levels(gesture))
    # cols = int(cols)
    gesture.seek(0)
    matrix = pd.DataFrame()
    num_lines = 0
    for k in gesture: num_lines+=1
    gesture.seek(0)
    map1 = task2.get_termfreqs(gesture)
    # print(map1)
    gesture.seek(0)
    for p in range(num_lines):
        # print(p)
        gesture.seek(0)
        k = gesture.readlines()[p]
        i = k.split(" ")
        sensor = int(i[1].strip())
        time = int(i[2].strip())
        word = i[3].strip()
        # tfval = task2.termfreq(str(sensor), str(word), gesture)
        # matrix[sensor-1][word-1] = tfval
        tfval = map1.get(word)
        col = 2 # for tfidf2
        tfvals = getcells(filenum, col, vectorfile)
        # print(tfvals)
        # print(type(tfvals))
        map2 = {}        
        n=0
        for i in map1.keys():
            map2[i] = tfvals[n]
            n+=1
        # print(map2)
        tfval = map2[word]
        # print(type(tfval))
        matrix.loc[sensor,time] = float(tfval)
    print(matrix)
    return matrix



def generate_heatmap(gesture_file, values):
    # You will indeed have a 2D matrix of grayscale intensity -- 
    # rows will correspond to the 20 sensors; however the columns 
    # will correspond to the words identified on that sensor in time order.
     
    # change this, this should work on wrd files instead of vector files   
    # Openedfile = open("output/vectors.txt", "r")
    Openedfile = open("output/"+gesture_file,"r")
    vectorfile = open("output/vectors.txt", "r")
    filenum = int(gesture_file.replace('.wrd',''))
    if(values == 1):
        # ans1 = get_tf_getures(Openedfile)
        ans1 = heatmap_matrix_tf(Openedfile, filenum, vectorfile)
        print(ans1)
        heatmap(ans1)
    elif(values == 2):
        # ans2 = get_tfidf1_getures(Openedfile)
        ans2 = heatmap_matrix_tfidf(Openedfile, filenum, vectorfile) 
        heatmap(ans2)
    elif(values == 3):
        # ans3 = get_tfidf2_getures(Openedfile)
        ans3 = heatmap_matrix_tfidf2(Openedfile, filenum, vectorfile)
        heatmap(ans3)
# heatmap()

# we need to use both clustering and scaling so that heatmap looks good
# z score scaling - subtract with mean and divide by standard deviation


if __name__ == "__main__":
    gesture_file = sys.argv[1]
    values = int(sys.argv[2])
    generate_heatmap(gesture_file, values)

# python task3.py 1.wrd 1