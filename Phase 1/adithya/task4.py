from sklearn.neighbors import NearestNeighbors
import numpy as np
import re
import task2
import math
import sys
import scipy
if __name__ == "__main__":
    gesture_file = sys.argv[1]
    col = int(sys.argv[2])


Openedfile = open(gesture_file, "r")

# def get_tf_getures(Openedfile):
#     output  = []
#     for i in Openedfile:
#         if(i != "\n"):
#         # print(i)
#         # a = i[i.find(char1)+1 : i.find(char2)]
#             a = re.findall(r'\<(.*?)\>', i)
#             k = a[0]
#             arr = k.split(',')
#             # print(arr)
#             arr = np.array(arr).astype(np.float)
#             output.append(arr)
#     return output

def getcells(filenum, col, vectorfile):
    Openedfile = open("output/vectors.txt", "r")
    lines=Openedfile.readlines()
    line = lines[filenum-1]
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

def idf(no_of_sensors, sensors_with_term):
    if(sensors_with_term == 0): return 0.0
    return math.log(no_of_sensors/sensors_with_term)


# nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)

# Openedfile = open("ex1.wrd", "r")
directory = "../codes/output/"

slots = task2.find_all_terms_inDir(directory)
# print(slots)

tfvals = task2.get_termfreqs(Openedfile)
# print(tfvals)

def tfidf(tf, idf1, maxtf):
    return (0.5+ 0.5*tf/maxtf)*idf1

levels = task2.find_all_terms_inDir(directory)
levels = sorted(levels)
idf1_vec = []
idf2_vec = []
Openedfile.seek(0)
tf_map = task2.get_termfreqs(Openedfile)
# print(tf_map)
# print("the levels are")

# print(levels)


def get_tfidf_arr(Openedfile,j, temp_tf):
    Openedfile.seek(0)
    # print(j)
    sensors_with_term = task2.find_sensors_with_term(Openedfile, j)
    Openedfile.seek(0)
    # print(sensors_with_term)
    temp_idf = idf(no_of_sensors, sensors_with_term)
    temp_maxtf1 = float(max(tf_map, key=tf_map.get))
    # print(temp_idf)
    temp_tfidf1 = tfidf(temp_tf,temp_idf, temp_maxtf1)
    # print(temp_tfidf1)
    return temp_tfidf1
    # idf1_vec.append(temp_tfidf1)
    # return idf1_vec

def get_tfidf2_arr(Openedfile, j, temp_tf):
    Openedfile.seek(0)
    # sensors_with_term2 = task2_ver2.find_sensors_with_term(Openedfile, j)
    # print("num senors term2 is ")
    # print(sensors_with_term2)
    temp_idf2 = idf(task2.find_no_of_files(directory), task2.find_files_with_term(directory, j))
    temp_maxtf2 = float(max(tf_map, key=tf_map.get))
    temp_tfidf2 = task2.tfidf2(temp_tf, temp_idf2, temp_maxtf2)
    return temp_tfidf2
    # idf2_vec.append(temp_tfidf2)
    # return idf2_vec

for j in levels:
    temp_tf = tf_map.get(j,0)
    # calculating the tf vector
    Openedfile.seek(0)
    no_of_sensors = task2.find_no_of_sensors(Openedfile)
    # calculating the tfidf vector
    Openedfile.seek(0)
    idf1_vec.append(get_tfidf_arr(Openedfile,j, temp_tf))
    #calculating the tfidf2 vector
    idf2_vec.append(get_tfidf2_arr(Openedfile, j, temp_tf))
    Openedfile.seek(0)


# print("idf_vec is")
# print(idf1_vec)
# print(idf2_vec)



#find no of lines in vector.txt
Openedfile_vec = open("output/vectors.txt", "r")
counter = 0
# Reading from file 
Content = Openedfile_vec.read() 
CoList = Content.split("\n")
for i in CoList:
    if i:
        counter += 1
print(counter)

Openedfile_vec.seek(0)
lines = Openedfile_vec.readlines()

# print(tf_map)
q = task2.find_all_terms_inDir(directory)
q = sorted(q)
# print("sorted terms is")
# print(q)
# print(q)
# print("q is")
# print(len(q))
big_tfvec = []
for i in q:
    if i in tf_map:
        # print(i)
        # big_tfmap[i] = tf_map[i]
        big_tfvec.append(tf_map[i])
    else:
        # print(i)
        # big_tfmap[i] = 0.0
        big_tfvec.append(0.0)
# print("big_tfvec is")
# print(big_tfvec)
# print(big_tfvec)
# tf_vec = sorted(tf_map)
# tf_vec = [float(i) for i in tf_vec]
idf1_vec = [float(i) for i in idf1_vec]
idf2_vec = [float(i) for i in idf2_vec]

# print(tf_vec)
# print("big vector is ")
# print(big_tfvec)
distances = []
for i in range(counter):
    tf_stored = getcells(i,0,Openedfile_vec)
    idf1_stored = getcells(i,1,Openedfile_vec)
    idf2_stored = getcells(i,2,Openedfile_vec)
    tf_stored = [float(i) for i in tf_stored]
    idf1_stored = [float(i) for i in idf1_stored]
    idf2_stored = [float(i) for i in idf2_stored]
    if(col==0):
        # print(len(tf_stored))
        # print(len(big_tfvec))
        # if(i==2):
            # print("tfstored is")
            # print(tf_stored)
        dist = scipy.spatial.distance.euclidean(tf_stored, big_tfvec)
        distances.append(dist)
    if(col==1):
        dist = scipy.spatial.distance.euclidean(idf1_stored, idf1_vec)
        distances.append(dist)
    if(col==2):
        dist = scipy.spatial.distance.euclidean(idf2_stored, idf2_vec)
        distances.append(dist)
# print("distances are")
# print(distances)
top10_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:10]
top10 = [i for i in top10_indices]
print(top10)


# tfidf2vals = 

# output = get_tf_getures(Openedfile)
# # print(output)
# output = np.array(output)
# output = output.flatten()
# print(output)

# distances, indices = nbrs.kneighbors(X)

# print(indices)

# python task4.py output/1.wrd 0