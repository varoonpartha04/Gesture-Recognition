from os import listdir
import math
import sys
import re


file=open("output/1.wrd","r")
def findWrdFilenames( path_to_dir, suffix=".wrd" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

def getAllBands(file):
    mySet = set()
    for i in file:
        j = i.split(" ")
        band = j[3].strip()
        mySet.add(band)
    return mySet


def getTotalSensors(file):
    sensornumber_set=set()
    for i in file:
        j=i.split(" ")
        id=j[1].strip()
        sensornumber_set.add(id)
    return len(sensornumber_set)

def getTermSensors(file,term):
    term_sensor_set = set()
    for i in file:
        j = i.split(" ")
        sensor_id = j[1]
        sensor_id = sensor_id.strip()
        current_term = j[3]
        current_term = current_term.strip()
        if(term == current_term):
            term_sensor_set.add(sensor_id)
    return len(term_sensor_set)
def getTF(file):
    file.seek(0)
    mydict = dict()
    for i in file:
        j = i.split(" ")
        term = j[3].strip()
        
        if term in mydict:
            mydict[term]=mydict.get(term)+1
        else:
            mydict[term]= 1
    sum_tf = 0
    for i in mydict:
        sum_tf += mydict[i]
    for i in mydict:
        mydict[i] = mydict[i]/sum_tf
    return mydict




def getIDF(totalSensors, termSensors):

    if(termSensors == 0):
        return 0.0
    else:
        return math.log(totalSensors/termSensors)

def getTfIdf(tf, idf, maximumtf):
    return (0.5+ 0.5*tf/maximumtf)*idf

def getTfIdf2(tf, idf2, maximumtf):
    return (0.5+ 0.5*tf/maximumtf)*idf2


def getTerms(directory):
    myset=set()
    filesList=findWrdFilenames(directory)
    for file in filesList:
            openFile = open(directory+file, "r")
            result = getAllBands(openFile)
            myset = myset.union(result)
    return myset



def get_maximumtf(i, file):
    mydict = dict()
    totalcount = 0
    for line in file:
        k = line.split(" ")
        sensor_id = k[1].strip()
        value = k[3].strip()
        if(sensor_id == i and value in mydict):
            mydict[value] += 1
            totalcount+=1
        if (sensor_id == i and value not in mydict):
            mydict[value] = 1
            totalcount+=1
    return max(mydict.values())/totalcount

def getTotalFiles(directory):
    return len(findWrdFilenames(directory))

# print(getTotalFiles("output/"))

def getIdfValues(file, bands):
    result = []
    for band in bands:
        file.seek(0)
        no_of_sensors = getTotalSensors(file)
        file.seek(0)
        termSensors = getTermSensors(file, band)
        result.append(getIDF(no_of_sensors, termSensors))

def getTermFilesCount(directory, term):

    filesList = findWrdFilenames(directory)
    count=0
    for file in filesList:
            openFile = open(directory+file, "r")
            for i in openFile:
                j = i.split(" ")
                current_term = j[3].strip()
                if(current_term == term):
                    count+=1
                    break
    return count

# ---------------------------------------------------------
def writeToOutput(sensors, file,bands, outputfile, directory):
    file.seek(0)
    tf_vec = []
    idf_vec = []
    idf2_vec = []
    dict_TF = getTF(file)
    tf_vec = []
    idf1_vec = []
    idf2_vec = []
    for i in bands:
        file.seek(0)
        res_tf = dict_TF.get(i,0)

        tf_vec.append(res_tf)
        file.seek(0)
        j=getTermSensors(file, i)
        res_idf = getIDF(sensors, j)
        temp_maxtf = float(max(dict_TF, key=dict_TF.get))
        res_tfidf = getTfIdf(res_tf,res_idf, temp_maxtf)
        idf_vec.append(res_tfidf)
        file.seek(0)

        m=getTotalFiles(directory)
        n=getTermFilesCount(directory, i)
        res_idf2 = getIDF(m,n )
        res_maxtf2 = float(max(dict_TF, key=dict_TF.get))
        res_tfidf2 = getTfIdf2(res_tf, res_idf2, res_maxtf2)
        idf2_vec.append(res_tfidf2)
    a = str(", ".join(repr(e) for e in tf_vec))
    b = str(", ".join(repr(e) for e in idf1_vec))
    c = str(", ".join(repr(e) for e in idf2_vec))
    outputfile.write("< "+a +" >"+"< "+b+" >"+"< "+c +" >")
    outputfile.write("\n")
    file.close()

def writeAll(file, outputfile, directory):
    bands = getTerms(directory)
    file.seek(0)
    totalSensors = getTotalSensors(file)
    file.seek(0)
    writeToOutput(totalSensors, file, bands, outputfile, directory)
    return None


def tryint(s):
    try:
        return int(s)
    except:
        return s

def alnumkey(s):
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(files):
    files.sort(key=alnumkey)





def main(directory):
    files = findWrdFilenames(directory)
    sort_nicely(files)
    open(directory+'vectors.txt', 'w').close()
    outputfile = open(directory+"vectors.txt", "a")
    for file in files:
        openFile = open(directory+file, "r")
        writeAll(openFile, outputfile, directory)
    outputfile.close()

if __name__ == "__main__":
    directory = sys.argv[1]
    print("Wait")
    main(directory)
    print("Done")
