from os import listdir
import pandas as pd
from scipy.stats import norm
import numpy as np
import itertools
import math
import sys






# https://stackoverflow.com/questions/9234560/find-all-csv-files-in-a-directory-using-python
def findCsvFilenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]




# https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
def normalize(file):
    minimum = file.min(axis=1)
    maximum = file.max(axis=1)
    normalized = file.sub(minimum, axis='rows')
    normalized = (normalized.div(maximum-minimum, axis='rows')*2)-1
    return normalized




# -------------------------------------------------------------------------
def get_band_length(i, r):
    '''
    function to return the length of  band in float
    '''
    upper_limit = (i-r)/r
    lower_limit = (i-r-1)/r
    numerator = (norm(0, 0.25).cdf(upper_limit))-(norm(0, 0.25).cdf(lower_limit))
    denominator=(norm(0, 0.25).cdf(1))-(norm(0, 0.25).cdf(-1))
    return (numerator/denominator)

def get_length_list(r):
    '''
    funtion to return all band lengths
    '''
    length_list = []
    for i in range(1, 2*r+1):
        length_list.append(get_band_length(i, r))
    return length_list



def getBins(length_list):
    '''
    function to return a list of bins in between -1 & +1
    '''
    bins_list = []
    sums = -1
    for i in length_list:
        sums =sums+ 2*i
        bins_list.append(sums)
    return bins_list




def quantizeData(normalized, bins_list):
    '''
    function to return binning of csv files
    '''

    return pd.DataFrame(np.digitize(normalized, bins=bins_list))


# ----------------------------------------------------------
def rollingOneRow(s, w, data, i):
    result=[]
    start = 0
    while (start+w<data.size):
        res=data.iloc[i][start:start+w]
        start=start+s
        mystr=""
        for _i,v in res.items():
            mystr+=str(v)
        if (len(mystr)>0 and (len(mystr)<w)):
            result.append("nan")
        elif (len(mystr)==w):
            result.append(mystr)
    return result


def rollingAllRows(data):
    result = []
    for i in range(len(data)):
        result.append(rollingOneRow(data=data, w=3, s=2, i=i))
    result = np.array(result).ravel()
    return result


def rollingWindowGenerator(shift, n):
    ans = [i for i in range(0, n, shift)]
    return ans


# ----------------------------------------

def getChain(data):
    return list(itertools.chain(*data))
def getSensorId(data, s):
    rows = data.shape[0]
    _columns = math.ceil(data.shape[1]/s)
    sensorId = [[y]*_columns for y in range(0, rows)]
    sensorId=getChain(sensorId)
    return sensorId



# ------------------------------
def processFile(file, csv, s, r, w):
    normalized=normalize(csv)
    length_list=get_length_list(r)
    bins_list=(getBins(length_list))
    bins_list.insert(0,-1.1)
    bins_list[-1]=1.1
    quantized=quantizeData(normalized,bins_list)
    rollingWindow=np.array(rollingAllRows(quantized)).ravel()
    rows=csv.shape[0]
    n=normalized.shape[1]
    first = (rollingWindowGenerator(s, n))*rows
    files = [file]*len(first)
    sensors = getSensorId(csv, s)
    fName = file.replace('.csv', '')
    f = open("output/"+fName+".wrd", "w")
    for i in range(len(files)):
        if(not rollingWindow[i] == "nan"):
            f.write(str(files[i]) + " ")
            f.write(str(sensors[i]+1)+" ")
            f.write(str(first[i])+" ")
            f.write(str(rollingWindow[i]))
            f.write("\n")
    f.close()

def main(directory,s,w,r):
    filesList=findCsvFilenames(directory)
    for file in filesList:
        csv=pd.read_csv(directory+file, header=None)
        processFile(file,csv,s,r,w)

if __name__ == "__main__":
    directory = sys.argv[1]
    s = int(sys.argv[2])
    w = int(sys.argv[3])
    r = int(sys.argv[4])
    main(directory, s, w, r)
 
