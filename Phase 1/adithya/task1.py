import itertools
import pandas as pd
import glob
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import normalize
from scipy.stats import norm
import numpy as np
import math
import sys
import warnings
# this function iterates through all files in a folder

warnings.filterwarnings("ignore")
def get_csv_files(directory):
    only_csv_files = [f for f in listdir(directory) if isfile(
        join(directory, f)) if f.endswith('.csv')]
    return only_csv_files


def normalize_csv(csvfile):
    minrows = csvfile.min(axis=1)
    maxrows = csvfile.max(axis=1)
    csv_opened2 = csvfile.sub(minrows, axis='rows')
    csv_opened3 = csv_opened2.div(maxrows-minrows, axis='rows')
    return csv_opened3*2-1

# long code at https://moonbooks.org/Articles/How-to-integrate-a-normal-distribution-in-python-/


def denominator():
    a = norm(0, 0.25).cdf(-1)
    b = norm(0, 0.25).cdf(1)
    return b-a


def find_lim1(i, r):
    return (i-r)/r


def find_lim2(i, r):
    return (i-r-1)/r


def find_quant_levels(i, r):
    lim1 = find_lim1(i, r)
    lim2 = find_lim2(i, r)
    l1a = norm(0, 0.25).cdf(lim1)
    l1b = norm(0, 0.25).cdf(lim2)
    return l1a-l1b/denominator()


def populate_quant_lengths(r):
    quant_levels = []
    for i in range(1, 2*r+1):
        quant_levels.append(find_quant_levels(i, r))
    return quant_levels


def populate_quant_levels(quant_lengths):
    quant_levels = []
    sums = -1
    for i in quant_lengths:
        sums += 2*i
        quant_levels.append(sums)
    # print(quant_levels)
    return quant_levels

# quantize code
# pd.np.digitize( a.col4, bins = [0.3,0.6,0.9 ]  )


def quantize(normalized, quant_levels):
    ans = np.digitize(normalized, bins=quant_levels, right=True)
    return ans


def rolling_win_gen_starts(shift, n):
    ans = [i for i in range(0, n, shift)]
    return ans


def rolling_win_shift(shift, win, data, i):
    start = 0  # it's the index of your 1st valid value.
    # print(data.iloc[i])
    # res = data.iloc[i].rolling(win).mean()[start::shift]
    # print(data.size)
    output = []
    while(start+win<data.size):
    # if(start+shift+1<data.size):
        res = data.iloc[i][start:start+win]
        start+=shift
        resstr  = ""
        for index, value in res.items():
            resstr+=str(value)
        if(len(resstr) == win):
            output.append(resstr)
        elif(len(resstr) < win and len(resstr)>0):
            output.append("nan")
        
    # res=""
    # for j in range(start, start+shift+1):
    #     # print(i)
    #     print(type(data))
    #     print(type(data.iloc[i,j]))
    #     print(data.iloc[i,j])
    #     res += str(data.iloc[i,j])
    #     break
    # for i in data
    # res = 
    # print("res is")
    # print(res)
    # print("done")

    # print(resstr)
    # print(output)
    # print(len(output))
    return output
    # return np.array(res), start


def rolling_all_rows(data, window, shift):
    output = []
    for i in range(len(data)):
        tempout = rolling_win_shift(data=data, win=window, shift=shift, i=i)
        # print("tempout is")
        # print(tempout)
        output.append(rolling_win_shift(data=data, win=window, shift=shift, i=i))
        
    # output = np.array(output)
    # print("output is")
    # print(output)
    return output


def fill_sensor_id(data, shift):
    #     print(data)
    rows = data.shape[0]
    cols = math.ceil(data.shape[1]/shift)
    # print(cols)
    sensor_id = [[y]*cols for y in range(0, rows)]
#     sensor_id.flatten()
    return sensor_id


def flatten(data):
    return list(itertools.chain(*data))


def process_rolling(rolling):
    # ans = []
    # for i in rolling:
    #     ans.append(i[0].tolist())
    # ans = flatten(ans)
    # return ans
    return rolling


def round_rolling(rolling):
    # ans = []
    # for i in rolling:
    #     if(not math.isnan(i)):
    #         ans.append(math.ceil(i))
    #     else:
    #         ans.append(float("nan"))
    # return ans
    return rolling
# a = fill_sensor_id(csv10)


def process(filename, csv_file, shift, resolution, window_len):
    # normalized the data
    normalized = normalize_csv(csv_file)
    # find the quant lengths
    # print(normalized)
    r = resolution
    quant_lengths = populate_quant_lengths(r)
    # find the quant levels
    quant_levels = populate_quant_levels(quant_lengths)
    quant_levels.insert(0, -1.1)
    quant_levels[-1]=1.1
    # print(quant_levels)
    # quantized the values based on bins
    quantized = quantize(normalized, quant_levels)
    quantized = pd.DataFrame(data=quantized)
    rolling = rolling_all_rows(quantized, window_len, shift)
    rolling = process_rolling(rolling)
    rolling = round_rolling(rolling)
    rolling = np.array(rolling)
    rolling = rolling.ravel()
    #rolling = rolling[0]
#     print(rolling[0])
#     print(rolling)
    # print("no of rows is")
    rows = csv_file.shape[0]
    # print(rows)
    # print("no of cols is")
    cols = csv_file.shape[1]
    # print(cols)
    starts = (rolling_win_gen_starts(shift, n=normalized.shape[1]))*rows
    files = [filename]*len(starts)
    # print(len(rolling))
    # print(len(starts))
    # print(len(files))
    sensor_id = fill_sensor_id(csv_file, shift)
    sensor_id = flatten(sensor_id)
    # print(len(sensor_id))
    # print(sensor_id)
    filename = filename.replace('.csv', '')
    f = open("output/"+filename+".wrd", "w")
    # print(len(files))
    # print(len(sensor_id))
    # print(len(starts))
    # print(len(rolling))
    for i in range(len(files)):
        # print(i)
        # print(rolling[i])
        # if(not math.isnan(rolling[i])):
        if(not rolling[i] == "nan"):
            f.write(str(files[i]) + " ")
            f.write(str(sensor_id[i]+1)+" ")
            f.write(str(starts[i])+" ")
            f.write(str(rolling[i]))
            f.write("\n")
    f.close()

# w -> windows length
# s -> shift
# r -> resolution
# ans = []


def gesture_words(directory, shift, window_len, resolution):
    files = get_csv_files(directory)
    for file in files:
        csv_opened = pd.read_csv(directory+file, header=None)
        process(csv_file=csv_opened, shift=shift, filename=file, resolution=resolution, window_len = window_len)

# directory = "../Z/"
# gesture_words(directory)

if __name__ == "__main__":
    directory = sys.argv[1]
    shift = int(sys.argv[2])
    window_len = int(sys.argv[3])
    resolution = int(sys.argv[4])
    print("wait a few seconds")
    gesture_words(directory, shift, window_len, resolution)
    print("done, check your gesture files now")
# command to run  - python task1.py ../Z/ 2 3 3
