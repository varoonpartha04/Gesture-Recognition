from readAndWriteData import readCSV, write_csv_from_dict, write_json
import re
from os import listdir
from os.path import isfile, join
import statistics
from sklearn import preprocessing
import scipy
from scipy.integrate import quad
import os
import math
import pandas as pd
from collections import defaultdict

def read_all_files(dir):
    files = [f for f in listdir(dir) if isfile(join(dir, f)) and re.search('\.csv', f)]
    data = {}
    for _, file in enumerate(files):
        file_path = dir + "/" + file
        key = file[:-4] #Remove the .csv extension from file while keeping it as key
        data[key] = readCSV(file_path)
    return data

def normalize_all_data(data):
    normalized_data = {}
    for k, v in data.items():
        normalized_data[k] = normalize_data(v)
    return normalized_data

def normalize_data(data):
    normalize_function = lambda x, max_x, min_x: (2*(x - min_x)/(max_x - min_x)) - 1
    normalized_value = []
    for row in data:
        max_x = max(row)
        min_x = min(row)
        if max_x == min_x:
            normalized_value.append([0 for x in row])
        else:
            normalized_value.append([normalize_function(x, max_x, min_x) for x in row])
    return normalized_value


def get_band_range(length, x_range = (-1, 1)):
    current_band = x_range[0]
    band_range = []
    for value in length:
        current_band += value
        band_range.append(current_band)
    band_range[-1] = x_range[1]+0.1
    return band_range


def quantize_data(normalized_data, band_range):
    quantized_data = {}
    get_band = lambda x, band_range: min([i+1 for i, r in enumerate(band_range) if x <= r])

    for k, v in normalized_data.items():
        temp_data = []
        for d in v:
            temp_data.append([get_band(x, band_range) for x in d])
        quantized_data[k] = temp_data
    return quantized_data


def get_gausian_band_length(resolution = 3, mean = 0, std = 0.25, x_range = (-1,1)):
    normal_distribution_function = lambda x: scipy.stats.norm.pdf(x,mean,std)
    length, x1, x2 = [], x_range[0], x_range[1]
    denominator, err = quad(normal_distribution_function, x1, x2)

    for i in range(1, 2*resolution+1):
	    x1 = (i-resolution-1)/resolution
	    x2 = (i-resolution)/resolution
	    res, err = quad(normal_distribution_function, x1, x2)
	    length.append(2*res/denominator)
    return length


def create_words(data, window, shift, component_id):
    created_words = {}
    for file_id, file_data in data.items():
        created_words_file = {}
        for sensor_id, sensor_data in enumerate(file_data):
            for index in range(0, len(sensor_data), shift):
                if (index + window) < len(sensor_data):
                    key_for_created_words = (component_id, file_id, sensor_id, index)
                    # key_for_created_words = (component_id, sensor_id)
                    # key_for_created_words = key_for_created_words
                    value_for_created_words = "".join([str(x) for x in sensor_data[index:index+window]])

                    created_words_file[key_for_created_words] = value_for_created_words
        created_words[file_id] = created_words_file
    return created_words

def create_avg(data, window, shift, component_id):
    created_words = {}
    for file_id, file_data in data.items():
        created_words_file = {}
        for sensor_id, sensor_data in enumerate(file_data):
            for index in range(0, len(sensor_data), shift):
                if (index + window) < len(sensor_data):
                    key_for_created_words = (component_id, file_id, sensor_id, index)
                    # key_for_created_words = (file_id, sensor_id)
                    # value_for_created_words = "".join([str(x) for x in sensor_data[index:index+window]])
                    value_for_created_words = statistics.mean(sensor_data[index:index+window])
                    created_words_file[key_for_created_words] = value_for_created_words
        created_words[file_id] = created_words_file
    return created_words

def create_output_directory(output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

def process_component(data,band_range, window_size, split_size, component_id):
    normalized_data = normalize_all_data(data)
    quantized_data = quantize_data(normalized_data, band_range)

    created_words = create_words(quantized_data, window_size, split_size, component_id)
    # created_avgs = create_avg(quantized_data, window_size, split_size)
    created_avgs = create_avg(normalized_data, window_size, split_size, component_id)

    return (created_words, created_avgs)

def save_words(words, output_directory, extension = ".wrd"):
    # Process and save files separately
    file_words_collection = defaultdict(dict)
    for l in words:
        for file_id, words_dict in l.items():
            file_words_collection[file_id].update(words_dict)
    for file_id, words_dict in file_words_collection.items():
        write_csv_from_dict(words_dict, output_directory + "/" + str(file_id) + extension)


def average_and_stdev(data, folder):
    dict_mean = {}
    dict_std = {}
    for file_id, file_data in data.items():
        # print(file_id)
        file_data = pd.DataFrame(file_data)
        file_data = pd.DataFrame(file_data)
        mean_df = file_data.mean(axis=1).to_frame()
        std_df = file_data.std(axis=1).to_frame()
        l = mean_df.shape[0]
        b = 1
        mean_df2 = mean_df.assign(newind= [str((folder,file_id,str(i))) for i in range(1,l+1)])
        mean_df2 = mean_df2.set_index("newind")
        mean_df2.columns = [str(file_id)]
        std_df2 = std_df.assign(newind= [str((folder,file_id,str(i))) for i in range(1,l+1)])
        std_df2 = std_df2.set_index("newind")
        std_df2.columns = [str(file_id)]
        temp_mean_dict = mean_df2.to_dict()
        temp_std_dict = std_df2.to_dict()
        dict_mean.update(temp_mean_dict)
        dict_std.update(temp_std_dict)
    return (dict_mean, dict_std)


def main():
    directory_name = input("Enter the path of directory: ")

    resolution = int(input("Enter resolution: "))
    window_size = int(input("Enter window size: "))
    split_size = int(input("Enter split size: "))
    data_folders = ["X", "Y", "Z", "W"]
    output_directory = "Outputs"
    create_output_directory(output_directory)

    length = get_gausian_band_length(resolution)
    print("Calculating Guassian Band Length")
    band_range = get_band_range(length)
    print("Calculating Guassian Band Range")

    words_collection = []
    average_collection = []
    dict_mean_final = {}
    dict_std_final = {}
    for folder in data_folders:
        print("Processing folder: " + folder)
        data = read_all_files(directory_name + folder + "/")
        average_output, stdev_output = average_and_stdev(data, folder)
        dict_mean_final.update({folder:average_output})
        dict_std_final.update({folder:stdev_output})
        words, average = process_component(data, band_range, window_size, split_size, folder)
        words_collection.append(words)
        average_collection.append(average)
    print("Saving words to file")
    save_words(words_collection, output_directory, ".wrd")
    save_words(average_collection, output_directory, ".awrd")
    write_json(dict_mean_final, output_directory + "/means.json")
    write_json(dict_std_final, output_directory + "/std_dev.json")


main()
