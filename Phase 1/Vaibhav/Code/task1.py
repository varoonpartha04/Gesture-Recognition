from readAndWriteData import readCSV, write_csv_from_dict, write_json
import re
from os import listdir
from os.path import isfile, join
from sklearn import preprocessing
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os

def main():
    directory_name = input("Enter the path of directory: ")
    resolution = int(input("Enter resolution: "))
    window_size = int(input("Enter window size: "))
    split_size = int(input("Enter split size: "))

    print("Reading data..")
    data = read_all_files(directory_name + "/")

    print("Normalizing data..")
    normalized_data = normalize_all_data(data)

    print("Quantizing data..")
    length = get_gausian_band_length(resolution)
    band_range = get_band_range(length)
    quantized_data = quantize_data(normalized_data, band_range)

    print("Creating words..")
    create_output_directory()
    created_words = create_words(quantized_data, window_size, split_size)
    for file_id, words_dict in created_words.items():
        write_csv_from_dict(words_dict, "Outputs/" + str(file_id) + ".wrd")
    print("All the feature words are saved in output directory in the same folder as the application in .wrd format." )

def create_output_directory():
    if not os.path.exists('Outputs'):
        os.makedirs('Outputs')

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
            

def create_words(data, window, shift):
    created_words = {}
    for file_id, file_data in data.items():
        created_words_file = {}
        for sensor_id, sensor_data in enumerate(file_data):
            for index in range(0, len(sensor_data), shift):
                if (index + window) < len(sensor_data):
                    key_for_created_words = (file_id, sensor_id, index)
                    value_for_created_words = "".join([str(x) for x in sensor_data[index:index+window]])
                    
                    created_words_file[key_for_created_words] = value_for_created_words
        created_words[file_id] = created_words_file
    return created_words      


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

def read_all_files(dir):
    files = [f for f in listdir(dir) if isfile(join(dir, f)) and re.search('\.csv', f)]
    data = {}
    for i, file in enumerate(files):
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

main()