from readAndWriteData import read_features_csv, write_json
from collections import defaultdict
import math
import re
from os import listdir
from os.path import isfile, join

def main():
    directory_name = input("Enter the path of directory: ")
    print("Reading files..")
    features = read_all_files(directory_name)
    features = convert_to_dict(features)
    print("Calculating tf values..")
    tf_value = calculate_tf(features)
    print("Calculating tf idf values..")
    tf_idf_values = calculate_tf_idf(features)
    print("Calculating tf idf 2 values..")
    tf_idf2_values = calculate_tf_idf2(features)
    vector_space = {
        "tf_value": tf_value,
        "tf_idf_value": tf_idf_values, 
        "tf_idf2_values": tf_idf2_values
    }
    print("Saving vector space in the output directory. The file name is vector.txt")
    write_json(vector_space, "Outputs/vector.txt")

def read_all_files(dir):
    files = [f for f in listdir(dir) if isfile(join(dir, f)) and re.search('\.csv', f)]
    data = {}
    for file in files:
        file_path =  "Outputs/" + file[:-4] + ".wrd"
        data[file] = read_features_csv(file_path)
    return data

def convert_to_dict(words_data):
    feature_dict = {}
    for file_id, data in words_data.items():
        for d in data:
            temp = d[0].replace("(", "").replace(")", "").split(",")
            key = tuple([x for x in temp])
            val = d[1]
            feature_dict[key] = val

    return feature_dict

def get_tf_value(frequency_word_file):
    tf_count = {}
    for id, word_count in frequency_word_file.items():
        total_words_in_file = sum(word_count.values())
        frequency = {}
        for word, count in word_count.items():
            key = word
            value = count/total_words_in_file
            frequency[key] = value
        tf_count[id] = frequency
    return tf_count

def get_idf_value_file(frequency_word_file):
    no_of_files = len(frequency_word_file)
    word_in_file_count = defaultdict(int)
    idf_count = {}

    idf_function = lambda x, y: math.log(x/y)

    for file_id, word_count in frequency_word_file.items():
        for word in word_count:
            word_in_file_count[word] += 1
    
    for word, count in word_in_file_count.items():
        key = word
        value = idf_function(no_of_files, count)
        idf_count[key] = value
    return idf_count

def get_tf_idf_value(tf_values, idf_values):
    tf_idf_values = {}

    tf_function = lambda x, max_x: 0.5 + (0.5 * x / max_x) 
    tf_idf_function = lambda x, y: x * y

    for file_id, word_frequency in tf_values.items():
        max_frequency = max(word_frequency.values())
        tf_idf_value_file = {}
        for word, frequency in word_frequency.items():
            left_value = tf_function(frequency, max_frequency)
            tf_idf_value = tf_idf_function(left_value, idf_values[word])
            tf_idf_value_file[word] = tf_idf_value
        
        tf_idf_values[file_id] = tf_idf_value_file
    return tf_idf_values

def calculate_tf(data):
    frequency_word_file = defaultdict(lambda: defaultdict(int))
    for k, v in data.items():
        file_id = k[0]
        word = v
        frequency_word_file[file_id][word] += 1
    return get_tf_value(frequency_word_file)

def calculate_tf_idf(data):
    frequency_word_file = defaultdict(lambda: defaultdict(int))
    for k, v in data.items():
        file_id = k[0]
        word = v
        frequency_word_file[file_id][word] += 1

    tf_values =  get_tf_value(frequency_word_file)
    idf_values = get_idf_value_file(frequency_word_file)
    return get_tf_idf_value(tf_values, idf_values)

def calculate_tf_idf2(data):
    frequency_word_sensor_file = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    tf_idf2_values = {}
    for k, v in data.items():
        file_id = k[0]
        sensor_id = k[1]
        word = v
        frequency_word_sensor_file[file_id][sensor_id][word] += 1

    for file_id, value in frequency_word_sensor_file.items():
        tf_values = get_tf_value(value)
        idf_values = get_idf_value_file(value)
        tf_idf_value = get_tf_idf_value(tf_values, idf_values)

        tf_idf2_values[file_id] = tf_idf_value


    return tf_idf2_values


def represent_data_in_vector(tf_idf_data):
    return

main()