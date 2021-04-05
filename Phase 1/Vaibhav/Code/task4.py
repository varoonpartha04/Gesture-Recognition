from readAndWriteData import read_features_csv, read_json
from collections import defaultdict
from scipy.spatial import distance

def minkowski_distance(vector1, vector2, dimensions = 2):
    return distance.minkowski(vector1, vector2, dimensions)

def prepare_vectors(data1, data2):
    vector1 = []
    vector2 = []
    for word, value in data1.items():
        vector1.append(value)
        if word in vector2:
            vector2.append(data2[word])
        else:
            vector2.append(0)
    for word, value in data2.items():
        if word not in data1:
            vector2.append(value)
            vector1.append(0)
    return vector1, vector2

def prepare_vectors_idf2(data1, data2):
    vector1 = []
    vector2 = []
    for sensor, sensor_value1 in data1.items():
        sensor_value2 = data2[sensor]
        vec1, vec2 = prepare_vectors(sensor_value1, sensor_value2)
        vector1.extend(vec1)
        vector2.extend(vec2)
    return vector1, vector2

def get_similar_files_tf(file_id, vector_space, no_of_similar_files = 10):
    key_for_vector_space = "tf_value"
    tf_value_for_given_file = vector_space[key_for_vector_space][file_id]
    distance_from_each_file = {}
    for file, tf_values in vector_space[key_for_vector_space].items():
        if (file != file_id):
            vector1, vector2 = prepare_vectors(tf_values, tf_value_for_given_file)
            distance = minkowski_distance(vector1, vector2)
            distance_from_each_file[file] = distance
    sorted_files = [k for k, v in sorted(distance_from_each_file.items(), key=lambda item: item[1])]
    return sorted_files[:no_of_similar_files] if no_of_similar_files < len(sorted_files) else sorted_files

def get_similar_files_tf_idf(file_id, vector_space, no_of_similar_files = 10):
    key_for_vector_space = "tf_idf_value"
    tf_idf_value_for_given_file = vector_space[key_for_vector_space][file_id]
    distance_from_each_file = {}
    for file, tf_idf_values in vector_space[key_for_vector_space].items():
        if (file != file_id):
            vector1, vector2 = prepare_vectors(tf_idf_values, tf_idf_value_for_given_file)
            distance = minkowski_distance(vector1, vector2)
            distance_from_each_file[file] = distance
    sorted_files = [k for k, v in sorted(distance_from_each_file.items(), key=lambda item: item[1])]
    return sorted_files[:no_of_similar_files] if no_of_similar_files < len(sorted_files) else sorted_files

def get_similar_files_tf_idf2(file_id, vector_space, no_of_similar_files = 10):
    key_for_vector_space = "tf_idf2_values"
    tf_value_for_given_file = vector_space[key_for_vector_space][file_id]
    distance_from_each_file = {}
    for file, tf_values in vector_space[key_for_vector_space].items():
        if (file != file_id):
            vector1, vector2 = prepare_vectors_idf2(tf_values, tf_value_for_given_file)
            distance = minkowski_distance(vector1, vector2)
            distance_from_each_file[file] = distance
    sorted_files = [k for k, v in sorted(distance_from_each_file.items(), key=lambda item: item[1])]
    return sorted_files[:no_of_similar_files] if no_of_similar_files < len(sorted_files) else sorted_files



def main():
    # Take file input
    file_path = input("Enter the file path: ")
    file_name = file_path.split("/")[-1][:-4] + ".wrd"
    file_id_key = "'" + file_name[:-4] + "'"
    vector_space = read_json("Outputs/vector.txt")
    tf_similar_files = get_similar_files_tf(file_id_key, vector_space)
    tf_idf_similar_files = get_similar_files_tf_idf(file_id_key, vector_space)
    tf_idf2_similar_files = get_similar_files_tf_idf2(file_id_key, vector_space)
    print("According to TF Values, similar files are: " + str(tf_similar_files))
    print("According to TF IDF Values, similar files are: " + str(tf_idf_similar_files))
    print("According to TF IDF2 Values, similar files are: " + str(tf_idf2_similar_files))

main()

