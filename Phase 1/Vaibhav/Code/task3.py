import seaborn as sns
import matplotlib.pyplot as plt
from readAndWriteData import read_features_csv, read_json
from collections import defaultdict

def prepare_tf_for_heatmap(file_name, sequence_data, vector_space):
   key_for_vector_space = "tf_value"
   dictionary_for_heatmap = defaultdict(list)
   for data in sequence_data:
      temp = data[0].replace("(", "").replace(")", "").split(",")
      sensor_id = int(temp[1])
      word = data[1]
      tf_value_for_word = vector_space[key_for_vector_space][file_name][word]
      dictionary_for_heatmap[sensor_id].append(tf_value_for_word)
   result_for_heatmap = []
   for value in (dictionary_for_heatmap.values()):
      result_for_heatmap.append(value)
   return result_for_heatmap


def prepare_tf_idf_for_heatmap(file_name, sequence_data, vector_space):
   key_for_vector_space = "tf_idf_value"
   dictionary_for_heatmap = defaultdict(list)
   for data in sequence_data:
      temp = data[0].replace("(", "").replace(")", "").split(",")
      sensor_id = temp[1].strip()
      word = data[1]
      tf_value_for_word = vector_space[key_for_vector_space][file_name][word]
      dictionary_for_heatmap[sensor_id].append(tf_value_for_word)
   result_for_heatmap = []
   for value in (dictionary_for_heatmap.values()):
      result_for_heatmap.append(value)
   return result_for_heatmap

def prepare_tf_idf2_for_heatmap(file_name, sequence_data, vector_space):
   key_for_vector_space = "tf_idf2_values"
   dictionary_for_heatmap = defaultdict(list)
   for data in sequence_data:
      temp = data[0].replace("(", "").replace(")", "").split(",")
      sensor_id = temp[1]
      word = data[1]
      tf_value_for_word = vector_space[key_for_vector_space][file_name][sensor_id][word]
      dictionary_for_heatmap[sensor_id].append(tf_value_for_word)
   result_for_heatmap = []
   for value in (dictionary_for_heatmap.values()):
      result_for_heatmap.append(value)
   return result_for_heatmap

def show_heat_map(data, file_name, type):
   sns.set(font_scale=0.5)
   sns.heatmap(data,cmap=sns.color_palette("Greys"),  annot=False)
   plt.xlabel("Time Series")
   plt.ylabel("Sensors")
   plt.title(file_name + " " + type)
   plt.savefig("Outputs/" + file_name + "_" + type + ".png")


def main():
   sns.set_theme()
   # Read the whole file with path
   file_path = input("Enter the file path: ")
   file_name = file_path.split("/")[-1][:-4] + ".wrd" 
   print("Reading output of task 1 & 2..")
   # Fetch the file name and find it in output folder
   sequence_data = read_features_csv("Outputs/" + file_name)
   file_id =  file_name[:-4]
   file_id_key = "'" + file_name[:-4] + "'"
   vector_space = read_json("Outputs/vector.txt")
   tf_data_heatmap = prepare_tf_for_heatmap(file_id_key, sequence_data, vector_space)
   tf_idf_data_heatmap = prepare_tf_idf_for_heatmap(file_id_key, sequence_data, vector_space)
   tf_idf2_data_heatmap = prepare_tf_idf2_for_heatmap(file_id_key, sequence_data, vector_space)
   print("Heat maps can be found in output directory.")
   show_heat_map(tf_data_heatmap, file_id, "tf")
   show_heat_map(tf_idf_data_heatmap, file_id, "tf_idf")
   show_heat_map(tf_idf2_data_heatmap, file_id, "tf_idf2")

 
main()
