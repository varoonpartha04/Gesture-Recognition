import os
import math
import pickle
from scipy.spatial import distance
import argparse
import numpy as np
import scipy
from sklearn.cluster import KMeans
import random
from readAndWriteData import readCSV, write_csv_from_dict, write_json, read_json, read_features_csv

def cluster(gesture_contributions):
    latent_groupings = {}
    for key,value in gesture_contributions.items():
        group_name = max(value, key=lambda key: value[key])
        if group_name not in latent_groupings:
            latent_groupings[group_name] = []
        latent_groupings[group_name].append(key)

    latent_groupings = {k: v for k, v in sorted(latent_groupings.items(), key=lambda item: item[0])}
    for latent_name,value in latent_groupings.items():
        print(latent_name, ": ", value)

def process_input(vector_model):
    if vector_model == "1":
        source_path = 'Outputs/svd_transformed_similarity.json'
        gesture_contributions = read_json(source_path) 
        cluster(gesture_contributions)
    elif vector_model == "2":
        source_path = 'Outputs/nmf_transformed_similarity.json'
        gesture_contributions = read_json(source_path) 
        cluster(gesture_contributions)
    else:
        print("Please provide the correct input")

def main():
    print("Choose one of the vector model from the following: \n1. SVD \n2. NMF")
    vector_model = input("Choose 1 - 2: ")
    process_input(vector_model)

main()
