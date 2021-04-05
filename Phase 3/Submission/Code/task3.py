import numpy as np
import random
from scipy.stats import norm
import math
import heapq
import json
import pandas as pd

def read_json(file_name):
    with open(file_name, 'r') as json_file:
        return json.load(json_file)

def get_labelled_data(features_file, labels_file):
    features_json = read_json(features_file)
    features_pd = pd.DataFrame.from_dict(features_json, orient='index')
    labels = pd.read_excel (labels_file, header=None)
    labels = labels.set_index(0)
    labels.index = labels.index.map(str)
    labelled_data = pd.concat([features_pd, labels], axis=1)
    return labelled_data
def sim_graph_from_sim_max(similarity_matrix, gestures_list, num_comp):
    gestures_pd = {"gestureId": gestures_list}
    index = 0
    for row in similarity_matrix:
        gestures_pd[gestures_list[index]] = row
        index += 1
    gestures_dataframe = pd.DataFrame(gestures_pd)
    gestures_dataframe = gestures_dataframe.set_index("gestureId")
    sim_graph = np.empty((0, len(similarity_matrix)))
    # sim_matrix = np.empty((0, len(eucl_dist)))
    for row in similarity_matrix:
        k_largest = np.argsort(-np.array(row))[1:num_comp + 1]
        sim_graph_row = [d if i in k_largest else 0 for i, d in enumerate(row)]
        sim_graph = np.append(sim_graph, np.array([sim_graph_row]), axis=0)
    # print(sim_graph)
    row_sums = sim_graph.sum(axis=1)
    sim_graph = sim_graph / row_sums[:, np.newaxis]
    return sim_graph
	
def LSH(L, k, query_gesture_id, t):
	similarity_dict = read_json("./Outputs/similaritymatrix_pca.json")
	gestures_list =  similarity_dict.keys()
	gest_df = pd.DataFrame(data={"gestures": gestures_list})
	gest_df.to_csv("./gestures_list.csv", sep=',',index=False)
	gestures_list = list(gestures_list)
	similarity_matrix = np.array([[similarity_dict[row][column] for column in (similarity_dict[row])] for row in (similarity_dict)])
	query_gesture = similarity_matrix[gestures_list.index(query_gesture_id)]
	gesture_vectors = np.delete(similarity_matrix,gestures_list.index(query_gesture_id),0)
	gestures_list = np.delete(gestures_list,gestures_list.index(query_gesture_id),0)
	w = 3
	b = np.zeros((L,k))
	for i in range(L):
		for j in range(k):
			b[i][j] = random.uniform(0,w)
			n,d = gesture_vectors.shape

	gaussian = norm()
	LSH_family = np.zeros((L,k,d))
	for i in range(L):
		for j in range(k):
			LSH_family[i][j] = gaussian.rvs(size=d)
	hash_tables = [{}]*L
	for i in range(L):
		for j in range(n):
			temp = []
			x = gesture_vectors[j]
			for m in range(k):
				temp.append(math.floor(np.dot(LSH_family[i][m],x)+b[i][m]/w))
			if str(temp) not in hash_tables[i]:
				hash_tables[i][str(temp)] = []
			hash_tables[i][str(temp)].append(gestures_list[j])
	possible_points = []
	for i in range(L):
		temp = []
		for j in range(k):
			temp.append(math.floor(np.dot(LSH_family[i][j],query_gesture)+b[i][j]/w))
		if str(temp) not in hash_tables[i]:
			possible_points.append('None')
		else:
			possible_points.append(hash_tables[i][str(temp)])
	distances = {}
	c = 0
	alpha = 0
	for i in possible_points:
		if not i=='None':
			c+=1
			for j in i:
				alpha+=1
				distances[str(j)] = np.linalg.norm(gesture_vectors[np.where(gestures_list==j)[0][0]]-query_gesture)
	result = heapq.nsmallest(t,distances)
	return result,c,alpha
# def main():
# 	L = input("Enter Number of layers L")
# 	k = input("Enter number of hashes per layer k")
# 	print(gestures_list)
# 	query_gesture_id = input("Enter query gesture id")
# 	t = input("Enter the value of t")
# 	L = int(L)
# 	k = int(k)
# 	t = int(t)

# 	r,c,a = LSH(L, k,gestures_list, t)
# 	print(r)
# 	print("Number of buckets searched ={}".format(c))
# 	print("Number of gestures considered = {}".format(alpha))
# main()

