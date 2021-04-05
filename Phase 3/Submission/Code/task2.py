from readAndWriteData import read_json;
import numpy as np;
import pandas as pd;
from math import sqrt;
from random import randrange
from task1 import ppr
from readAndWriteData import readCSV

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate Gini index 
def gini_index(nodes, classes):
	count_instances = float(sum([len(node) for node in nodes]))
	val_gini = 0.0
	for node in nodes:
		node_size = float(len(node))
		if node_size == 0:
			continue
		score = 0.0
		for each_class in classes:
			p = [row[-1] for row in node].count(each_class)/node_size
			score += p*p
		val_gini += (1.0-score) * (node_size/count_instances)
	return val_gini

# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	ind_b, val_b, score_b, groups_b = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			val_gini = gini_index(groups, class_values)
			if val_gini < score_b:
				ind_b, val_b, score_b, groups_b = index, row[index], val_gini, groups
	return {'index':ind_b, 'value':val_b, 'groups':groups_b}

def to_terminal(node):
	possible_answers = [row[-1] for row in node]
	res = max(set(possible_answers), key=possible_answers.count)
	return res

# Create child splits for a node 
def split(node, max_depth, min_size, depth):
	l, r = node['groups']
	del(node['groups'])
	if not l or not r:
		node['left'] = node['right'] = to_terminal(l + r)
		return
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(l), to_terminal(r)
		return
	if len(l) <= min_size: node['left'] = to_terminal(l)
	else:
		node['left'] = get_split(l)
		split(node['left'], max_depth, min_size, depth+1)
	if len(r) <= min_size: node['right'] = to_terminal(r)
	else:
		node['right'] = get_split(r)
		split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, depth_maximum, size_minimum):
	root_node = get_split(train)
	split(root_node, depth_maximum, size_minimum, 1)
	return root_node

# decision tree prediction code
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	pred_result = list()
	for gesture in test:
		pred = predict(tree, gesture)
		pred_result.append(pred)
	return(pred_result)

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)
 
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
 
# Make a prediction with neighbors
def predict_knn_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
 
# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict_knn_classification(train, row, num_neighbors)
		predictions.append(output)
	return(predictions)

def get_labelled_data(features_file, labels_file):
    features_json = read_json(features_file)
    features_pd = pd.DataFrame.from_dict(features_json, orient='index')
    labels = pd.read_excel (labels_file, header=None)
    labels = labels.set_index(0)
    labels.index = labels.index.map(str)
    labelled_data = pd.concat([features_pd, labels], axis=1)
    return labelled_data

def get_train_test_data(df):
    # Shuffle your dataset 
    shuffle_df = df.sample(frac=1)
    # Define a size for your train set 
    train_size = int(0.7 * len(df))
    # Split your dataset 
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]
    return train_set, test_set

def get_features_labels_seperately(train_set, test_set):
    train_X = train_set.iloc[:, :-1].to_numpy()
    train_y = train_set.iloc[:, -1].to_numpy()
    test_X = test_set.iloc[:, :-1].to_numpy()
    test_y = test_set.iloc[:, -1].to_numpy()
    return train_X, train_y, test_X, test_y

def accuracy(list1, list2):
    count = 0
    for i in range(len(list1)):
        if(list1[i]==list2[i]):
            count+=1
    return count/len(list1)

def most_freq_label(labels_list):
	counter = {}
	max_val = 0
	max_elem = ""
	for i in labels_list:
		if i in counter: counter[i]+=1
		else: counter[i] = 1
	for i in labels_list:
		if(counter[i]>max_val):
			max_val = counter[i]
			max_elem =i 
	return max_elem

def ppr_based_classifer(query_gestures, labels_dict):
	gestures_list = pd.read_csv("gestures_list.csv")
	similarity_graph = pd.read_csv("similarity_graph.csv", header=None)
	similarity_graph = similarity_graph.values
	gestures_list = gestures_list.values.flatten()
	ppr_values = ppr(similarity_graph, gestures_list, query_gestures, max_iterations=500, alpha=0.85)
	ppr_values = ppr_values[:10]
	near_gestures = [ppr_val['gestureId'] for ppr_val in ppr_values]
	near_labels = [labels_dict[str(near_gesture)][1] for near_gesture in near_gestures]
	# print(near_gestures)
	# print(near_labels)
	# print(most_freq_label(near_labels))
	return most_freq_label(near_labels)


def get_labels_dict(filename):
	only_labels = pd.read_excel (filename, header=None)
	only_labels[0] = only_labels[0].astype(str)
	only_labels = only_labels.set_index(0)
	only_labels_dict = only_labels.to_dict('index')
	return only_labels_dict
	# print(only_labels_dict["3"][1])


def main():
	k = int(input("Type 0 to use test set, type 1 to use your input: "))
	labelled_data = get_labelled_data("Outputs/pca_transformed.json", "all_labels.xlsx")
	print(labelled_data)
	# print(labelled_data.loc['1'].tolist()[:-1])
	train_set, test_set = get_train_test_data(labelled_data)
	train_set_np = train_set.to_numpy()
	test_set_np = test_set.to_numpy()
	labels_dict = get_labels_dict("all_labels.xlsx")
	# print(labels_dict)
	train_X, train_y, test_X, test_y = get_features_labels_seperately(train_set, test_set)
	# print(type(train_X))

	#is user chooses his own test set, change the test set
	if( k == 1):
		n = int(input("type the no of gestures you want the predictions for: "))
		test_X = []
		test_y = []
		for i in range(n):
			temp = input("enter next gestures: ")
			test_X.append(labelled_data.loc[temp].tolist()[:-1])
			test_y.append(labelled_data.loc[temp].tolist()[-1])
	# print(test_y)
	# print("test_X is")
	# print(test_X)
	knn_predictions = []
	# print(test_X[1])
	for i, row in enumerate(test_X):
		knn_prediction = predict_knn_classification(train_set_np, test_X[i], 5)
		knn_predictions.append(knn_prediction)
	decision_tree_predictions = decision_tree(train_set_np, test_X, 5, 10)
	ppr_predictions = []
	for i, row in enumerate(test_X):
		ppr_prediction = ppr_based_classifer([i], labels_dict)
		ppr_predictions.append(ppr_prediction)
	print("knn prediction")
	print(knn_predictions)
	print("decision tree predictions")
	print(decision_tree_predictions)
	print("ppr prediction")
	print(ppr_predictions)
	# ppr_based_classifer(["1","2"], labels_dict)
	n_folds = 5
	max_depth = 5
	min_size = 10
	print("actual labels are: ")
	print(test_y)
	print("knn accuracy is: " + str(accuracy(knn_predictions, test_y)))
	print("decision tree accuracy is: " + str(accuracy(decision_tree_predictions, test_y)))
	print("ppr based classifier accuracy is: " + str(accuracy(ppr_predictions, test_y)))


# this funcition is to be used by ui function in task 6
def task2_main(k=1, query_list=["1"]):
	# k = int(input("Type 0 to use test set, type 1 to use your input: "))
	labelled_data = get_labelled_data("Outputs/pca_transformed.json", "all_labels.xlsx")
	# print(labelled_data.loc['1'].tolist()[:-1])
	train_set, test_set = get_train_test_data(labelled_data)
	train_set_np = train_set.to_numpy()
	test_set_np = test_set.to_numpy()
	labels_dict = get_labels_dict("all_labels.xlsx")
	train_X, train_y, test_X, test_y = get_features_labels_seperately(train_set, test_set)
	
	#is user chooses his own test set, change the test set
	if( k == 1):
		# n = int(input("type the no of gestures you want the predictions for: "))
		test_X = []
		test_y = []
		# for i in range(n):
		for temp in query_list:
			# temp = input("enter next gestures: ")
			test_X.append(labelled_data.loc[temp].tolist()[:-1])
			test_y.append(labelled_data.loc[temp].tolist()[-1])
	# print(test_y)
	# print("test_X is")
	# print(test_X)
	knn_predictions = []
	for i, row in enumerate(test_X):
		knn_prediction = predict_knn_classification(train_set_np, test_X[i], 5)
		knn_predictions.append(knn_prediction)
	decision_tree_predictions = decision_tree(train_set_np, test_X, 5, 10)
	ppr_predictions = []
	for i, row in enumerate(test_X):
		ppr_prediction = ppr_based_classifer([i], labels_dict)
		ppr_predictions.append(ppr_prediction)
	print("knn prediction")
	print(knn_predictions)
	print("decision tree predictions")
	print(decision_tree_predictions)
	print("ppr prediction")
	print(ppr_predictions)
	# ppr_based_classifer(["1","2"], labels_dict)
	n_folds = 5
	max_depth = 5
	min_size = 10
	print("actual labels are: ")
	print(test_y)
	print("knn accuracy is: " + str(accuracy(knn_predictions, test_y)))
	print("decision tree accuracy is: " + str(accuracy(decision_tree_predictions, test_y)))
	print("ppr based classifier accuracy is: " + str(accuracy(ppr_predictions, test_y)))
	return knn_predictions, decision_tree_predictions, ppr_predictions

if __name__ == "__main__":
    main()
	# task2_main(1,["1","5","7"])