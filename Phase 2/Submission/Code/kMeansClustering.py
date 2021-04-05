import random
import scipy.io
import numpy as np
from collections import defaultdict

def centers_for_strategy1(dataset, k):
    center_indices = random.sample(range(len(dataset)), k)
    centers = [dataset[i] for i in center_indices]
    
    return centers

def centers_for_strategy2(dataset, k):
    centers = []
    
    #first center 
    center_index = random.randrange(1, len(dataset))
    centers.append(dataset[center_index])
    #subsequent centers
    for _ in range(1, k):
        distance = {}
        for data in dataset:
            data_in_center = [(data == center).all() for center in centers]
            if not any(data_in_center):
                dist = 0
                for center in centers:
                    dist += euclidean_distance(center, data)
                distance[dist] = data
        
        #Next center
        max_distance = max(distance.keys())
        next_center = distance[max_distance]
        centers.append(next_center)

    return centers

# Similarity Method is Euclidean Distance
def euclidean_distance(point1, point2):
    # sqrt of [(x1-x2)^2 + (y1-y2)^2]
    diff = 0
    for p1, p2 in zip(point1, point2):
        diff += ((p1 - p2)**2)
    distance = np.sqrt(diff)
    
    return distance

def find_cluster(centers, data_point):
    '''
    centers = list of all the centers
    data_point whose distance with the center is the smallest belongs to that cluster
    '''
    distances_from_centers = []
    for center in centers:
        dist = euclidean_distance(center, data_point)
        distances_from_centers.append(dist)
    
    # Distance from center of cluster the datapoint belongs to
    distance_from_center = min(distances_from_centers)
    
    #Find center with shortest distance
    cluster_index = distances_from_centers.index(distance_from_center)
    
    return cluster_index, distance_from_center

def update_centers(old_centers, dataset, cluster_data):
    '''
    old_centers = list of old centers
    dataset = list of dataset
    cluster_data = cluster index in ith position for ith position of dataset
    '''
    new_centers = []
    for index in range(len(old_centers)):
        # dataset and cluster_data are mapped 1 on 1
        member_data = [dataset[i] for i, x in enumerate(cluster_data) if x['index'] == index]

        # To check if a member exists for the center, if not copy the old center
        if member_data:
            new_center = []
            for idx in range(len(member_data[0])):
                vals = [x[idx] for x in member_data]
                val = (sum(vals))/float(len(vals))
                new_center.append(val)
        else:
            new_center = old_centers[index]
            
        new_centers.append(new_center)
    
    return new_centers

def compare_centers(old_centers, new_centers):
    for old, new in zip(old_centers, new_centers):
        for o, n in zip(old, new):
            if o != n:
                return False
    return True

def k_means_clustering(k, centers, dataset):
    old_centers = centers
    while(True):
        cluster_data = []
        
        for data_point in dataset:
            cluster_id, distance = find_cluster(old_centers, data_point)
            cluster_data.append({'index':cluster_id, 'distance':distance})

        new_centers = update_centers(old_centers, dataset, cluster_data) 
        
        if compare_centers(old_centers, new_centers):
            return centers
        else:
            old_centers = new_centers
    
def group_data(dataset, gesture_names, centers):
    center_group = defaultdict(list)
    for data, name in zip(dataset, gesture_names):
        cluster_id, _ = find_cluster(centers, data)
        center_group[cluster_id].append(name)
    
    return center_group

def run(strategy, cluster_size, dataset, gesture_names):
    if strategy == 1:
        centers = centers_for_strategy1(dataset, cluster_size)
    else:
        centers = centers_for_strategy2(dataset, cluster_size)
    centers = k_means_clustering(cluster_size, centers, dataset)
    return group_data(dataset, gesture_names, centers)
    
    # print(cluster_size, centers, cost)
