import os
import sys
from csv import reader
import math
import numpy as np
import copy

class Node(object):
    label = ''
    is_leaf = None
    children = []
    feat_name_idx = None
    depth = 0
    value = ''
    feat_value = None
    features = []

    def __init__(self, features=[], parent_node=None, label="", feat_name_idx=0, children=[], is_leaf=False):
        self.features = features
        self.label = ""
        self.parent_node = parent_node
        self.children = []
        self.is_leaf = False
        self.feat_name_idx = 0

def id3(data, features, target_feature, depth, heuristic, feat_map_idx, parent_node=None):
    root = Node(features)
    if depth == 0:
        root.parent_node = None
    else:
        root.parent_node = parent_node

    root.depth = depth
    data_T = np.array(data).transpose()
    data_T = data_T.tolist()
    feature = list(data_T.pop(len(data_T)-1))
    k_0 = feature.count(0)
    k_1 = feature.count(1)
    if k_0 == 0:
        root.depth = depth
        root.label = 1
        root.is_leaf = True
        root.value = 1
        root.feat_value = 1
    elif k_1 == 0:
        root.depth = depth
        root.label = 0
        root.is_leaf = True
        root.value = 0
        root.feat_value = 0
    else:
        if len(data[0]) <= 1:
            if k_1 > k_0:
                root.depth = depth
                root.label = 1
                root.is_leaf = True
                root.value = 1
                root.feat_value = 1
            else:
                root.depth = depth
                root.label = 0
                root.is_leaf = True
                root.value = 0
                root.feat_value = 0
            return root
        else:
            best_feature_idx = test_attribute(copy.deepcopy(data), heuristic, target_feature)
            if len(features) > best_feature_idx:
                root.feat_name_idx = feat_map_idx[features[best_feature_idx]]
                root.label = features[best_feature_idx]
                root.depth = depth
                features.pop(best_feature_idx)
            data_0, data_1 = [], []

            for i in range(len(data)):
                if data[i][best_feature_idx] == 0:
                    data_0.append(data[i])
                if data[i][best_feature_idx] == 1:
                    data_1.append(data[i])

            if not data_0:
                leaf = Node(features)
                if k_0 > k_1:
                    leaf.label = 0
                    leaf.depth = depth
                    leaf.is_leaf = True
                    leaf.value = 0 # edit
                    leaf.feat_value = 0 # edit
                    root.children.append(leaf)
                else:
                    leaf.label = 1
                    leaf.depth = depth
                    leaf.is_leaf = True
                    leaf.value = 0
                    leaf.feat_value = 0
                    root.children.append(leaf)
            else:
                data_0T = np.array(data_0).transpose()
                data_0T = data_0T.tolist()
                temp = data_0T.pop(best_feature_idx)
                data_0 = np.transpose(np.array(data_0T))
                data_0 = data_0.tolist()
                ele = id3(copy.deepcopy(data_0), copy.deepcopy(features), target_feature-1, depth+1, heuristic, feat_map_idx, root)
                root.children.append(ele)
                root.children[root.children.index(ele)].feat_value = 0

            if not data_1:
                leaf = Node(features)
                if k_0 > k_1:
                    leaf.label = 0
                    leaf.depth = depth
                    leaf.is_leaf = True
                    leaf.value = 0 # edit
                    leaf.feat_value = 0 # edit
                    root.children.append(leaf)
                else:
                    leaf.label = 1
                    leaf.depth = depth
                    leaf.is_leaf = True
                    leaf.value = 1
                    leaf.feat_value = 1
                    root.children.append(leaf)
            else:
                data_1T = np.array(data_1).transpose()
                data_1T = data_1T.tolist()
                temp = data_1T.pop(best_feature_idx)
                data_1 = np.transpose(np.array(data_1T))
                data_1 = data_1.tolist()
                ele = id3(copy.deepcopy(data_1), copy.deepcopy(features), target_feature-1, depth+1, heuristic, feat_map_idx, root)
                root.children.append(ele)
                root.children[root.children.index(ele)].feat_value = 1
    return root

def test_attribute(data, heuristic, label_idx):
    gain = []
    data = np.array(data)
    num_feats = data.shape[1]-1
    if heuristic == 'entropy':
        heuristic_value = initial_entropy(data, label_idx)
        for i in range(num_feats):
            gain.append(info_gain(data, i, label_idx, 'entropy', heuristic_value))
    elif heuristic == 'variance':
        heuristic_value = initial_variance(data, label_idx)
        for i in range(num_feats):
            gain.append(info_gain(data, i, label_idx, 'variance', heuristic_value))
    max_idx = gain.index(max(gain))
    return max_idx

def predict(row, root):
    if root.is_leaf:
        return root.value
    else:
        return predict(row, root.children[row[root.feat_name_idx]])

def accuracy(data, root):
    correct = 0
    for row in data:
        prediction = predict(row, root)
        if prediction == row[-1]:
            correct +=1
    return (correct*100)/len(data)

def info_gain(data, feature_idx, label_idx, heuristic, previous_heuristic_score):
    data_calc = np.array(data)
    k_0 = np.count_nonzero(data==0, 0)[feature_idx]
    k_1 = np.count_nonzero(data==1, 0)[feature_idx]
    feature = list(data[:, feature_idx])
    l = len(feature)
    if heuristic == 'entropy':
        e_0 = k_0/(k_0+k_1) * feature_entropy(data, feature_idx, 0, label_idx)
        e_1 = k_1/(k_0+k_1) * feature_entropy(data, feature_idx, 1, label_idx)
        gain = previous_heuristic_score - e_0 - e_1
    elif heuristic == 'variance':
        v_0 = k_0/(k_0+k_1) * feature_variance(data, feature_idx, 0, label_idx)
        v_1 = k_1/(k_0+k_1) * feature_variance(data, feature_idx, 1, label_idx)
        gain = previous_heuristic_score - v_0 - v_1
    return gain

def entropy(k, l):
    try:
        e = (k/l) * math.log((k/l), 2)
    except:
        e = 0
    return e

def initial_entropy(data, label_idx):
    data = np.array(data)
    k_0 = np.count_nonzero(data==0, 0)[label_idx]
    k_1 = np.count_nonzero(data==1, 0)[label_idx]
    l = len(data)
    e_0 = entropy(k_0, l)
    e_1 = entropy(k_1, l)
    return -(e_0+e_1)

def feature_entropy(data, feature_idx, feature_value, label_idx):
    k_0, k_1 = 0, 0
    e = 0
    l = len(data)
    for i in range(l):
        if data[i][feature_idx] == feature_value:
            if data[i][label_idx] == 0:
                k_0 += 1
            else:
                k_1 += 1
    tot = k_0 + k_1
    e_0 = entropy(k_0, tot)
    e_1 = entropy(k_1, tot)
    return -(e_0+e_1)

def feature_variance(data, feature_idx, feature_value, label_idx):
    k_0, k_1 = 0, 0
    l = len(data)
    for i in range(l):
        if data[i][feature_idx] == feature_value:
            if data[i][label_idx] == 0:
                k_0 += 1
            else:
                k_1 += 1
    tot = k_0 + k_1
    try:
        v_0 = k_0/tot
    except:
        v_0 = 0
    try:
        v_1 = k_1/tot
    except:
        v_1 = 0
    return (v_0*v_1)

def initial_variance(data, label_idx):
    data = np.array(data)
    k_0 = np.count_nonzero(data==0, 0)[label_idx]
    k_1 = np.count_nonzero(data==1, 0)[label_idx]
    l = len(data)
    return (k_0/l)*(k_1/l)

def parse_data(data_file):
    fp = open(data_file, 'r')
    lines = fp.readlines()
    info = []
    for idx, line in enumerate(lines):
        if idx == 0:
            info.append([str(ele) for ele in line.split(',')])
        else:
            info.append([int(ele) for ele in line.split(',')])
    return info

def print_formatted_tree(root, feat_name):
    if feat_name == 'none':
        pass
    else:
        if root.is_leaf:
            print(feat_name, '=', root.feat_value, ':', root.value)
        else:
            print(feat_name, '=', root.feat_value, ':')
    for i in range(len(root.children)):
        print_formatted_tree(root.children[i], (root.depth * "| ") + root.label)

def main(flags):

    # Complete execution

    if len(flags) < 5:
        print("Not enough arguments")
        sys.exit(0)

    train_path = flags[0]
    dev_path = flags[1]
    test_path = flags[2]

    train_data = parse_data(train_path)
    dev_data = parse_data(dev_path)
    test_data = parse_data(test_path)

    class_idx = len(train_data[0]) - 1
    features = train_data.pop(0)
    features.pop(-1)
    dev_data.pop(0)
    test_data.pop(0)

    feat_map_idx = {}
    for i, feat in enumerate(features):
        feat_map_idx[feat] = i

    heuristic = flags[3]

    root = Node()
    root = id3(train_data, copy.deepcopy(features), class_idx, 0, heuristic.lower(), feat_map_idx, None)

    dataset_path = train_path.split('/')
    set_no = ''
    for path_split in dataset_path:
        if 'data' in path_split and '1' in path_split:
            set_no = 1
            break
        elif 'data' in path_split and '2' in path_split:
            set_no = 2
            break

    acc = accuracy(test_data, root)
    acc_str = f"{heuristic} Heuristic Accuracy for dataset {set_no}: {acc}"

    to_print = flags[4]
    if to_print.lower() == 'yes':
        print_formatted_tree(root, 'none')
        print(acc_str)

    out_file = open('accuracy.txt', 'a+')
    out_file.write(acc_str+'\n')

    out_file.close()


if __name__=='__main__':
    main(sys.argv[1:])