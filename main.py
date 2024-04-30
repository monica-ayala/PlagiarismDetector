import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tensorflow import keras
from keras import models, layers
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import javalang
from javalang import parse
import collections
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os

train_pairs = pd.read_csv('dataset/train_pairs.csv', delimiter=',', usecols=["pairs"])['pairs'].values
test_pairs = pd.read_csv('dataset/test_pairs.csv', delimiter=',', usecols=["pairs"])['pairs'].values
veredict = pd.read_csv('dataset/veredict.csv', delimiter=',')

def parse_java_code_to_ast(java_code):
    tree = javalang.parse.parse(java_code)
    return tree

def traverse_ast(node, transitions, last_node_type=None):
    if not node:
        return
    if isinstance(node, list):
        for item in node:
            traverse_ast(item, transitions, last_node_type)
    
    current_node_type = type(node).__name__
    
    if last_node_type:
        transitions[last_node_type][current_node_type] += 1
    
    if hasattr(node, 'children'):
        children = node.children
        if not isinstance(children, list):
            children = [children]
        
        for child in children:
            traverse_ast(child, transitions, current_node_type)


def build_transition_matrix(tree):
    transitions = defaultdict(lambda: defaultdict(int))
    traverse_ast(tree, transitions)
    return transitions

def display_transition_matrix(matrix):
    print("Transition Matrix:")
    for src, dests in matrix.items():
        print(f"{src}:")
        for dest, count in dests.items():
            print(f"  {dest}: {count}")

def normalize_transitions(transitions):
    transition_matrix = {}

    for src, dest_counts in transitions.items():
        total = sum(dest_counts.values())  
        if total == 0:
            continue
        
        transition_matrix[src] = {
            dest: count / total for dest, count in dest_counts.items()
        }

    return transition_matrix

def get_node_types(transitions1, transitions2):
    node_types = set(transitions1.keys()) | set(transitions2.keys())
    for key in transitions1:
        node_types.update(transitions1[key].keys())
    for key in transitions2:
        node_types.update(transitions2[key].keys())
    
    return sorted(node_types) 

def matrix_to_vector(matrix, node_types):
    vector = []
    for src in node_types:
        for dest in node_types:
            vector.append(matrix.get(src, {}).get(dest, 0))
    return vector

def calculate_cosine_similarity(vec1, vec2):
    similarity = cosine_similarity([vec1], [vec2])
    return similarity[0][0] 

def load_pair(id1, id2, csv_path):
    veredict = None
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == id1 and row[1] == id2:
                veredict = row[2]
                break
    
    if veredict is None:
        return "Verdict not found for the given IDs", None, None

    file_path1 = os.path.join("dataset/java/", f"{id1}.java")
    file_path2 = os.path.join("dataset/java/", f"{id2}.java")

    try:
        with open(file_path1, 'r') as file1:
            code1 = file1.read()
    except FileNotFoundError:
        return f"File not found: {file_path1}", None, None

    try:
        with open(file_path2, 'r') as file2:
            code2 = file2.read()
    except FileNotFoundError:
        return f"File not found: {file_path2}", None, None
    
    return veredict, code1, code2

def do_all(java_code1, java_code2, veredict):
    tree = parse_java_code_to_ast(java_code1)
    tree2 = parse_java_code_to_ast(java_code2)
    matrix1 = build_transition_matrix(tree)
    matrix2 = build_transition_matrix(tree2)
    matrix1 = normalize_transitions(matrix1)
    matrix2 = normalize_transitions(matrix2)

    node_types = get_node_types(matrix1, matrix2)  
    print("Node Types:", node_types)

    vector1 = matrix_to_vector(matrix1, node_types)
    vector2 = matrix_to_vector(matrix2, node_types)
    print("Shape:", len(vector1), len(vector2))

    cosine_sim = calculate_cosine_similarity(vector1, vector2)

    # print("Cosine Similarity:", cosine_sim)
    # print("Veredict:", veredict)
    
    return cosine_sim
    
correct = 0
for pair in train_pairs:
    id1, id2 = pair.split('_')
    print(f"Processing pair {id1} vs {id2}")
    veredict, java_code1, java_code2 = load_pair(id1, id2, "dataset/veredict.csv")
    cosine_sim = do_all(java_code1, java_code2, veredict)
    treshold = 0.85
    veredict_cs = 0
    if(cosine_sim > treshold):
        veredict_cs = 1
    
    if(veredict_cs == int(veredict)):
        correct += 1
    
print(f"Accuracy: {correct} and {len(train_pairs)}")
    

# Preprocess the data
# Preprocess the data
# TODO: Eliminate white spaces, comments, etc.
# TODO: Generate AST Trees from the code
# TODO: Calculate the Distance between the AST Trees (Maybe using Markov Chains)

# Create the dataset after preprocessing
# TODO: Join the pairs and the veredict in a single dataset (feature, target)

# Create the model
# model = models.Sequential()
# TODO: Choose the best model for the problem, for Classification. (CNN, RNN, Xgboost, etc.)

# Compile the model
# TODO: Change loss function as needed
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# Train the model and save the history

# Evaluate the model