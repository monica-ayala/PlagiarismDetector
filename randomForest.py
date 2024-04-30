from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
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
from sklearn.metrics import confusion_matrix, classification_report


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

load_java_code = lambda file: open(file, 'r').read()

def gather_all_node_types(java_files):
    all_node_types = set()
    for java_file in java_files:
        java_code = load_java_code(java_file) 
        tree = parse_java_code_to_ast(java_code)
        matrix = build_transition_matrix(tree)
        for src in matrix:
            all_node_types.add(src)
            for dest in matrix[src]:
                all_node_types.add(dest)
    return sorted(all_node_types)


def do_all(java_code1, java_code2, veredict):
    tree = parse_java_code_to_ast(java_code1)
    tree2 = parse_java_code_to_ast(java_code2)
    matrix1 = build_transition_matrix(tree)
    matrix2 = build_transition_matrix(tree2)
    matrix1 = normalize_transitions(matrix1)
    matrix2 = normalize_transitions(matrix2)

    node_types = get_node_types(matrix1, matrix2)  

    vector1 = matrix_to_vector(matrix1, all_node_types)
    vector2 = matrix_to_vector(matrix2, all_node_types)
    cosine_sim = calculate_cosine_similarity(vector1, vector2)

    return cosine_sim, vector1, vector2

java_files = [f"dataset/java/{file}" for file in os.listdir("dataset/java")]
all_node_types = gather_all_node_types(java_files)
def prepare_data(train_pairs, veredict_data):
    X = []
    y = []
    for pair in train_pairs:
        id1, id2 = pair.split('_')
        veredict, java_code1, java_code2 = load_pair(id1, id2, "dataset/veredict.csv")
        if veredict is not None and java_code1 is not None and java_code2 is not None:
            cosine_sim, vector1, vector2 = do_all(java_code1, java_code2, veredict)
            difference_vector = np.array(vector1) - np.array(vector2)
            X.append(difference_vector)
            y.append(int(veredict))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Preparar los datos de entrenamiento
X_train, X_test, y_train, y_test = prepare_data(train_pairs, veredict)

# Crear y entrenar el modelo de Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
# Evaluar el modelo
accuracy = rf_model.score(X_test, y_test)
print(f"Accuracy (Random Forest): {accuracy}")

# Calcular el F1-score
f1 = f1_score(y_test, y_pred)
print(f"F1-score: {f1}")

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calcula el reporte de clasificación
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
