from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd
import javalang
from javalang import parse
import collections
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import csv
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import numpy as np

all_pairs = pd.read_csv('dataset/all_pairs.csv', delimiter=',', usecols=["pairs"])['pairs'].values

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
            
def get_ngram_similarity(text1, text2, n=2):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    ngrams1 = vectorizer.fit_transform([text1])
    ngrams2 = vectorizer.transform([text2])
    
    ngrams1_binary = (ngrams1.toarray()[0] > 0).astype(int)
    ngrams2_binary = (ngrams2.toarray()[0] > 0).astype(int)
    
    similarity = jaccard_score(ngrams1_binary, ngrams2_binary, average='binary')
    return similarity

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
        return "Veredict not found for the given IDs", None, None

    file_path1 = os.path.join("dataset/java/", f"{id1}.java")
    file_path2 = os.path.join("dataset/java/", f"{id2}.java")

    try:
        with open(file_path1, 'r', encoding='utf-8') as file1:
            code1 = file1.read()
    except FileNotFoundError:
        return f"File not found: {file_path1}", None, None

    try:
        with open(file_path2, 'r', encoding='utf-8') as file2:
            code2 = file2.read()
    except FileNotFoundError:
        return f"File not found: {file_path2}", None, None
    
    return veredict, code1, code2

load_java_code = lambda file: open(file, 'r', encoding='utf-8').read()

def gather_all_node_types(java_files):
    all_node_types = set()
    for java_file in java_files:
        try:
            java_code = load_java_code(java_file)
            tree = parse_java_code_to_ast(java_code)
            matrix = build_transition_matrix(tree)
            for src in matrix:
                all_node_types.add(src)
                for dest in matrix[src]:
                    all_node_types.add(dest)
        except UnicodeDecodeError as e:
            print(f"Error al leer el archivo {java_file}: {e}")
    return sorted(all_node_types)

def average_line_length_from_string(code):
    lines = code.splitlines()  
    if not lines:
        return 0 
    total_length = sum(len(line.strip()) for line in lines)  
    average_length = total_length / len(lines) if lines else 0  
    return average_length
    
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
    vec1 = matrix_to_vector(matrix1, node_types)
    vec2 = matrix_to_vector(matrix2, node_types)
    cosine_sim = calculate_cosine_similarity(vec1, vec2)
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)

    euc = pairwise_distances(vec1, vec2, metric='euclidean')[0][0]
    man = pairwise_distances(vec1, vec2, metric='manhattan')[0][0]
    che = pairwise_distances(vec1, vec2, metric='chebyshev')[0][0]
    bigram_similarity = get_ngram_similarity(java_code1, java_code2, 4)
    
    avg1 = average_line_length_from_string(java_code1)
    avg2 = average_line_length_from_string(java_code2)
    avg_diff = abs(avg1 - avg2)
    avg_len_diff = abs(len(java_code1.splitlines()) - len(java_code2.splitlines()))

    return cosine_sim, euc, man, che, bigram_similarity, avg_diff, avg_len_diff, vector1, vector2

java_files = [f"dataset/java/{file}" for file in os.listdir("dataset/java")]
all_node_types = gather_all_node_types(java_files)

def prepare_data(all_pairs):
    X = []
    y = []
    for pair in all_pairs:
        id1, id2 = pair.split('_')
        veredict, java_code1, java_code2 = load_pair(id1, id2, "dataset/veredict.csv")
        if veredict is not None and java_code1 is not None and java_code2 is not None:
            cosine_sim, euc, man, che, bigram_similarity, avg_diff, avg_len_diff, vector1, vector2 = do_all(java_code1, java_code2, veredict)
            difference_vector = np.array(vector1) - np.array(vector2)
            feature_vector = np.append(difference_vector, [cosine_sim, euc, man, che, bigram_similarity, avg_diff, avg_len_diff])
            X.append(feature_vector)
            y.append(int(veredict))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prepare_data(all_pairs)

def save_data(X_train, X_test, y_train, y_test):

    np.save("dataset/X_train.npy", X_train)
    np.save("dataset/X_test.npy", X_test)
    np.save("dataset/y_train.npy", y_train)
    np.save("dataset/y_test.npy", y_test)
    
save_data(X_train, X_test, y_train, y_test)