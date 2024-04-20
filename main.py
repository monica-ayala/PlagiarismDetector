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

train_pairs = pd.read_csv('dataset/train_pairs.csv', delimiter=',', usecols=["pairs"])['pairs'].values
test_pairs = pd.read_csv('dataset/test_pairs.csv', delimiter=',', usecols=["pairs"])['pairs'].values
veredict = pd.read_csv('dataset/veredict.csv', delimiter=',')

java_code= """
import java.util.*;
import java.io.*;

public class Main {

    public static void main(String[] args) throws IOException {
        Scanner sc = new Scanner(System.in);
        PrintWriter pw = new PrintWriter(System.out);
        int t = sc.nextInt();
        while(t-- > 0){
            int n = sc.nextInt();
            String[] s = new String[n];
            for(int i=0; i<n; i++)
                s[i] = sc.next();
            int MAX = 0;
            for(char c = 'a'; c <= 'e'; c++){
                PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder()); //Big comes in top;
                for(int i=0; i<n; ++i) {
                    int curChar = 0;
                    int otherChar = 0;
                    for(int j=0; j<s[i].length(); j++) {
                        if(s[i].charAt(j) == c)
                            curChar++;
                        else
                            otherChar++;
                    }
                    int diff = curChar - otherChar;
                    pq.add(diff);
                }
                int cur = 0;
                int numberOfWords = 0;
                while(!pq.isEmpty()){
                    if(cur + pq.peek() > 0){
                        cur += pq.poll();
                        numberOfWords++;
                    }else{
                        break;
                    }
                }
                MAX = Math.max(MAX, numberOfWords);
            }
            pw.println(MAX);
        }
        pw.close();
    }

    
}
"""

java_code1 = """
import java.util.*;
public class E1547 {
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int q = sc.nextInt();
        for(int i = 0; i < q; i++){
            int n = sc.nextInt();
            int k = sc.nextInt();
            int[][] t = new int[k][2];
            for(int j = 0; j < k; j++){
                t[j][0] = sc.nextInt();//room
            }
            for(int j = 0; j < k; j++){
                t[j][1] = sc.nextInt();//air
            }
            long[] left = new long[n];
            long[] right = new long[n];
            long tmp = Integer.MAX_VALUE;
            long[] max =new long[n];
            for(int j = 0; j < n; j++){
                max[j] = Integer.MAX_VALUE;
            }
            for (int j = 0; j < k; j++) {
                max[t[j][0]-1] = t[j][1];
            }
            for (int j = 1; j <= n; j++) {
                tmp = Math.min(tmp+1, max[j-1]);
                left[j-1] = tmp;
            }
            for(int j = n; j >= 1; j--){
                tmp = Math.min(tmp+1, max[j-1]);
                right[j-1] = tmp;
            }
            for(int j = 0; j < n; j++){
                System.out.print(Math.min(left[j], right[j]) + " ");
            }
            System.out.println();
        }
    }
}
"""

java_code2 = """
import java.util.*;
import java.io.*;

public class Main {

    public static void main(String[] args) throws IOException {
        Scanner sc = new Scanner(System.in);
        PrintWriter pw = new PrintWriter(System.out);
        int t = sc.nextInt();
        while(t-- > 0){
            int n = sc.nextInt();
            String[] s = new String[n];
            for(int i=0; i<n; i++)
                s[i] = sc.next();
            int MAX = 0;
            for(char c = 'a'; c <= 'e'; c++){
                PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder()); //Big comes in top;
                for(int i=0; i<n; ++i) {
                    int curChar = 0;
                    int otherChar = 0;
                    for(int j=0; j<s[i].length(); j++) {
                        if(s[i].charAt(j) == c)
                            curChar++;
                        else
                            otherChar++;
                    }
                    int diff = curChar - otherChar;
                    pq.add(diff);
                }
                int cur = 0;
                int numberOfWords = 0;
                while(!pq.isEmpty()){
                    if(cur + pq.peek() > 0){
                        cur += pq.poll();
                        numberOfWords++;
                    }else{
                        break;
                    }
                }
                MAX = Math.max(MAX, numberOfWords);
            }
            pw.println(MAX);
        }
        pw.close();
    }

    
}
"""

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

from sklearn.metrics.pairwise import cosine_similarity
def calculate_cosine_similarity(vec1, vec2):
    similarity = cosine_similarity([vec1], [vec2])
    return similarity[0][0] 

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

cosine_sim = calculate_cosine_similarity(vector1, vector2)

print("Cosine Similarity:", cosine_sim)

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