#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 21:20:57 2022

@author: adamsun
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(0)

#Conduct binary search for regularization
def parametric_binary_search(train_embeddings, train_labels, val_embeddings, val_labels, iterations = 5, start = -5, stop = 7):
    for i in range(iterations):
        mid = (start + stop)/2
        accuracies = []
        regs = np.logspace(start = start, stop = stop, num = iterations)
        for reg in regs:
            classifier = LogisticRegression(random_state=0, C = 1/reg, max_iter=1000, verbose=0)
            classifier.fit(train_embeddings, train_labels)
            predictions = classifier.predict(val_embeddings)
            accuracy = np.mean((val_labels == predictions).astype(float))
            accuracies.append(accuracy) 
        if regs[np.argmax(accuracies)] >= 10**mid:
            start = mid
        else:
            stop = mid #Use maximum accuracy to decide split
    return 1/regs[np.argmax(accuracies)]

def main():
    datasets = ["smsm"]
    models = ["clip", "miles"]
    shot_counts = [1, 2, 4, 8, 16, 32, 64]
    vlm_embedding_path = 'full_vlm_embeddings' #embedding path for embeddings exported by export_vlm_embeddings.ipynb
    for dataset in datasets:
        for model in models:
            print(f"For {model} on {dataset}: ")
            train_path = f'{vlm_embedding_path}}/{model}_embeddings/{dataset}.v.train/'
            val_path = f'{vlm_embedding_path}/{model}_embeddings/{dataset}.v.val/'
            test_path = f'{vlm_embedding_path}/{model}_embeddings/{dataset}.v.test/'
            train_labels = np.load(train_path + 'video_category_indices.npy')
            train_embeddings = np.load(train_path + 'video_embeddings.npy')
            val_labels = np.load(val_path + 'video_category_indices.npy')
            val_embeddings = np.load(val_path + 'video_embeddings.npy')
            test_labels = np.load(test_path + 'video_category_indices.npy')
            test_embeddings = np.load(test_path + 'video_embeddings.npy')
            for shots in shot_counts:
                print(f"For {shots} shots: ")
                unique_ys = np.unique(train_labels)
                sample_indices = np.array([])
                for unique_y in unique_ys: #Pick dataset for few shot
                    random_samples = np.random.choice(np.argwhere(train_labels==unique_y).flatten(), shots, replace=False)
                    sample_indices = np.append(sample_indices, random_samples)
                train_sample_embeddings = train_embeddings[sample_indices.astype(int)]
                train_sample_labels = train_labels[sample_indices.astype(int)]
                #Parametric Binary Search on Val Set, returns inverse of ideal regularization strength
                C = parametric_binary_search(train_sample_embeddings, train_sample_labels, val_embeddings, val_labels)
                classifier = LogisticRegression(random_state=0, C = C, max_iter=1000, verbose=0)
                classifier.fit(train_sample_embeddings, train_sample_labels)
                test_pred = classifier.predict(test_embeddings)
                accuracy = np.mean((test_labels == test_pred).astype(float))
                print(f"Achieved {accuracy} accuracy")

if __name__ == "__main__":
    main()