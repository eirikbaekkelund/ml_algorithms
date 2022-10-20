#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COMP0088 lab exercises for week 3.

Add your code as specified below.

A simple test driver is included in this script. Call it at the command line like this:

  $ python week_3.py

A 4-panel figure, `week_3.pdf`, will be generated so you can check it's doing what you
want. You should not need to edit the driver code, though you can if you wish.
"""

from calendar import c
from itertools import count
import sys, os, os.path
import argparse
import pprint
from turtle import distance
from bitarray import test

import numpy as np
import math
import numpy.random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import utils


#### ADD YOUR CODE BELOW

# -- Question 1 --
def rank_array(arr_distance, k_neighbours):
    """
    Rank the indices of the array based on distance and find
    the indices of the K smallest values.

    Parameters:
    arr_distance - array containing distance between points in training set and test point
    k_neighbours - the number of neighbours

    Returns:
    arr_distance - array transformed to the ranked indices of the K points
    """
    assert(k_neighbours >= 1)

    distance_copy = arr_distance.copy()
    distance_copy.sort()
    
    ranks = {}
    rank1 = 0
    # Rank elements
    for i in range(len(distance_copy)):
        val = distance_copy[i]
        if val not in ranks.keys():
            ranks[val] = rank1
            rank1 += 1
    
    rank2 = 0
    k_nearest = []
     # Assign ranks to elements
    for idx in range(len(arr_distance)):
        element = arr_distance[idx]
        if ranks[element] in range(k_neighbours) and rank2 < k_neighbours:
            k_nearest.append(idx)
            rank2 += 1
        if len(k_nearest) == k_neighbours:
            return k_nearest

def vote_nn(indices, train_y):
    """
    
    """
    unique, counts = np.unique([train_y[indices[i]] for i in range(len(indices))], return_counts=True)
    counter_dict = dict(zip(unique, counts))

    return max(counter_dict, key=counter_dict.get)

def vote(y):
    """
    Return the class with highest number of occurances in a prediction
    from an array

    Parameters:
    y - an array with predictions

    Returns:
    cls - class with the highest number of predictions
    ! special case when several have the same number of occurences,
      then it defaults to the lowest in numerical value
    """
    unique, counts = np.unique(y, return_counts=True)
    return unique[np.argmax(counts)]

def euclidean_distance(train_X, test_point):
    """
    
    """
    return [np.sqrt(np.sum((train_X[i] - test_point)**2)) for i in range(train_X.shape[0])]
    
def nearest_neighbours_predict ( train_X, train_y, test_X, neighbours=1 ):
    """
    Predict labels for test data based on neighbourhood in
    training set.
    
    # Arguments:
        train_X: an array of sample data for training, where rows
            are samples and columns are features.
        train_y: vector of class labels corresponding to the training
            samples, must be same length as number of rows in X
        test_X: an array of sample data to generate predictions for,
            in same layout as train_X.
        neighbours: how many neighbours to canvass at each test point
        
    # Returns
        test_y: predicted labels for the samples in test_X
    """
    assert(train_X.shape[0] == train_y.shape[0])
    assert(train_X.shape[1] == test_X.shape[1])
    
    test_y = []
    for point in test_X:
        arr_distance = euclidean_distance(train_X, point)
        rank_index = rank_array(arr_distance, neighbours)
        test_y.append(vote_nn(rank_index, train_y))

    return np.array(test_y)
    

# -- Question 2 --

def misclassification ( y, cls, weights=None ):
    """
    Calculate (optionally-weighted) misclassification error for
    a given set of labels if assigned the given class.
    
    # Arguments
        y: a set of class labels
        cls: a candidate classification for the set
        weights: optional weights vector specifying relative
            importance of the samples labelled by y
    
    # Returns
        err: the misclassification error of the candidate labels
    """
    if weights is not None:
        return np.sum( weights @ np.where( y != cls, 1, 0) )
  
    return ( 1 / len(y) ) * np.sum( np.where( y != cls, 1, 0) ) 

def misclassification_array(y_pred, y_truth):
    """
    Calculate misclassification array for a given
    set of labels if assigned the given class.

    Parameters:
    y_pred - predicted class labels
    y_truth - ground truth labels
    """
    return np.where( y_pred != y_truth, 1, 0)
    

def decision_node_split ( X, y, cls=None, weights=None, min_size=3 ):
    """
    Find (by brute force) a split point that best improves the weighted
    misclassification error rate compared to the original one (or not, if
    there is no improvement possible).
    
    Features are assumed to be numeric and the test condition is
    greater-or-equal.
    
    # Arguments:
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of class labels corresponding to the samples,
            must be same length as number of rows in X
        cls: class label currently assigned to the whole set
            (if not specified we use the most common class in y, or
            the lowest such if 2 or more classes occur equally)
        weights: optional weights vector specifying relevant importance
            of the samples
        min_size: don't create child nodes smaller than this
    
    # Returns:
        feature: index of the feature to test (or None, if no split)
        thresh: value of the feature to test (or None, if no split)
        c0: class assigned to the set with feature < thresh
            (or None, if no split)
        c1: class assigned to the set with feature >= thresh
            (or None, if no split)
    """
    assert(X.shape[0] == len(y))

    best_feature, best_threshold = None, None
    best_c0, best_c1 = None, None

    if cls == None:
        cls = utils.vote(y)
    
    if weights is None:
        weights = np.ones(X.shape[0])/X.shape[0]
        
    init_loss = misclassification(y=y, cls=cls, weights=weights)
    if init_loss == 0:
        return None, None, None, None

    loss_gain = 0
    n_features = X.shape[-1]
    
    for feature in range(n_features):
        for tresh in X[:,feature]:
            X1 = X[:,feature] > tresh
            # negation of X1
            X2 = ~X1

            # /TODO ask: 
            # how does this account for splits?
            if (np.sum(X1) < min_size) or (np.sum(X2) < min_size):
                continue
            y1 = y[X1]
            y2 = y[X2]

            weights1 = weights[X1]
            weights2 = weights[X2]

            c1 = np.unique(y1)
            c2 = np.unique(y2)

            loss1 = [misclassification(y=y1, cls=cl, weights=weights1) for cl in c1]
            loss2 = [misclassification(y=y2, cls=cl, weights=weights2) for cl in c2]
            
            # gives index of corresponding to class with 
            # minimum loss passed to unique classes
            node_1_class = c1[np.argmin(loss1)]
            node_2_class = c2[np.argmin(loss2)]
            
            # predicting the class with minimum loss leads to following loss
            best_loss1 = np.min(loss1)
            best_loss2 = np.min(loss2)

            # take old loss and subtract new loss to see gain
            loss_improvement = init_loss - (best_loss1 +  best_loss2)

            if loss_improvement > loss_gain:
                best_feature = feature
                best_threshold = tresh
                loss_gain = loss_improvement
                best_c0 = node_1_class
                best_c1 = node_2_class
                
    if best_feature == None:
        return None, None, None, None
    
    else:
        return best_feature, best_threshold, best_c0, best_c1


def decision_tree_train ( X, y, cls=None, weights=None,
                          min_size=3, depth=0, max_depth=10 ):
    """
    Recursively choose split points for a training dataset
    until no further improvement occurs.
    
    # Arguments:
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of class labels corresponding to the samples,
            must be same length as number of rows in X
        cls: class label currently assigned to the whole set
            (if not specified we use the most common class in y, or
            the lowest such if 2 or more classes occur equally)
        weights: optional weights vector specifying relevant importance
            of the samples
        min_size: don't create child nodes smaller than this
        depth: current recursion depth
        max_depth: maximum allowed recursion depth
    
    # Returns:
        tree: a dict containing (some of) the following keys:
            'kind' : either 'leaf' or 'decision'
            'class' : the class assigned to this node (leaf)
            'feature' : index of feature on which to split (decision)
            'thresh' : threshold at which to split the feature (decision)
            'below' : a nested tree for when feature < thresh (decision)
            'above' : a nested tree for when feature >= thresh (decision)
    """
    if cls == None:
        cls = utils.vote(y)
    if depth == max_depth:
        # return majority vote if no nodes
        return {'kind' : 'leaf', 'class' : cls}
    else:
        # we perform a split
        feat, thresh, cl0, cl1 = decision_node_split(X=X,
                                                    y=y,
                                                    cls=cls,
                                                    weights=weights,
                                                    min_size=min_size)

    
    if feat is None:
        # No improvement on loss by split, so return majority vote
        return {'kind' : 'leaf', 'class' : cls}

    X1 = X[:, feat] >= thresh
    X2 = ~X1

    return {'kind' : 'decision',
            'feature' : feat,
            'thresh': thresh,
            'below' : decision_tree_train(X = X[X2,:],
                                          y=y[X2],
                                          cls=cl1,
                                          weights=None if weights is None else weights[X2],
                                          depth=depth+1,
                                          min_size=min_size,
                                          max_depth=max_depth
                                          ),
            'above' : decision_tree_train(X = X[X1,:],
                                          y=y[X1],
                                          cls=cl0,
                                          weights=None if weights is None else weights[X1],
                                          depth=depth+1,
                                          min_size=min_size,
                                          max_depth=max_depth
                                          )}
    
def predictor_tree(tree, x):
    """
    
    """
    while True:
        if tree['kind'] == 'leaf':
            return tree['class']
        
        tree = tree['above'] if x[tree['feature']] >= tree['thresh'] else tree['below']

def decision_tree_predict ( tree, X ):
    """
    Predict labels for test data using a fitted decision tree.
    
    # Arguments
        tree: a decision tree dictionary returned by decision_tree_train
        X: an array of sample data, where rows are samples
            and columns are features.

    # Returns
        y: array of the predicted labels 
    """
    
    return np.array([ predictor_tree( tree, X[i,:] ) for i in range(X.shape[0]) ]) 


# -- Question 3 --

def random_forest_train ( X, y, k, rng, min_size=3, max_depth=10 ):
    """
    Train a (simplified) random forest of decision trees.
    
    # Arguments:
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of binary class labels corresponding to the
            samples, must be same length as number of rows in X
        k: the number of trees in the forest
        rng: an instance of numpy.random.Generator
            from which to draw random numbers        min_size: don't create child nodes smaller than this
        max_depth: maximum tree depth
    
    # Returns:
        forest: a list of tree dicts as returned by decision_tree_train
    """
    forest = []
    rng = np.random.default_rng(seed=12345)

    for _ in range(k):
        # get random set of training data from X & y (corresponding indices)
        # train decision tree on that data
        # append the trained tree to the forest
        idx = rng.choice(np.arange(len(y)), size=len(y))
        X_slice  = X[idx,:]
        y_slice = y[idx]

        forest.append(decision_tree_train(X_slice, y_slice, 
                                          min_size=min_size, 
                                          max_depth=max_depth))
  
    return forest
    

def random_forest_predict ( forest, X ):
    """
    Predict labels for test data using a fitted random
    forest of decision trees.
    
    # Arguments
        forest: a list of decision tree dicts
        X: an array of sample data, where rows are samples
            and columns are features.

    # Returns
        y: the predicted labels
    """
    tree_predictions = np.array([decision_tree_predict(tree, X) for tree in forest])

    return np.array([vote(tree_predictions[:,i]) 
                    for i in range(max(tree_predictions.shape))
                    ])



# -- Question 4 --

def normalize_weights_adaboost(weights, alpha, y_hat, y):
    weights = np.array([ weights[i] * np.exp( - alpha * y[i] * y_hat[i] )  
                             for i in range(len(y))
                            ])
    return weights / np.sum(weights)


def adaboost_train ( X, y, k, min_size=1, max_depth=1, epsilon=1e-8 ):
    """
    Iteratively train a set of decision tree classifiers
    using AdaBoost.
    
    # Arguments:
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of binary class labels corresponding to the
            samples, must be same length as number of rows in X
        k: the maximum number of weak classifiers to train
        min_size: don't create child nodes smaller than this
        max_depth: maximum tree depth -- by default we just
            use decision stumps
        epsilon: threshold below which the error is considered 0
    
    # Returns:
        trees: a list of tree dicts as returned by decision_tree_train
        alphas: a vector of weights indicating how much credence to
            given each of the decision tree predictions
    """
    trees, alphas = [], []
    weights = ( 1 / len(y) ) * np.ones( len(y) )

    for _ in range(k):
        #train ensemble, get prediction and update weights
        trees.append(decision_tree_train(X=X,
                                        y=y,
                                        weights=weights,
                                        min_size=min_size,
                                        max_depth=max_depth))
        y_hat = decision_tree_predict(trees[-1], X)
      
        error = weights @ misclassification_array(y_pred=y_hat, y_truth=y)

        # get latest alpha 
        if error > epsilon:
            alphas.append( ( 1 / 2 ) * np.log( ( 1 - error) / error ))
        
        # error ~= 0 --> perfect prediction
        else:
            alphas.append(1)
            break

        weights = normalize_weights_adaboost(weights=weights, 
                                             alpha=alphas[-1], 
                                             y_hat=y_hat, 
                                             y=y)
    
    alphas = np.array(alphas)

    return trees, alphas


def adaboost_predict ( trees, alphas, X ):
    """
    Predict labels for test data using a fitted AdaBoost
    ensemble of decision trees.
    
    # Arguments
        trees: a list of decision tree dicts
        alphas: a vector of weights for the trees
        X: an array of sample data, where rows are samples
            and columns are features.

    # Returns
        y: the predicted labels
    """
    preds = np.array( [decision_tree_predict(tree, X) for tree in trees] )
    preds_weighted = alphas @ preds

    return np.where(preds_weighted >= 0, 1, 0)
    

#### TEST DRIVER

def process_args():
    ap = argparse.ArgumentParser(description='week 3 coursework script for COMP0088')
    ap.add_argument('-s', '--seed', help='seed random number generator', type=int, default=None)
    ap.add_argument('-n', '--num_samples', help='number of samples to use', type=int, default=50)
    ap.add_argument('-k', '--neighbours', help='number of neighbours for k-NN fit', type=int, default=3)
    ap.add_argument('-m', '--min_size', help='smallest acceptable tree node', type=int, default=3)
    ap.add_argument('-w', '--weak', help='how many weak classifiers to train for AdaBoost', type=int, default=10)
    ap.add_argument('-f', '--forest', help='how many trees to train for random forest', type=int, default=10)
    ap.add_argument('-r', '--resolution', help='grid sampling resolution for classification plots', type=int, default=20)
    ap.add_argument('-d', '--data', help='CSV file containing training data', default='week_3_data.csv')
    ap.add_argument('file', help='name of output file to produce', nargs='?', default='week_3.pdf')
    return ap.parse_args()

if __name__ == '__main__':
    args = process_args()
    rng = numpy.random.default_rng(args.seed)
    
    print(f'loading data from {args.data}')
    df = pd.read_csv(args.data)
    X = df[['X1','X2']].values[:args.num_samples,:]
    y = df['Multi'].values[:args.num_samples]

    fig = plt.figure(figsize=(10, 10))
    axs = fig.subplots(nrows=2, ncols=2)
    
    print(f'Q1: checking {args.neighbours}-nearest neighbours fit')
    # this is a fudge -- there's no training phase, so check implementation with a dummy prediction
    dummy = nearest_neighbours_predict ( X[:3,:], y[:3], X[:3,:], neighbours=args.neighbours )
    if dummy is None:
        print('decision tree not implemented')
        utils.plot_unimplemented(axs[0,0], f'{args.neighbours}-Nearest Neighbours')
    else:    
        print(f'Q1: plotting {args.neighbours}-nearest neighbours fit')    
        nn_cls = lambda z: nearest_neighbours_predict ( X, y, z, neighbours=args.neighbours )
        utils.plot_classification_map(axs[0,0], nn_cls, X, y, resolution=args.resolution, title=f'{args.neighbours}-Nearest Neighbours')
    
    print('Q2: testing misclassification error')
    all_right = misclassification(np.ones(3), 1)
    all_wrong = misclassification(np.ones(3), 0)
    fifty_fifty = misclassification(np.concatenate((np.ones(3), np.zeros(3))), 1)
    
    right_msg = 'correct' if np.isclose(all_right, 0) else 'wrong, should be 0'
    wrong_msg = 'correct' if np.isclose(all_wrong, 1) else 'wrong, should be 1'
    fifty_msg = 'correct' if np.isclose(fifty_fifty, 0.5) else 'wrong should b 0.5'
    
    print(f' all right: {all_right} - {right_msg}')
    print(f' all wrong: {all_wrong} - {wrong_msg}')
    print(f' fifty-fifty: {fifty_fifty} - {fifty_msg}')
    
    print('Q2: fitting decision tree')
    tree = decision_tree_train ( X, y, min_size=args.min_size )
    
    if tree is None:
        print('decision tree not implemented')
        utils.plot_unimplemented(axs[0,1], 'Decision Tree')
    else:
        print('Q2: plotting decision tree fit')
        tree_cls = lambda z: decision_tree_predict ( tree, z )
        utils.plot_classification_map(axs[0,1], tree_cls, X, y, resolution=args.resolution, title='Decision Tree')
    
    print(f'Q3: fitting random forest with {args.forest} trees')
    forest = random_forest_train ( X, y, args.forest, rng=rng, min_size=args.min_size )
    
    if forest is None:
        print('random forest not implemented')
        utils.plot_unimplemented(axs[1,0], 'Random Forest')
    else:
        print('Q3: plotting random forest fit')
        forest_cls = lambda z: random_forest_predict ( forest, z )
        utils.plot_classification_map(axs[1,0], forest_cls, X, y, resolution=args.resolution, title=f'Random Forest ({args.forest} Trees)')
        
    print('Q4: fitting adaboost ensemble')
    # swap to binary labels since we're only doing 2-class AdaBoost
    y = df['Binary'].values[:args.num_samples]
    trees, alphas = adaboost_train ( X, y, args.weak )
    
    if trees is None:
        print('adaboost not implemented')
        utils.plot_unimplemented(axs[1,1], 'AdaBoost')
    else:   
        print('Q4: plotting AdaBoost fit')
        ada_cls = lambda z: adaboost_predict ( trees, alphas, z )
        utils.plot_classification_map(axs[1,1], ada_cls, X, y, resolution=args.resolution, title=f'AdaBoost ({args.weak} Stumps)')

    fig.tight_layout(pad=1)
    fig.savefig(args.file)
    plt.close(fig)
