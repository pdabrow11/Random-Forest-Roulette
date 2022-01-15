from math import sqrt
from random import seed
from random import randrange
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import numpy as np
import numpy.random as npr

def cross_validation_split(dataset, folds_num):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds_num)
	for i in range(folds_num):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
def calc_accuracy(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
def evaluate_algorithm(dataset, folds_num, trees_num, *args):
	attributes_num = int(sqrt(len(dataset[0])-1))
	folds = cross_validation_split(dataset, folds_num)
	scores = list()
	scores_sklearn = list()
	res = RandomForestClassifier(n_estimators=trees_num)
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		y_test_set = list()
		X, y = list(), list()
		for row in fold:
			#row_copy = list(row)
			#row_copy[-1] = None
			test_set.append(row[:-1])
			y_test_set.append(row[-1])
		predicted = random_forest(train_set, test_set, trees_num, attributes_num, *args)

		actual = [row[-1] for row in fold]
		accuracy = calc_accuracy(actual, predicted)
		scores.append(accuracy)

		# Sklearn
		for row in train_set:
			X.append(row[:-1])
			y.append(row[-1])
		#res = clone(res)
		res.fit(X, y)
		accuracy_sklearn = res.score(test_set, y_test_set)*100
		scores_sklearn.append(accuracy_sklearn)
	return scores, scores_sklearn
 
def split_attribute(dataset, index, value):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 
def subset(dataset, ratio):
	sample = list()
	sample_num = round(len(dataset) * ratio)
	while len(sample) < sample_num:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

def gini_index(groups, classes):
	examples_num = int(sum([len(group) for group in groups]))
	gini = 0.0
	for group in groups:
		size = int(len(group))
		if size == 0:
			continue
		P, E = 0.0, 1.0
		for single_class in classes:
			P = [row[-1] for row in group].count(single_class) / size
			E -= P ** 2
		gini += (size / examples_num) * E
	return gini
 
def roulette_split(dataset, attributes_num, threshold):
	classes = list(set(row[-1] for row in dataset))

	index_list, val_list, group_list, fit = [], [], [], []
	attributes = list()
	while len(attributes) < attributes_num:
		index = randrange(len(dataset[0])-1)
		if index not in attributes:
			attributes.append(index)
	counter = 0
	for index in attributes:
		for row in dataset:
			groups = split_attribute(dataset, index, row[index])
			gini = gini_index(groups, classes)
			index_list.append(index)
			val_list.append(row[index])
			group_list.append(groups)
			fit.append(1-gini)
			counter += 1
	wheel_size = 0
	fit_args_sorted = np.argsort(fit)
	for i in range (0, int(threshold*counter)):
		fit[fit_args_sorted[i]] = 0
	for i in range (0, counter):
		wheel_size += fit[i]
	
	selection_probs = [fit[i]/wheel_size for i in range (0, counter)]

	winner = int(npr.choice(np.arange(counter), 1, p=selection_probs))
	return {'val':val_list[winner], 'groups':group_list[winner], 'index':index_list[winner]}

def terminal(group):
	results = [row[-1] for row in group]
	return max(results, key=results.count)

def one_class(node):
	res = True
	for i in range(0, len(node)):
		if node[0][-1] != node[i][-1]:
				res = False
				return res
	return res

def are_the_same(node):
	res = True
	for i in range(0, len(node[0])-1):
		for j in range(0, len(node)):
			for k in range(0, len(node)):
				if node[j][i] != node[k][i]:
					res = False
					return res
	return res

def split(node, attributes_num, depth, threshold):
	left, right = node['groups']
	del(node['groups'])

	if not left or not right:
		node['left'] = node['right'] = terminal(left + right)
		return

	if one_class(left) or are_the_same(left):
		node['left'] = terminal(left)
	else:
		node['left'] = roulette_split(left, attributes_num, threshold)
		split(node['left'], attributes_num, depth+1, threshold)

	if one_class(right) or are_the_same(right):
		node['right'] = terminal(right)
	else:
		node['right'] = roulette_split(right, attributes_num, threshold)
		split(node['right'], attributes_num, depth+1, threshold)
 

def build_tree(train, attributes_num, threshold):
	root = roulette_split(train, attributes_num, threshold)
	split(root, attributes_num, 1, threshold)
	return root
 

def predict(node, row):
	if row[node['index']] < node['val']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
 

def bagging_predict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)
 

def random_forest(train, test, attributes_num, trees_num, sample_size, threshold):
	trees = list()
	for i in range(trees_num):
		sample = subset(train, sample_size)
		tree = build_tree(sample, attributes_num, threshold)
		trees.append(tree)
	predictions = [bagging_predict(trees, row) for row in test]
	return(predictions)