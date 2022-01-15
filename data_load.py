import csv

def read_file(file):
    with open(file, 'r') as myfile:
        data = myfile.readlines()
    return data

def load_data(data, var):
    dataset = list()
    for row in data:
        row = row.rstrip("\n")
        if var == 1:
            dataset.append(row.split(","))
        if var == 2:
            dataset.append(row.split(" "))
        if var == 3:
            dataset.append(row.split(";"))
    return dataset


def process_dat_data(data):
    dataset = list()
    for row in data:
        row_temp = list()
        for i in range(1, len(row)):
            res = row[i].find(':')
            row_temp.append(row[i][res+1:])
        row_temp.append(row[0])
        dataset.append(row_temp)
    return dataset

def str2int(dataset):
    for i in range(0, len(dataset[0])):
        class_values = [row[i] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for j, value in enumerate(unique):
            lookup[value] = float(j)
        for row in dataset:
            row[i] = lookup[row[i]]

def str2float(dataset):
    for row in dataset:
        for i in range(0, len(row)):
            row[i] = float(row[i])
    return dataset

def write_data(data):
    with open("tests_res.csv", 'a') as file:
        f = csv.writer(file, delimiter =',', lineterminator='\n')
        f.writerows(data)