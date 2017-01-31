from random import seed
from random import randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt

coefficients_global = [0, 0]
Ccoef = [0.0 for i in range(2)]
sumlist = list()


# Load a CSV file (The csv file containing data set)
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Find the min and max values for each column
def dataset_sum(dataset):
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        t_sum = sum(col_values)
        sumlist.append([t_sum])
    return sumlist


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] / minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate root mean squared error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        rmse = rmse_metric(actual, predicted)
        scores.append(rmse)
    return scores


# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return yhat


# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch, Ccoef):
    for epoch in range(n_epoch):
        for row in train:
            yhat = predict(row, Ccoef)
            error = yhat - row[-1]
            Ccoef[0] = Ccoef[0] - (l_rate * error)
            for i in range(len(row) - 1):
                Ccoef[i + 1] = Ccoef[i + 1] - (l_rate * error * row[i])
    # print(Ccoef[0], Ccoef[1], error)
    return Ccoef


# Linear Regression Algorithm With Stochastic Gradient Descent
def linear_regression_sgd(train, test, l_rate, n_epoch):
    predictions = list()
    coef = [0, 0]
    for i in range(10):
        coef = coefficients_sgd(train, l_rate, n_epoch, coef)
    print("coefficients", coef)
    coefficients_global[0] += coef[0]
    coefficients_global[1] += coef[1]
    for row in test:
        yhat = predict(row, coef)
        predictions.append(yhat)
    return (predictions)


# Linear Regression on wine quality dataset
seed(1)
# load and prepare data
filename = 'train.csv'
dataset = load_csv(filename)
dataset1 = load_csv(filename)
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
# normalize
sum_value = dataset_sum(dataset)
normalize_dataset(dataset, sum_value)
# evaluate algorithm
n_folds = 5
l_rate = 0.9
n_epoch = 50
scores = evaluate_algorithm(dataset, linear_regression_sgd, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean RMSE: %.3f' % (sum(scores) / float(len(scores))))
row_test = [0.0006471689306290525, 100000]
coefficients_global[0] = coefficients_global[0] / 5
coefficients_global[1] = coefficients_global[1] / 5
print("coefficients final", coefficients_global[0], coefficients_global[1])
print("prediction for 10000 sqft", predict(row_test, coefficients_global))
print(dataset[0])

ans = []
for row in dataset:
    plt.plot(row[0], row[1], 'ro')
plt.xlabel('Square feet')
plt.ylabel('Price')
# plt.axis([0,0.0012,0,0.0012])
c = 0
for row in dataset:
    ans = ((float(row[0]) * float(coefficients_global[1])) + float(coefficients_global[0]))
    plt.plot(row[0], ans, 'g^')
    c += 1

# plt.axis([minmax[0][0], minmax[0][1],minmax[1][0], minmax[1][1]])
plt.show()
