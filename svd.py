from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix
import scipy.linalg as linalg
import csv
import os
from math import sqrt

total_users = 943
total_movies = 1682


def get_data():
    matrix_a = np.zeros(shape=(total_users, total_movies))
    with open("u.csv", "r") as File:
        reader = csv.reader(File)
        for row in reader:
            row = row[0].split("\t")
            matrix_a[(int(row[0]) - 1), (int(row[1]) - 1)] = float(row[2])
    return csr_matrix(matrix_a)


def split_matrix(matrix_a):
    transpose_a = csr_matrix.transpose(matrix_a)
    matrix_u = np.dot(matrix_a, transpose_a)
    matrix_v = np.dot(transpose_a, matrix_a)
    vals_u, vecs_u = linalg.eigh(matrix_u.todense())
    vals_v, vecs_v = linalg.eigh(matrix_v.todense())
    eig_vals = list(vals_u)
    eig_vals.reverse()
    matrix_sigma = np.zeros(shape=(total_users, total_movies))
    for i in xrange(total_users):
        try:
            matrix_sigma[i, i] = sqrt(eig_vals[i])
        except ValueError:
            matrix_sigma[i, i] = sqrt(-1*eig_vals[i])
        except IndexError as e:
            print e
    matrix_sigma = csr_matrix(matrix_sigma)
    vecs_u = vecs_u.transpose()
    temp = list(vecs_u)
    temp.reverse()
    matrix_u = np.array(temp).transpose()

    vecs_v = vecs_v.transpose()
    temp1 = list(vecs_v)
    temp1.reverse()
    matrix_v = np.array(temp1)
    return matrix_u, matrix_sigma, matrix_v


def rmse(matrix_a, matrix_u, matrix_sigma, matrix_v):
    error = 0
    for user in xrange(total_users):
        user_vector = matrix_a[user, :].toarray()[0]
        temp_u = matrix_u[user]
        temp_s = temp_u * matrix_sigma
        temp_v = csr_matrix(temp_s) * matrix_v
        error_vector = temp_v - user_vector
        for i in error_vector[0]:
            error += i*i
    error = error / (943 * 1682)
    return error ** 0.5


def get_energy(matrix_sigma):
    energy = 0
    eigen_list = []
    for i in xrange(matrix_sigma.shape[0]):
        eigen = matrix_sigma[i, i]
        eigen_list.append(eigen)
        energy += eigen ** 2
    return energy, eigen_list


def minimize_sigma(matrix_u, matrix_sigma, matrix_v):
    energy, eigen_list = get_energy(matrix_sigma)
    new_energy = energy
    count = 0
    while new_energy > 0.9 * energy:
        count += 1
        del(eigen_list[len(eigen_list) - 1])
        new_energy = sum([i * i for i in eigen_list])
    final_length = len(eigen_list)
    count = -1 * count
    matrix_sigma = np.delete(matrix_sigma.todense().transpose(),
                             np.s_[final_length:], 0).transpose()
    matrix_sigma = np.delete(matrix_sigma, np.s_[final_length:], 0)
    matrix_u = np.delete(matrix_u.transpose(), np.s_[count:], 0).transpose()
    count = matrix_sigma.shape[0]
    matrix_v = np.delete(matrix_v, np.s_[count:], 0)
    return matrix_u, matrix_sigma, matrix_v


def preprocess(matrix, store=True):
    if store:
        averages = np.zeros(matrix.shape[0])
        matrix = matrix.todense()
        for i in xrange(matrix.shape[0]):
            ave = average(matrix[i])
            averages[i] = ave
            for j in xrange(matrix[i].shape[1]):
                if not matrix[i][0, j] == 0:
                    matrix[i][0, j] -= ave
        np.save("averages.npy", averages)
        return csr_matrix(matrix), averages
    else:
        averages = np.load("averages.npy")
        matrix = matrix.todense()
        for i in xrange(matrix.shape[0]):
            ave = averages[i]
            for j in xrange(matrix[i].shape[1]):
                if not matrix[i][0, j] == 0:
                    matrix[i][0, j] -= ave
        return csr_matrix(matrix), averages


def average(vector):
    sum = 0
    count = 0
    for i in xrange(vector.shape[1]):
        if not vector[0, i] == 0:
            sum += vector[0, i]
            count += 1
    if count == 0:
        return 0
    return sum/count


def predict_singular(row, matrix_u, matrix_sigma, matrix_v, averages, column):
    row = int(row)
    column = int(column)
    temp_u = matrix_u[row]
    temp_s = temp_u * matrix_sigma
    temp_v = csr_matrix(temp_s) * matrix_v
    prediction = temp_v[0, int(column)] + averages[row]
    return prediction


def save_test_data(testfile):
    test_rows = np.zeros(20000)
    test_columns = np.zeros(20000)
    test_ratings = np.zeros(20000)
    count = 0
    with open(testfile, "r") as f:
        for line in f:
            line = [int(word) for word in line.strip().split(",")]
            test_rows[count] = int(line[0])
            test_columns[count] = int(line[1])
            test_ratings[count] = int(line[2])
            count += 1
    np.save("test_rows.npy", test_rows)
    np.save("test_columns.npy", test_columns)
    np.save("test_ratings.npy", test_ratings)


def save_predictions(matrix_u, matrix_sigma, matrix_v, db="predictions.npy"):
    prediction = np.zeros(20000)
    test_rows = np.load("test_rows.npy")
    test_columns = np.load("test_columns.npy")
    averages = np.load("averages.npy")
    for i in xrange(20000):
        predicted = predict_singular(test_rows[i], matrix_u, matrix_sigma,
                                     matrix_v, averages, test_columns[i])
        prediction[i] = predicted
    np.save(db, prediction)


def brute_rmse(data="predictions.npy"):
    test_ratings = np.load("test_ratings.npy")
    prediction = np.load(data)
    rmse = 0
    for i in xrange(20000):
        actual = test_ratings[i]
        predicted = int(prediction[i])
        rmse += (predicted - actual) ** 2
    return (rmse/20000)**0.5


def precision_at_top_k(db="predictions.npy", k=200):
    predicted = np.load(db)
    test_ratings = np.load("test_ratings.npy")
    relevant_entries = predicted.argsort()[::-1][0:k]
    threshold = int(predicted[relevant_entries[relevant_entries.shape[0] - 1]])
    relevant_count = 0
    for entry in relevant_entries:
        if test_ratings[entry] >= threshold:
            relevant_count += 1
    precision = relevant_count / 200
    return precision


def spearman_correlation(db="predictions.npy"):
    predicted = np.load(db)
    test_ratings = np.load("test_ratings.npy")
    d = 0
    n = test_ratings.shape[0]
    for i in xrange(n):
        d += (predicted[i] - test_ratings[i]) ** 2
    coeff = (6 * d) / (n * (n ** 2 - 1))
    return 1 - coeff


def main():
    matrix_a = get_data()
    matrix_a, averages = preprocess(matrix_a)
    print "starting to split"
    matrix_u, matrix_sigma, matrix_v = split_matrix(matrix_a)
    if "predictions.npy" not in os.listdir("."):
        save_predictions(matrix_u, matrix_sigma, matrix_v)
    print "rmse before minimimization", brute_rmse()
    print "precision at top k before minimization", precision_at_top_k(
        db="predictions.npy")
    print "spearman correlation before minimzation", spearman_correlation(
        db="predictions.npy")
    matrix_u, matrix_sigma, matrix_v = minimize_sigma(matrix_u, matrix_sigma,
                                                      matrix_v)
    if "min_predictions.npy" not in os.listdir("."):
        save_predictions(matrix_u, matrix_sigma, matrix_v,
                         db="min_predictions.npy")
    print "rmse after minimisation", brute_rmse(data="min_predictions.npy")
    print "precision at top k after minimization", precision_at_top_k(
        db="min_predictions.npy")
    print "spearman correlation after minimization", spearman_correlation(
        db="min_predictions.npy")

if __name__ == "__main__":
    main()

# rmse before minimimization 1.38522561339
# precision at top k before minimization 0.625
# spearman correlation before minimzation 0.999999974957
# rmse after minimisation 1.38518951772
# precision at top k after minimization 0.64
# spearman correlation after minimization 0.999999975063

