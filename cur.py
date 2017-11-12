from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix
import scipy.linalg as linalg
import csv
from math import sqrt
import os

sum_of_squares = 106257.295544
total_users = 943
total_movies = 1682
genre = 100


def get_data():
    matrix_a = np.zeros(shape=(total_users, total_movies))
    with open("u.csv", "r") as File:
        reader = csv.reader(File)
        for row in reader:
            row = row[0].split("\t")
            matrix_a[(int(row[0]) - 1), (int(row[1]) - 1)] = float(row[2])

    matrix_a = csr_matrix(matrix_a)
    return matrix_a


def split_matrix(matrix_a):
    transpose_a = matrix_a.transpose()
    matrix_u = np.dot(matrix_a, transpose_a)
    matrix_v = np.dot(transpose_a, matrix_a)
    vals_u, vecs_u = linalg.eigh(matrix_u)
    vals_v, vecs_v = linalg.eigh(matrix_v)
    eig_vals = list(vals_u)
    eig_vals.reverse()
    matrix_sigma = np.zeros(shape=(matrix_a.shape[0], matrix_a.shape[1]))
    for i in xrange(matrix_a.shape[0]):
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
    matrix_v = np.array(temp1).transpose()
    return matrix_u, matrix_sigma, matrix_v


def get_row_probability(matrix_a, row):
    temp_array = matrix_a[row].toarray()[0]
    squre_sum = 0
    for i in temp_array:
        squre_sum += i * i
    return squre_sum/sum_of_squares


def get_column_probability(matrix_a, column):
    temp_array = [l[0] for l in np.array(matrix_a[:,column].toarray())]
    square_sum = 0
    for i in temp_array:
        square_sum += i * i
    return square_sum/sum_of_squares


def pseudo_inverse(matrix):
    for i in xrange(matrix.shape[0]):
        for j in xrange(matrix.shape[1]):
            if not matrix[i, j] == 0:
                matrix[i, j] = 1/matrix[i, j]
    return matrix


def cur(matrix_a, replace=True):
    matrix_c = np.zeros(shape=(total_users, genre))
    row_probabilities = []
    for i in xrange(matrix_a.shape[0]):
        prob = get_row_probability(matrix_a, i)
        row_probabilities.append(prob)
    col_probabilities = []
    for i in xrange(matrix_a.shape[1]):
        prob = get_column_probability(matrix_a, i)
        col_probabilities.append(prob)
    matrix_c_list = np.random.choice(range(0, total_movies), genre,
                                     p=col_probabilities, replace=replace)
    i = 0
    for temp in matrix_c_list:
        try:
            temp_array = [l[0] for l in np.array(matrix_a[:, temp].toarray())]
            prob = (genre * col_probabilities[temp]) ** 0.5
            temp_array = np.array([j * prob for j in temp_array])
            matrix_c[:, i] = temp_array
            i += 1
        except IndexError as e:
            print e
    i = 0
    matrix_r = np.zeros(shape=(genre, total_movies))
    matrix_r_list = np.random.choice(range(0, total_users), genre,
                                     p=row_probabilities, replace=replace)
    done = []
    for temp in matrix_r_list:
        temp_array = list(matrix_a[temp].toarray()[0])
        prob = (genre * row_probabilities[temp]) ** 0.5
        if temp in done:
            prob = prob * 2
        temp_array = np.array([j * prob for j in temp_array])
        matrix_r[i] = temp_array
        i += 1
        done.append(temp)
    i = 0
    matrix_w = np.zeros((genre, genre))
    r = 0
    c = 0
    for i in matrix_r_list:
        for j in matrix_c_list:
            matrix_w[r][c] = matrix_a[i, j]
            c += 1
        c = 0
        r += 1
    matrix_x, matrix_sigma, matrix_y = split_matrix(matrix_w)
    matrix_sigma = pseudo_inverse(matrix_sigma.todense())
    matrix_sigma_square = matrix_sigma*matrix_sigma
    temp = matrix_y*matrix_sigma_square
    matrix_u = temp.dot(matrix_x.transpose())
    matrix_c = csr_matrix(matrix_c)
    matrix_r = csr_matrix(matrix_r)
    return matrix_c, matrix_u, matrix_r


def predict_singular(row, column, matrix_c, matrix_u, matrix_r):
    averages = np.load("averages.npy")
    row = int(row)
    column = int(column)
    temp_c = matrix_c[row]
    temp_u = temp_c * matrix_u
    temp_r = csr_matrix(temp_u) * matrix_r
    prediction = temp_r[0, int(column)] + averages[row]
    return prediction


def rmse(matrix_a, matrix_c, matrix_u, matrix_r):
    error = 0
    for user in xrange(total_users):
        user_vector = matrix_a[user, :].toarray()[0]
        temp_u = matrix_c[user]
        temp_s = temp_u * matrix_u
        temp_v = csr_matrix(temp_s) * matrix_r
        error_vector = temp_v - user_vector
        for i in xrange(error_vector.shape[1]):
            entry = error_vector[0, i]
            error += entry * entry
    error = error / (943 * 1682)
    return error ** 0.5


def brute_rmse(data="cur_predictions.npy"):
    test_ratings = np.load("test_ratings.npy")
    prediction = np.load(data)
    error = 0
    for i in xrange(20000):
        actual = test_ratings[i]
        predicted = int(prediction[i])
        error += (predicted - actual) ** 2
    return (error/20000)**0.5


def preprocess(matrix):
    averages = np.load("averages.npy")
    matrix = matrix.todense()
    for i in xrange(matrix.shape[0]):
        ave = averages[i]
        for j in xrange(matrix[i].shape[1]):
            if not matrix[i][0, j] == 0:
                matrix[i][0, j] -= ave
    return csr_matrix(matrix)


def save_predictions(matrix_c, matrix_u, matrix_r, db="cur_predictions.npy"):
    prediction = np.zeros(20000)
    test_rows = np.load("test_rows.npy")
    test_columns = np.load("test_columns.npy")
    for i in xrange(20000):
        predicted = predict_singular(test_rows[i], test_columns[i], matrix_c,
                                     matrix_u, matrix_r)
        prediction[i] = predicted
    np.save(db, prediction)


def precision_at_top_k(db="cur_predictions.npy", k=200):
    predicted = np.load(db)
    test_ratings = np.load("test_ratings.npy")
    relevant_entries = predicted.argsort()[::-1][0:k]
    threshold = int(predicted[relevant_entries[relevant_entries.shape[0] - 1]])
    print threshold
    relevant_count = 0
    for entry in relevant_entries:
        if test_ratings[entry] >= threshold:
            relevant_count += 1
    precision = relevant_count / 200
    return precision


def spearman_correlation(db="cur_predictions.npy"):
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
    matrix_a = preprocess(matrix_a)
    matrix_c, matrix_u, matrix_r = cur(matrix_a, replace=True)
    if "cur_predictions.npy" not in os.listdir("."):
        save_predictions(matrix_c, matrix_u, matrix_r, db="cur_predictions.npy")
    matrix_c, matrix_u, matrix_r = cur(matrix_a, replace=False)
    if "cur_mod_predictions.npy" not in os.listdir("."):
        save_predictions(matrix_c, matrix_u, matrix_r,
                         db="cur_mod_predictions.npy")
    print "rmse with repetition: ", brute_rmse()
    print "precision with repetition: ", precision_at_top_k()
    print "spearman correlation with repetition: ", spearman_correlation()
    print "rmse without repetition: ", brute_rmse(data="cur_mod_predictions.npy")
    print "precision without repetition: ", precision_at_top_k(
        db="cur_mod_predictions.npy")
    print "spearman correlation without repetition: ", spearman_correlation(
        db="cur_mod_predictions.npy")

if __name__ == '__main__':
    main()

# rmse with repetition:  3.20068742616
# precision with repetition:  0.0
# spearman correlation with repetition:  0.99999983671
# rmse without repetition:  47.5651500576
# precision without repetition:  0.0
# spearman correlation without repetition:  0.999965709662


