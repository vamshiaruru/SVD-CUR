from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix
import scipy.linalg as linalg
import csv
from random import randint
from math import sqrt

total_users = 943
total_movies = 1682
genre = 60


def get_data():
    matrix_a = np.zeros(shape=(total_users, total_movies))
    with open("u.csv", "r") as File:
        reader = csv.reader(File)
        for row in reader:
            row = row[0].split("\t")
            matrix_a[(int(row[0]) - 1), (int(row[1]) - 1)] = float(row[2])

    matrix_a = csr_matrix(matrix_a)
    return matrix_a


def svd(matrix_w):
    transpose_w = csr_matrix(matrix_w).transpose()
    matrix_x = matrix_w*transpose_w
    matrix_y = transpose_w*matrix_w
    vals_x, vecs_x = linalg.eigh(matrix_x)
    vals_y, vecs_y = linalg.eigh(matrix_y)
    eig_vals = list(vals_x)
    for i in range(len(eig_vals)):
        eig_vals[i] = abs(eig_vals[i])
    eig_vals.sort()
    eig_vals.reverse()
    matrix_sigma = np.zeros(shape=(genre, genre))
    for i in range(genre):
        matrix_sigma[i, i] = sqrt(eig_vals[i])
    for i in range(genre):
        if matrix_sigma[i][i] != 0:
            matrix_sigma[i, i] = (1 / eig_vals[i])
        else:
            matrix_sigma[i, i] = eig_vals[i]
    vecs_x = vecs_x.transpose()
    temp = list(vecs_x)
    temp.reverse()
    matrix_x = np.array(temp).transpose()

    vecs_y = vecs_y.transpose()
    temp = list(vecs_y)
    temp.reverse()
    matrix_y = np.array(temp).transpose()
    matrix_y.transpose()
    return matrix_x, matrix_sigma, matrix_y


def cur(matrix_a):
    matrix_c = np.zeros(shape=(total_users, genre))
    matrix_c_list = []
    for i in range(genre):
        temp = randint(0, total_movies)
        matrix_c_list.append(temp)
        matrix_c[:, i] = matrix_a[:, temp]
    matrix_r = np.zeros(shape=(genre, total_movies))
    matrix_r_list = []
    for i in range(genre):
        temp = randint(0, total_users)
        matrix_r_list.append(temp)
        matrix_r[i] = matrix_a[temp]
    matrix_w = [[0]*genre for i in range(genre)]
    r = 0
    c = 0
    for i in matrix_r_list:
        for j in matrix_c_list:
            matrix_w[r][c] = matrix_a[i][j]
            c += 1
        c = 0
        r += 1
    matrix_x, matrix_sigma, matrix_y = svd(matrix_w)
    matrix_sigma_square = matrix_sigma*matrix_sigma
    temp = (matrix_y.transpose())*matrix_sigma_square
    matrix_u = temp.dot(matrix_x.transpose())
    matrix_c = csr_matrix(matrix_c)
    matrix_r = csr_matrix(matrix_r)
    return matrix_c, matrix_u, matrix_r


def rmse(matrix_a, predict_a):
    error = 0
    for i in range(len(matrix_a)):
        for j in range(len(matrix_a[0])):
            print matrix_a[i][j], predict_a[i][j]
            error += (matrix_a[i][j] - predict_a[i][j])**2
    error = error / (943 * 19)
    return error ** 0.5


def main():
    matrix_a = get_data()

    # predict_a = cur(matrix_a)
    # print rmse(matrix_a, predict_a)


if __name__ == '__main__':
    main()