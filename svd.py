"""
A module to perform SVD decompostion on a given training dataset.
"""
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
    """
     A function to read data from test file, and return a sparse matrix.
    :return: A sparse user - movie rating matrix
    """
    matrix_a = np.zeros(shape=(total_users, total_movies))
    with open("u.csv", "r") as File:
        reader = csv.reader(File)
        for row in reader:
            row = row[0].split("\t")
            matrix_a[(int(row[0]) - 1), (int(row[1]) - 1)] = float(row[2])
    return csr_matrix(matrix_a)


def split_matrix(matrix_a):
    """
    A function to perform svd decomposition on the matrix given to it.The
    algorithm to peform the SVD decomposition for a matrix A is as follows:
        - Calculate x = A * A.transpose()
        - find all eigen values of x, and find corresponding eigen vectors
        - Construct matrix U by arragning eigen vectors, in decreasing order
          of their eigen value magnitude, as columns.
        - Similarly construct V from matrix y = A.transpose() * A
        - Construct the matrix sigma by placing square roots of eigen values
          of x in decreasing order of their magnitudes along principal diagonal
          of a zero square matrix of appropriate dimensions
        - return U, Sigma, V
    :param matrix_a: (scipy.sparse.csr_matrix) A sparse matrix that is to be
    decomposed via svd
    :return: the three matrices, U, Sigma, and V that are the result of SVD
    decomposition
    """
    transpose_a = csr_matrix.transpose(matrix_a)
    matrix_u = np.dot(matrix_a, transpose_a)
    matrix_v = np.dot(transpose_a, matrix_a)
    # scipy.linalg.eigh() is a function that returns two values - a list of of
    # all its eigen values, and corresponding, normalized eigen vectors
    # arranged as columns in increasing order of the magnitudes of corresponding
    # eigen values. Eigen values are returned in increasing order of their
    # magnitude.
    vals_u, vecs_u = linalg.eigh(matrix_u.todense())
    vals_v, vecs_v = linalg.eigh(matrix_v.todense())
    eig_vals = list(vals_u)
    # we need eigen values in decreasing order of their magnitudes.
    eig_vals.reverse()
    # construct matrix sigma.
    matrix_sigma = np.zeros(shape=(total_users, total_movies))
    for i in xrange(total_users):
        try:
            matrix_sigma[i, i] = sqrt(eig_vals[i])
        except ValueError:
            matrix_sigma[i, i] = sqrt(-1*eig_vals[i])
        except IndexError as e:
            print e
    matrix_sigma = csr_matrix(matrix_sigma)
    # we need to reverse the order of columns because we need them arranged in
    #  decreasing order of corresponding eigen values, not increasing order.
    # to accomplish this, we transpose our numpy.ndarray, convert to a list
    # of lists, perform list reverse operation, construct a numpy.nd array
    # from the obtained list and then transpose it back.
    # >>> [1,2]
    #     [3,4]
    # >>> (after step 1)[1, 3]
    #                   [2, 4]
    # >>> (after list reverse) [2 ,4]
    #                          [1, 3]
    # >>> after final transpose, [2, 1]
    # >>>                        [4, 3]
    vecs_u = vecs_u.transpose()
    temp = list(vecs_u)
    temp.reverse()
    matrix_u = np.array(temp).transpose()
    # do the same for matrix_v
    vecs_v = vecs_v.transpose()
    temp1 = list(vecs_v)
    temp1.reverse()
    matrix_v = np.array(temp1)
    return matrix_u, matrix_sigma, matrix_v


def rmse(matrix_a, matrix_u, matrix_sigma, matrix_v):
    """
    A function to calculate rmse on original matrix.
    :param matrix_a: (scipy.sparse.csr_matrix) the original matrix which we
    have decomposed.
    :param matrix_u: (numpy.ndarray) U in SVD decomposition of matrix_a
    :param matrix_sigma: (numpy.ndarray) sigma in SVD decomposition of matrix_a
    :param matrix_v: (numpy.ndarray) V in SVD decomposition of matrix_a
    :return: (double) rmse error
    """
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
    """
    A function to calculate energy of a SVD decomposition.
    Energy of a SVD decomposition is sum of squares of diagonal values in
    matrix sigma of the decomposition
    :param matrix_sigma: (numpy.ndarray) The matrix sigma of svd decomposition
    :return: (double) the energy of svd decomposition
             (list) list of eigen values (diagonal elements) of matrix sigma
    """
    energy = 0
    eigen_list = []
    for i in xrange(matrix_sigma.shape[0]):
        eigen = matrix_sigma[i, i]
        eigen_list.append(eigen)
        energy += eigen ** 2
    return energy, eigen_list


def minimize_sigma(matrix_u, matrix_sigma, matrix_v):
    """
    To make our decomposition more efficent we can minimize our matrices until
    the energy of our decomposition is 90% of energy of original
    decomposition. This function accompolishes that.
    We minimize our matrices the following way:
        - get energy of original matrix as energy
        - iteratively eliminate the least eigen value from eigen value list
          until the energy of new eigen value list is 90% of original energy
        - eliminate correspnding columns from matrix_sigma, and corresponding
          columns from matrix_u, corresponding rows from matrix_v
        - return the three minimized matrices
    :param matrix_u: (numpy.ndarray) U in SVD decomposition
    :param matrix_sigma: (numpy.ndarray) sigma in SVD decomposition
    :param matrix_v: (numpy.ndarray) V in SVD decomposition
    :return: the three matrices minimized.
    """
    energy, eigen_list = get_energy(matrix_sigma)
    new_energy = energy
    count = 0
    while new_energy > 0.9 * energy:
        count += 1
        del(eigen_list[len(eigen_list) - 1])
        new_energy = sum([i * i for i in eigen_list])
    final_length = len(eigen_list)
    count = -1 * count
    # eliminate columns corresponding to deleted eigen values from matrix_sigma
    matrix_sigma = np.delete(matrix_sigma.todense().transpose(),
                             np.s_[final_length:], 0).transpose()
    # eliminate corresponding rows as well, to get square matrix_sigma again
    matrix_sigma = np.delete(matrix_sigma, np.s_[final_length:], 0)
    # eliminate corresponding rows from transpose of matrix u and then perform
    #  transpose again to get original matrix back
    matrix_u = np.delete(matrix_u.transpose(), np.s_[count:], 0).transpose()
    count = matrix_sigma.shape[0]
    # delete rows from matrix v
    matrix_v = np.delete(matrix_v, np.s_[count:], 0)
    return matrix_u, matrix_sigma, matrix_v


def preprocess(matrix, store=True):
    """
    A function to preprocess the matrix. In this step, from every user's
    rating, we subtract that user's average rating.
    :param matrix: (scipy.sparse.csr_matrix) the matrix to be processed
    :param store: (Boolean) if store == true, this function also stores
    averages of each user rating in a numpy array. Otherwise, it just performs
    pre-processing of matrix a
    :return: (scipy.sparse.csr_matrix) preprocessed matrix
    """
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
    """
    calculates average of non zero values of vector
    - >>> vector = [1,1,1,0,0,0]
    - >>> average(vector) = (3/3) = 1
    :param vector: (numpy.ndarray) The array whose average is to be calculated
    :return: (double) average of vector
    """
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
    """
    This function predicts the rating a user gives to a movie, based on the
    matrices obtained from SVD decomposition.
    :param row: (int) Id of user
    :param column:(int) Id of movie, whose rating from user we have to determine
    :param matrix_u:(scipy.sparse.csr_matrix) U in SVD decopmosition
    :param matrix_sigma: (np.ndarray) sigma in SVD decomposition
    :param matrix_v:(scipy.sparse.csr_matrix) v in svd decomposition
    :param averages: (numpy.ndarray) array which contains average rating of each
    user.
    :return: (int) predicted rating
    """
    row = int(row)
    column = int(column)
    # for a user 'row' all his predictions can be calculated by taking the row
    #  corresponding to the user from matrix_u as r and performing r
    #  * sigma * v.transpose(). Now rating for a particular movie can be
    #  obtained by r[movie]
    temp_u = matrix_u[row]
    temp_s = temp_u * matrix_sigma
    temp_v = csr_matrix(temp_s) * matrix_v
    prediction = temp_v[0, int(column)] + averages[row]
    return prediction


def save_test_data(testfile):
    """
    A function to read the testfile and store users and movies, and their
    corresponding ratings into numpy arrays.
    :param testfile: (String) the name of test file.
    :return: None
    """
    test_rows = np.zeros(20000)
    test_columns = np.zeros(20000)
    test_ratings = np.zeros(20000)
    count = 0
    with open(testfile, "r") as f:
        for line in f:
            line = [int(word) for word in line.strip().split(",")]
            test_rows[count] = int(line[0]) - 1
            test_columns[count] = int(line[1]) - 1
            test_ratings[count] = int(line[2])
            count += 1
    np.save("test_rows.npy", test_rows)
    np.save("test_columns.npy", test_columns)
    np.save("test_ratings.npy", test_ratings)


def save_predictions(matrix_u, matrix_sigma, matrix_v, db="predictions.npy"):
    """
    To not run the entire algorithm every time, we can save our predictions
    for a given test data. This function does that.
    :param matrix_u:(numpy.ndarray) U in SVD decompoistion of given matrix
    :param matrix_sigma: (numpy.ndarray) Sigma in SVD decomposition of given
    matrix
    :param matrix_v: (numpy.ndarray) V in SVD decomposition of given matrix
    :param db: (String) name of numpy array containing actual predictions.
    :return:
    """
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
    """
    Calculate rmse error over test data. The test data and training data are
    disjoint so as to better evaluate the recommender system. Rmse error is
    calculated by the formula, sqrt(summation( (predicted - actual) ^ 2)/n
    :param data: (String) the name of the numpy array that contains our
    predictions.
    :return: (double) rmse error
    """
    test_ratings = np.load("test_ratings.npy")
    prediction = np.load(data)
    rmse = 0
    for i in xrange(20000):
        actual = test_ratings[i]
        predicted = prediction[i]
        rmse += (predicted - actual) ** 2
    return (rmse/20000)**0.5


def precision_at_top_k(db="predictions.npy", k=300):
    """
    A function to calculate precision for top_k ratings. we calculate the
    precision the following way:
    - Get top K entries from our predicted values
    - Get threshold for these top K entries
    - Get the number of entries amongst these K entries whose actual rating is
      greater than threshold as c
    - precision = c/k
    :param db: (String) name of the numpy array where our predictions are stored.
    :param k: (int) k in top K entries
    :return: (double) precision
    """
    predicted = np.load(db)
    test_ratings = np.load("test_ratings.npy")
    relevant_entries = predicted.argsort()[::-1][0:k]
    threshold = int(predicted[relevant_entries[relevant_entries.shape[0] - 1]])

    relevant_count = 0
    for entry in relevant_entries:
        if test_ratings[entry] >= threshold:
            relevant_count += 1
    precision = relevant_count / k
    return precision


def spearman_correlation(db="predictions.npy"):
    """
    Function to calculate spearman correlation.
    It is calculated by the following formula:
        1 - ((6 * sum((predicted - actual) ^ 2)/( n ( n^2 - 1)), where n is
        the number of entries we are checking against.
    Spearman correaltion gives how similar two vectors are. How strongly they
    are correlated is determined by the magnitude of correlation. The closer
    they are to 1, the stronger they are related.
    :param db: (String) name of the numpy array where our predictions are stored
    :return: (double) spearman correaltion
    """
    predicted = np.load(db)
    test_ratings = np.load("test_ratings.npy")
    d = 0
    n = test_ratings.shape[0]
    for i in xrange(n):
        d += (predicted[i] - test_ratings[i]) ** 2
    coeff = (6 * d) / (n * (n ** 2 - 1))
    return 1 - coeff


def main():
    """
    The main function.
    It does the following:
        - reads data from training file and converts it to a user movie matrix
        - performs preprocessing
        - performs SVD decomposition, and calculates rmse, precision on top k
          and spearman correlation
        - Performs minimzation until 90% of energy, and calculates them again.
    :return: None
    """
    matrix_a = get_data()
    matrix_a, averages = preprocess(matrix_a)
    print "starting to split"
    matrix_u, matrix_sigma, matrix_v = split_matrix(matrix_a)
    save_test_data("u1.test")
    if "predictions.npy" not in os.listdir("."):
        print "here"
        save_predictions(matrix_u, matrix_sigma, matrix_v)
    print "rmse before minimimization", brute_rmse()
    print "precision at top k before minimization", precision_at_top_k(
        db="predictions.npy")
    print "spearman correlation before minimzation", spearman_correlation(
        db="predictions.npy")
    matrix_u, matrix_sigma, matrix_v = minimize_sigma(matrix_u, matrix_sigma,
                                                      matrix_v)
    if "min_predictions.npy" not in os.listdir("."):
        print "here"
        save_predictions(matrix_u, matrix_sigma, matrix_v,
                         db="min_predictions.npy")
    print "rmse after minimisation", brute_rmse(data="min_predictions.npy")
    print "precision at top k after minimization", precision_at_top_k(
        db="min_predictions.npy")
    print "spearman correlation after minimization", spearman_correlation(
        db="min_predictions.npy")

if __name__ == "__main__":
    main()

# rmse before minimimization 1.2227316644
# precision at top k before minimization 0.686666666667
# spearman correlation before minimzation 0.999999977574
# rmse after minimisation 1.21357348695
# precision at top k after minimization 0.673333333
# spearman correlation after minimization 0.999999977909

