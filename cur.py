"""
A module to implement CUR decomposition
"""
from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix
import scipy.linalg as linalg
import csv
from math import sqrt
import os
import time

# here sum_of_squares is the sum of squares of all values after preprocessing the
# matrix, genre represent the number of rows and columns we are choosing randomly
sum_of_squares = 106257.295544
total_users = 943
total_movies = 1682
genre = 100


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

    matrix_a = csr_matrix(matrix_a)
    return matrix_a


def split_matrix(matrix_a):
    """
    A function to perform svd decomposition on the matrix given to it. It is
    similar to the function of same name in svd.py
    :param matrix_a: (scipy.sparse.csr_matrix) A sparse matrix that is to be
    decomposed via svd
    :return: the three matrices, U, Sigma, and V that are the result of SVD
    decomposition
    """
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
    """
    In CUR decomposition we choose rows and columns based on the probability
    function give by, p = sum_of_squares (of row or column) / sum_of_squares (of
    entire matrix). This function returns that probablity for a row.
    :param matrix_a:(scipy.sparse.csr_matrix) The actual user - movie -rating
    matrix that is to be decomposed.
    decomposed
    :param row: (int) the row number whose probablity is to be calculated
    :return: (double) probability of that row.
    """
    # get the required row
    temp_array = matrix_a[row].toarray()[0]
    # stores the sum of squares of elements of temp_array
    squre_sum = 0
    for i in temp_array:
        squre_sum += i * i
    return squre_sum/sum_of_squares


def get_column_probability(matrix_a, column):
    """
    In CUR decomposition we choose rows and columns based on the probability
    function give by, p = sum_of_squares (of row or column) / sum_of_squares (of
    entire matrix). This function returns that probablity for a column.
    :param matrix_a: (scipy.sparse.csr_matrix) The actual user - movie -rating
    matrix that is to be decomposed.
    :param column: (int) the column number whose probablity is to be calculated
    :return: (double) probability of that column.
    """
    temp_array = [l[0] for l in np.array(matrix_a[:,column].toarray())]
    square_sum = 0
    for i in temp_array:
        square_sum += i * i
    return square_sum/sum_of_squares


def pseudo_inverse(matrix):
    """
    Calculate pseudo inverse of the matrix given. A pseudo inverse of a matrix is
    calculated by finding arithmetic multiplicative inverse of all non zero
    elements of the matrix
    :param matrix: (scipy.sparse.csr_matrix) The actual user - movie -rating
    matrix that is to be decomposed.
    :return: (scipy.sparse.csr_matrix) pseudo inverse of given matrix.
    """
    for i in xrange(matrix.shape[0]):
        for j in xrange(matrix.shape[1]):
            if not matrix[i, j] == 0:
                matrix[i, j] = 1/matrix[i, j]
    return matrix


def cur(matrix_a, replace=True):
    """
    Perform CUR decomposition for a given matrix.
    The algorithm is as follows:
    - randomly sample "genre" number of rows and columns of matrix_a as
    matrix_r, matrix_c
    - Find intersection of the above two matrices as matrix_w
    - perform svd decoposition of matrix_w, to get x, matrix_sigma, y
    - get pseudo inverse of matrix_sigma
    - matrix_u = y * matrix_sigma ^ 2 * x.transpose()
    - return matrix_c, matrix_u, matrix_r
    :param matrix_a: (scipy.sparse.csr_matrix) The actual user - movie -rating
    matrix that is to be decomposed.
    :param replace: (boolean) replace = True => rows and columns are chosen
    with repetition. replace = False => rows and columns are chosen without
    repetition
    :return: matrix_c, matrix_u, matrix_r, which constitute to CUR
    decomposition of the required matrix.
    """
    matrix_c = np.zeros(shape=(total_users, genre))
    row_probabilities = []
    # get probabilities of all rows.
    for i in xrange(matrix_a.shape[0]):
        prob = get_row_probability(matrix_a, i)
        row_probabilities.append(prob)
    col_probabilities = []
    # get probabilities of all columns
    for i in xrange(matrix_a.shape[1]):
        prob = get_column_probability(matrix_a, i)
        col_probabilities.append(prob)
    # np.random.choice() is a function that is used to randomly choose a given
    # number of values from a given list according to the probabilities
    # specified. replace = True => with repetition, replace = False => without
    # repetition.
    matrix_c_list = np.random.choice(range(0, total_movies), genre,
                                     p=col_probabilities, replace=replace)
    i = 0
    # from randomly chosen list of columns, construct matrix_c
    for temp in matrix_c_list:
        # get the column from matrix_a
        temp_array = [l[0] for l in np.array(matrix_a[:, temp].toarray())]
        # here, prob is the scaling factor = sqrt(genre * probability)
        prob = (genre * col_probabilities[temp]) ** 0.5
        # scale the column
        temp_array = np.array([j * prob for j in temp_array])
        # replace the column in matrix_c with scaled column.
        matrix_c[:, i] = temp_array
        i += 1
    i = 0
    matrix_r = np.zeros(shape=(genre, total_movies))
    # similarly build matrix_r
    matrix_r_list = np.random.choice(range(0, total_users), genre,
                                     p=row_probabilities, replace=replace)
    for temp in matrix_r_list:
        temp_array = list(matrix_a[temp].toarray()[0])
        prob = (genre * row_probabilities[temp]) ** 0.5
        temp_array = np.array([j * prob for j in temp_array])
        matrix_r[i] = temp_array
        i += 1
    i = 0
    # now construct matrix_w as intersection of matrix_u and matrix_r. By
    # intersection, choose the elements (row, column) from matrix_a where row
    # belongs to matrix_r_list, column belongs to matrix_c_list
    matrix_w = np.zeros((genre, genre))
    r = 0
    c = 0
    for i in matrix_r_list:
        for j in matrix_c_list:
            matrix_w[r][c] = matrix_a[i, j]
            c += 1
        c = 0
        r += 1
    # perform svd decomposition for matirx_w
    matrix_x, matrix_sigma, matrix_y = split_matrix(matrix_w)
    # pseudo inverse for matrix_sigma
    matrix_sigma = pseudo_inverse(matrix_sigma.todense())
    matrix_sigma_square = matrix_sigma*matrix_sigma
    temp = matrix_y*matrix_sigma_square
    # calculate matrix_u and make them all sparse
    matrix_u = temp.dot(matrix_x.transpose())
    matrix_c = csr_matrix(matrix_c)
    matrix_r = csr_matrix(matrix_r)
    return matrix_c, matrix_u, matrix_r


def predict_singular(row, column, matrix_c, matrix_u, matrix_r):
    """
    This function predicts the rating a user gives to a movie, based on the
    matrices obtained from CUR decomposition
    :param row: (int) Id of user
    :param column:(int) Id of movie, whose rating from user we have to determine
    :param matrix_c:(scipy.sparse.csr_matrix) C in CUR decopmosition
    :param matrix_u: (np.ndarray) U in CUR decomposition
    :param matrix_r:(scipy.sparse.csr_matrix) R in CUR decomposition
    :return: (int) predicted rating
    """
    now = time.clock()
    # averages.npy contains precalculated averages of all the rows of matrix_a
    averages = np.load("averages.npy")
    # if by chance row, column are not integers, convert them to integers
    row = int(row)
    column = int(column)
    # for a user 'row' all his predictions can be calculated by taking the row
    #  as r and performing r * u * R. Now rating for a particular movie can be
    # obtained by r[movie]
    temp_c = matrix_c[row]
    temp_u = temp_c * matrix_u
    temp_r = csr_matrix(temp_u) * matrix_r
    # since we are adjusting the matrix by subtracting average of a user's
    # rating from every movie in that row, we have to adjust our prediction as
    #  well
    prediction = temp_r[0, int(column)] + averages[row]
    # adujst predictions, so that it remains bound between 0 and 5
    if prediction >= 5:
        prediction = 5
    elif prediction <= 0:
        prediction = 0
    else:
        prediction = int(prediction)
    print time.clock() - now
    return prediction


def rmse(matrix_a, matrix_c, matrix_u, matrix_r):
    """
    A function to calculate rmse on original matrix.
    :param matrix_a: (scipy.sparse.csr_matrix) the original matrix which we
    have decomposed.
    :param matrix_c: (scipy.sparse.csr_matrix) C in CUR decompostion of matrix_a
    :param matrix_u: (numpy.ndarray) U in CUR decomposition of matrix_a
    :param matrix_r: (scipy.sparse.csr_matrix) R in CUR decompostion of matrix_a
    :return: (double) rmse error
    """
    error = 0
    # loop through all the users and calculate errors
    for user in xrange(total_users):
        # calculate predicted rating the same way we have predicted individual
        #  rating
        user_vector = matrix_a[user, :].toarray()[0]
        temp_u = matrix_c[user]
        temp_s = temp_u * matrix_u
        # temp_s is a np.ndarray, so convert it to scipy.sparse.csr_matrix
        # beforehand.
        temp_v = csr_matrix(temp_s) * matrix_r
        error_vector = temp_v - user_vector
        for i in xrange(error_vector.shape[1]):
            entry = error_vector[0, i]
            error += entry * entry
    # calculate rmse error.
    error = error / (943 * 1682)
    return error ** 0.5


def brute_rmse(data="cur_predictions.npy"):
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
    error = 0
    for i in xrange(20000):
        actual = test_ratings[i]
        predicted = int(prediction[i])
        error += (predicted - actual) ** 2
    return (error/prediction.shape[0])**0.5


def preprocess(matrix):
    """
    A function to preprocess the matrix. In this step, from every user's
    rating, we subtract that user's average rating.
    :param matrix: (scipy.sparse.csr_matrix) the matrix to be processed
    :return: (scipy.sparse.csr_matrix) preprocessed matrix
    """
    averages = np.load("averages.npy")
    matrix = matrix.todense()
    for i in xrange(matrix.shape[0]):
        ave = averages[i]
        for j in xrange(matrix[i].shape[1]):
            if not matrix[i][0, j] == 0:
                matrix[i][0, j] -= ave
    return csr_matrix(matrix)


def save_predictions(matrix_c, matrix_u, matrix_r, db="cur_predictions.npy"):
    """
    To not run the entire algorithm every time, we can save our predictions
    for a given test data. This function does that.
    :param matrix_c: (scipy.sparse.csr_matrix) C in CUR decomposition
    :param matrix_u: (numpy.ndarray) U in CUR decomposition
    :param matrix_r: (scipy.sparse.matrix) R in CUR decompositon
    :param db: (String) name of the numpy array where we need to store our
    predictions.
    :return: None
    """
    prediction = np.zeros(20000)
    # test_rows.npy, #test_columns.npy have the user and movie ids for which
    # have to predict ratings.
    test_rows = np.load("test_rows.npy")
    test_columns = np.load("test_columns.npy")
    for i in xrange(20000):
        predicted = predict_singular(test_rows[i], test_columns[i], matrix_c,
                                     matrix_u, matrix_r)
        prediction[i] = predicted
    np.save(db, prediction)


def precision_at_top_k(db="cur_predictions.npy", k=300):
    """
    A function to calculate precision for top_k ratings. we calculate the
    precision the following way:
    - Get top K entries from our predicted values
    - Get threshold for these top K entries
    - Get the number of entires amongst these K entries whose actual rating is
      greator than threshold as c
    - precision = c/k
    :param db: (String) name of the numpy array where our predictions are stored.
    :param k: (int) k in top K entries
    :return: (double) precision
    """
    predicted = np.load(db)
    # test_ratings.npy array contains the given test_ratings. we calculate
    # error against these ratings.
    test_ratings = np.load("test_ratings.npy")
    relevant_entries = predicted.argsort()[::-1][0:k]
    # threshold is the rating of the last entry in our relevant entries
    threshold = int(predicted[relevant_entries[relevant_entries.shape[0] - 1]])
    relevant_count = 0
    for entry in relevant_entries:
        if test_ratings[entry] >= threshold:
            relevant_count += 1
    precision = relevant_count / k
    return precision


def spearman_correlation(db="cur_predictions.npy"):
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
    The main function. It does the following:
        - get data from the training set
        - preprocess the matrix
        - perform CUR decomposition, first with repetition
        - if the predcitions were not saved before, saves them now
        - performs CUR decomposition, without repetition now
        - If the predictions were not saved before, saves them now
        - calculates rmse, spearman correlation and precision on top K ratings
          for both the cases and prints them
    :return:
    """
    matrix_a = get_data()
    matrix_a = preprocess(matrix_a)
    matrix_c, matrix_u, matrix_r = cur(matrix_a, replace=False)
    if "cur_predictions.npy" not in os.listdir("."):
        print "saving predictions"
        save_predictions(matrix_c, matrix_u, matrix_r, db="cur_predictions.npy")
    matrix_c, matrix_u, matrix_r = cur(matrix_a, replace=True)
    if "cur_mod_predictions.npy" not in os.listdir("."):
        print "saving modified cur predictions"
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

# rmse with repetition:  2.92965014976
# precision with repetition:  0.256666666667
# spearman correlation with repetition:  0.999999871257
# rmse without repetition:  2.90932981974
# precision without repetition: 0.236666666667
# spearman correlation without repetition:  0.999999873037
