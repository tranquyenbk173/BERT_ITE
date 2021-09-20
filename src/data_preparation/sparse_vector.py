from collections import Counter
import json
import numpy as np


def json_string_to_tensorflow_sparse_vector(string_vector):
    """Convert string to sparse vector

    Parameters
    ----------
    string_vector : str
        string_vector to be converted

    Examples
    ---------
    >>> string_vector = '[[1, 0.123232], [2, 5.34234234]]'
    >>> json_string_to_tensorflow_sparse_vector(string_vector)
    [1, 2], [0.123232, 5.34234234])
    """
    indices = []
    values = []
    if string_vector is None or len(string_vector) < 1:
        return np.array([0], dtype=np.int64), np.array([0], dtype=np.float32)
    else:
        list_vector = json.loads(string_vector)
        if len(list_vector) == 0:
            return np.array([0], dtype=np.int64), np.array([0], dtype=np.float32)
        for kv in list_vector:
            indices.append(kv[0])
            values.append(kv[1])
    return np.array(indices, dtype=np.int64), np.array(values, dtype=np.float32)


def list_sparse_vector_to_json_string(sv):
    # type: (list) -> str
    """Convert vector in the form list of (int, float) to string

    Parameters
    ----------
    sv : list of (int, float)
        sparse_vector to be converted

    Examples
    ---------
    >>> sparse_vector = [(1, 0.123232), (2, 5.34234234)]
    >>> list_sparse_vector_to_json_string(sparse_vector)
    '[[1, 0.123232], [2, 5.34234234]]'
    """

    temp_sum = 0
    list_sparse = []

    for item in sv:
        temp_sum += item[1]

    for item in sv:
        list_sparse.append([int(item[0]), float(item[1] / temp_sum)])

    return json.dumps(list_sparse)


def dict_sparse_vector_to_json_string(sv):
    # type: (dict) -> str
    """Convert vector in the form dict of (int, float) to string

    Parameters
    ----------
    sv : dict of int, float
        sparse_vector to be converted

    Examples
    ---------
    >>> sparse_vector = {1: 0.123232, 2: 5.34234234}
    >>> dict_sparse_vector_to_string(sparse_vector)
    '[[1, 0.123232], [2, 5.34234234]]'
    """

    s = sum(sv.values())
    for key in sv.keys():
        sv[key] /= s
    list_sparse = []
    for key in sorted(sv.keys()):
        list_sparse.append([int(key), float(sv.get(key))])
    return json.dumps(list_sparse)


def json_string_to_dense_vector(string_vector, dimension):
    """
    Chuyen vector thua dang string: [[9, 0.010176822], [118, 0.010578092], [264, 0.020403702]]
    ve vector thuong K chieu cua numpy
    """

    vector = np.zeros(dimension, dtype=np.float32)
    if string_vector is None or len(string_vector) < 1:
        return vector

    x = json.loads(string_vector)

    for element in x:
        vector[element[0]] = element[1]
    return vector


def dense_vector_to_list_sparse_vector(vector, threshold):
    """
    Chuyen vector np ve vector thua dang string: [[9, 0.010176822], [118, 0.010578092], [264, 0.020403702]]
    """
    K = len(vector)
    # threshold = 1e-6

    list_temp = []
    for i in range(0, K):
        if vector[i] > threshold:
            list_temp.append([i, float(vector[i])])
    return list_temp


def dense_vector_to_json_string(dense_sv, threshold=0, normalizing=False):
    """

        Parameters
        ----------
        threshold: float
        normalizing: bool
        """
    K = len(dense_sv)
    # threshold = 1e-6

    temp_sum = 0
    list_temp = []
    for i in range(0, K):
        if dense_sv[i] > threshold:
            v = float(dense_sv[i])
            list_temp.append([i, v])
            temp_sum += v
    if normalizing:
        for i in range(len(list_temp)):
            list_temp[i][1] /= temp_sum
    return json.dumps(list_temp)


def weighted_sum(sv1, sv2, sv1_weight, dimension, threshold=0):
    """Sum two sparse vectors

    Parameters
    ----------
    sv1, sv2: str
        sparse_vector
    sv1_weight: float
        weight for sv1
    dimension: int
        dimension for dense vector
    threshold: float
    """
    dense_vec_1 = json_string_to_dense_vector(sv1, dimension)
    dense_vec_2 = json_string_to_dense_vector(sv2, dimension)
    sum_sv = sv1_weight * dense_vec_1 + (1 - sv1_weight) * dense_vec_2
    return dense_vector_to_json_string(sum_sv, threshold, normalizing=True)


def sum_with_count(sv1_with_count, sv2_with_count, dimension, threshold=0):
    dense_vec_1 = json_string_to_dense_vector(sv1_with_count[0], dimension)
    dense_vec_2 = json_string_to_dense_vector(sv2_with_count[0], dimension)
    sum_sv = dense_vec_1 + dense_vec_2
    sum_count = sv1_with_count[1] + sv2_with_count[1]
    return dense_vector_to_json_string(sum_sv, threshold, normalizing=False), sum_count


def add(sv1, sv2, old_vector_weight=0.25):
    """Sum two sparse vectors

    Parameters
    ----------
    sv1, sv2: list of float
        sparse_vector
    old_vector_weight: weight for sv1
    """
    sum_sv = old_vector_weight * sv1 + (1 - old_vector_weight) * sv2
    return sum_sv


def add_avg(sv1, sv2):
    sum_sv = sv1[0] + sv2[0]
    num_banners = sv1[1] + sv2[1]
    return sum_sv, num_banners
