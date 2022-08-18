import numpy as np




def c_min(arr):
    res = np.inf

    for i in range(arr.shape[0]):
        if res > arr[i]:
            res = arr[i]
    return res


def c_sum(arr):
    res = 0

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            res += arr[i, j]

    return res


def c_sum_axis_0(arr):
    res = np.zeros(arr.shape[1], dtype=np.float)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            res[j] += arr[i, j]

    return res


def c_sum_axis_1(arr):
    res = np.zeros(arr.shape[0], dtype=np.float)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            res[i] += arr[i, j]

    return res
