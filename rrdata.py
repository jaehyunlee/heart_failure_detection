import numpy as np
import wfdb
import random
import pickle

rr_list = []
label_list = []

onehot_normal = [1., 0., 0., 0.]
onehot_NYHA1 = [0., 1., 0., 0.]
onehot_NYHA2 = [0., 0., 1., 0.]
onehot_NYHA3 = [0., 0., 0., 1.]


def _make_data(filename, onehot):

    # Read File
    raw_rr = []
    annotation = wfdb.rdann('rrdata/'+filename, 'ecg', sampfrom=85, sampto=11160882)
    annotation.fs = 120

    # Make Raw RR List
    for index1 in range(len(annotation.sample) - 1):
        diff = annotation.sample[index1 + 1] - annotation.sample[index1]
        second = diff / 120
        raw_rr.insert(index1, second)

    # Get percentile
    sort_rr = np.sort(raw_rr)
    percent5 = len(raw_rr) // 20
    percentile_5 = sort_rr[percent5]
    percentile_95 = sort_rr[len(raw_rr) - percent5 - 1]

    # Preprocess
    prep_rr = []
    for index2 in range(len(raw_rr)):
        if percentile_5 < raw_rr[index2] < percentile_95:
            prep_rr.append(raw_rr[index2])

    # Make Data List
    for index3 in range(len(prep_rr)):
        if index3 % 784 == 783:
            rr_list.append(prep_rr[index3 - 783: index3 + 1])
            label_list.append(onehot)


def _shuffle(shuffle_start, shuffle_end, shuffle_num):
    for index2 in range(shuffle_num):
        random_number = random.randint(shuffle_start, shuffle_end - 1)
        tmp_rr = rr_list.pop(random_number)
        rr_list.append(tmp_rr)
        tmp_label = label_list.pop(random_number)
        label_list.append(tmp_label)


def make_data():

    _make_data('chf218', onehot_NYHA1)
    _make_data('chf219', onehot_NYHA3)
    _make_data('chf220', onehot_NYHA2)
    _make_data('nsr001', onehot_normal)
    _shuffle(0, len(label_list), len(label_list) * 100)

    rr_arr = np.array(rr_list)
    label_arr = np.array(label_list)

    with open('pickle/rr_data.p', 'wb') as file:
        pickle.dump(rr_arr, file)
        pickle.dump(label_arr, file)


def get_data():
    with open('pickle/rr_data.p', 'rb') as file:
        rr_arr = pickle.load(file)
        label_arr = pickle.load(file)
    return rr_arr, label_arr
