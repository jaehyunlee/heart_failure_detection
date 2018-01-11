import numpy as np
import wfdb
import random
import pickle

rr_list = []
label_list = []
base_dir = 'rrdata/'

onehot_normal = [1., 0., 0., 0.]
onehot_NYHA1 = [0., 1., 0., 0.]
onehot_NYHA2 = [0., 0., 1., 0.]
onehot_NYHA3 = [0., 0., 0., 1.]


def _make_data(filename, onehot):

    # Read File
    raw_rr = []
    annotation = wfdb.rdann(base_dir + filename, 'ecg', sampfrom=0, sampto=20000000)
    annotation.fs = 120

    # Make Raw RR List
    for index1 in range(len(annotation.sample) - 1):
        diff = annotation.sample[index1 + 1] - annotation.sample[index1]
        second = diff / 120
        raw_rr.insert(index1, second)

    # Get percentile
    sort_rr = np.sort(raw_rr)
    percent = len(raw_rr) * 10 // 100
    percentile_low = sort_rr[percent]
    percentile_high = sort_rr[len(raw_rr) - percent - 1]

    # Preprocess
    prep_rr = []
    for index2 in range(len(raw_rr)):
        if percentile_low <= raw_rr[index2] <= percentile_high:
            prep_rr.append(raw_rr[index2])

    # Make Data List
    for index3 in range(len(prep_rr)):
        if index3 % 784 == 783:
            rr_list.append(_normalize(prep_rr[index3 - 783: index3 + 1]))
            label_list.append(onehot)


def _shuffle(shuffle_num):
    for index2 in range(shuffle_num):
        random_number = random.randint(0, len(label_list) - 1)
        rr_list.append(rr_list.pop(random_number))
        label_list.append(label_list.pop(random_number))


def _normalize(sample_list):
    normalized_list = []
    if 0 < len(sample_list):
        max_value = max(sample_list)
        min_value = min(sample_list)

        for index1 in range(len(sample_list)):
            normalized_list.append((sample_list[index1] - min_value) / (max_value - min_value))

    return normalized_list


def make_data():

    # Read Heart Failure Subject Data
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    for i in range(201, 230):
        filename = 'chf%03d' % i
        with open(base_dir + filename + '.hea', 'r') as f:
            raw = f.read()
            nyha_class = raw.split(' ')[11]

            if len(nyha_class) == 2:
                if cnt1 < 4:
                    _make_data(filename, onehot_NYHA1)
                    cnt1 += 1
                    print('Reading ', filename, '(class 1)')
            elif len(nyha_class) == 3:
                if cnt2 < 4:
                    _make_data(filename, onehot_NYHA2)
                    cnt2 += 1
                    print('Reading ', filename, '(class 2)')
            elif len(nyha_class) == 4:
                if cnt3 < 4:
                    _make_data(filename, onehot_NYHA3)
                    cnt3 += 1
                    print('Reading ', filename, '(class 3)')

    # Read Normal Subject Data
    for i in range(1, 5):
        filename = 'nsr%03d' % i
        print('Reading ', filename, '(normal)')
        _make_data(filename, onehot_normal)

    # Shuffle Data
    print('Shuffling')
    _shuffle(len(label_list) * 100)

    # Make Test Data
    test_rr_list = []
    test_label_list = []
    for i in range(len(rr_list)//10):
        test_rr_list.append(rr_list.pop())
        test_label_list.append(label_list.pop())

    rr_arr = np.array(rr_list)
    label_arr = np.array(label_list)
    test_rr_arr = np.array(test_rr_list)
    test_label_arr = np.array(test_label_list)

    with open('pickle/rr_data.p', 'wb') as file:
        pickle.dump(rr_arr, file)
        pickle.dump(label_arr, file)
        pickle.dump(test_rr_arr, file)
        pickle.dump(test_label_arr, file)
    print('Save Done!')


def get_data():
    with open('pickle/rr_data.p', 'rb') as file:
        rr_arr = pickle.load(file)
        label_arr = pickle.load(file)
        test_rr_arr = pickle.load(file)
        test_label_arr = pickle.load(file)
    return rr_arr, label_arr, test_rr_arr, test_label_arr
