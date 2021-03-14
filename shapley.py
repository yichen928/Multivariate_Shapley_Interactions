import numpy as np
import time
from itertools import combinations, chain
import scipy.special
from keras.utils import to_categorical


def turn_list(s):
    if type(s) == list:
        return s
    elif type(s) == int:
        return [s]


def powerset(S, order=None):
    """
    Compute the power set(a set of all subsets) of a set.
    :param S: a set represented by list
    :param order: max number of selected elements from S
    :return: a dictionary that partitions subsets by cardinality.
    keys are the cardinality and values are lists of subsets.
    """
    if order is None:
        order = len(S)

    # empty set; ...; size is r sets;
    return {r:list(combinations(S, r)) for r in range(order + 1)}


def get_masks_sampleshapley(clist, cIdx, a_len, m=100):
    """Construct the position dict of sample shapley
    :param clist: current word combination list,  e.g. clist=[0,1,[2,3],4,5]
    :param cIdx: index of coalition,  2
    :param a_len: length of tokens in the sentence, 6
    :param m: # of samples
    :return list of sampled masks
    """
    d = len(clist)  # 5
    coa = turn_list(clist[cIdx])  # current coalition [2,3]

    positions_dict = {(i, fill): [] for i in range(1) for fill in [0, 1]}
    # sample m times
    for cnt in range(m):
        perm = np.random.permutation(d)
        preO = []
        for idx in perm:
            if idx != cIdx:
                preO.append(turn_list(clist[idx]))
            else:
                break

        preO_list = list(chain.from_iterable(preO))
        pos_excluded = np.sum(to_categorical(preO_list, num_classes=a_len), axis=0)
        pos_included = pos_excluded + np.sum(to_categorical(turn_list(clist[cIdx]), num_classes=a_len), axis=0)
        positions_dict[(0, 0)].append(pos_excluded)
        positions_dict[(0, 1)].append(pos_included)

    return positions_dict


def get_sampleshapley(positions_dict):

    keys, values = positions_dict.keys(), positions_dict.values()  # return a list of all values
    values = [np.array(value) for value in values]
    positions = np.concatenate(values, axis=0)  # [2*m, a_len]

    key_to_idx = {}
    count = 0
    for i, key in enumerate(keys):
        key_to_idx[key] = list(range(count, count + len(values[i])))
        count += len(values[i])

    return key_to_idx, positions


def singleton_shapley(slist):
    """compute shapley values for each word i"""
    d = len(slist)
    a_len = d
    m = 1000
    null_set = 1 - np.sum(to_categorical(list(range(d)), num_classes=a_len), axis=0)

    positions_dict = {(i, fill):[] for i in range(d) for fill in [0, 1]}

    for cnt in range(m):
        perm = np.random.permutation(d)
        pos_exc = null_set

        preO = []
        for idx in perm:
            preO.append([idx])
            preO_list = list(chain.from_iterable(preO))
            pos_inc = np.sum(to_categorical(preO_list, num_classes=a_len), axis=0)
            positions_dict[(idx, 0)].append(pos_exc)
            positions_dict[(idx, 1)].append(pos_inc)
            pos_exc = pos_inc

    keys, values = positions_dict.keys(), positions_dict.values()
    values = [np.array(value) for value in values]
    positions = np.concatenate(values, axis=0)  #[d*2m, a_len]

    key_to_idx = {}
    count = 0
    for i, key in enumerate(keys):
        key_to_idx[key] = list(range(count, count + len(values[i])))
        count += len(values[i])

    return key_to_idx, positions


def explain_shapley(predict, x, batch_dict, key_to_idx):
    """
    Compute the importance score/shapley value
    :param predict: network function
    :param d: d points needed to coompute shapley value in current coalition
    :param x: feature of input x
    :param batch_dict: all subsets needed to compute [S]
    :return:
    """
    f_logits = predict(batch_dict)  # [:, c]
    logits = predict(x)  # [1, c]

    # one-hot vector to denote the refer class of x
    discrete_probs = np.eye(len(logits[0]))[np.argmax(logits, axis=-1)]
    vals = np.sum(discrete_probs * f_logits, axis=1)

    # key_to_idx[key]: list of indices in original position
    key_to_val = {key: np.array([vals[idx] for idx in key_to_idx[key]]) for key in key_to_idx}

    # Compute importance scores
    phis = (key_to_val[(0, 1)] - key_to_val[(0, 0)])

    return phis


def explain_shapley_seperate(predict, x, batch_dict, key_to_idx):
    """
    Compute the importance score/shapley value
    :param predict: network function
    :param d: d points needed to coompute shapley value in current coalition
    :param x: feature of input x
    :param batch_dict: all subsets needed to compute [S]
    :return:
    """
    f_logits = predict(batch_dict)  # [:, c]
    logits = predict(x)  # [1, c]

    # one-hot vector to denote the refer class of x
    discrete_probs = np.eye(len(logits[0]))[np.argmax(logits, axis=-1)]
    vals = np.sum(discrete_probs * f_logits, axis=1)

    # key_to_idx[key]: list of indices in original position
    key_to_val = {key: np.array([vals[idx] for idx in key_to_idx[key]]) for key in key_to_idx}

    # Compute importance scores
    # phis = (key_to_val[(0, 1)] - key_to_val[(0, 0)])

    return key_to_val


def compute_scores(positions_dict, feature, a_len, predict, method='SampleShapley'):
    """
    Compute shapley values of each feature(including coalition) of clist
    :param clist: current combination list, e.g. [0,1,[2,3],4]
    :param cIdx: coalition index, e.g. 2
    :param feature: input features of x
    :param a_len: # of tokens in sentence x without padding and [CLS],[SEP]
    :return:
    """
    if method == 'SampleShapley':
        key_to_idx, positions = get_sampleshapley(positions_dict)
    else:
        print('Not Supported Method')

    # tokens: [CLS] the dog is hairy . [SEP]
    real_ids = feature.input_ids[1:a_len+1]
    inputs = np.array(real_ids) * positions
    #print('inputs shape is:', inputs.shape)

    batch_in = {'input_ids':[], 'input_mask':[], 'segment_ids':[], 'label_ids':[]}
    for j in range(inputs.shape[0]):
        input_ids = [feature.input_ids[0]]  # [CLS]
        input_ids = input_ids + list(inputs[j])
        input_ids = input_ids + feature.input_ids[(a_len+1):]  # [SEP] to max_seq_length padding

        # ft = extract.InputFeatures(
        #     input_ids=input_ids,
        #     input_mask=feature.input_mask,
        #     segment_ids=feature.segment_ids,
        #     label_id=feature.label_id,
        #     is_real_example=True
        # )
        batch_in['input_ids'].append(input_ids)
        batch_in['input_mask'].append(feature.input_mask)
        batch_in['segment_ids'].append(feature.segment_ids)
        batch_in['label_ids'].append(feature.label_id)

    batch_dict = {
        'input_ids': np.array(batch_in['input_ids']),
        'input_mask': np.array(batch_in['input_mask']),
        'segment_ids': np.array(batch_in['segment_ids']),
        'label_ids': np.array(batch_in['label_ids']),
    }
    x = {
        'input_ids': np.array([feature.input_ids]),
        'input_mask': np.array([feature.input_mask]),
        'segment_ids': np.array([feature.segment_ids]),
        'label_ids': np.array([feature.label_id])
    }

    shaps = explain_shapley(predict, x, batch_dict, key_to_idx)

    return shaps


def compute_scores_seperate(positions_dict, feature, a_len, predict, method='SampleShapley'):
    """
    Compute shapley values of each feature(including coalition) of clist, seperately
    :param clist: current combination list, e.g. [0,1,[2,3],4]
    :param cIdx: coalition index, e.g. 2
    :param feature: input features of x
    :param a_len: # of tokens in sentence x without padding and [CLS],[SEP]
    :return:
    """
    if method == 'SampleShapley':
        key_to_idx, positions = get_sampleshapley(positions_dict)
    else:
        print('Not Supported Method')

    # tokens: [CLS] the dog is hairy . [SEP]
    real_ids = feature.input_ids[1:a_len+1]
    inputs = np.array(real_ids) * positions
    #print('inputs shape is:', inputs.shape)

    batch_in = {'input_ids':[], 'input_mask':[], 'segment_ids':[], 'label_ids':[]}
    for j in range(inputs.shape[0]):
        input_ids = [feature.input_ids[0]]  # [CLS]
        input_ids = input_ids + list(inputs[j])
        input_ids = input_ids + feature.input_ids[(a_len+1):]  # [SEP] to max_seq_length padding

        # ft = extract.InputFeatures(
        #     input_ids=input_ids,
        #     input_mask=feature.input_mask,
        #     segment_ids=feature.segment_ids,
        #     label_id=feature.label_id,
        #     is_real_example=True
        # )
        batch_in['input_ids'].append(input_ids)
        batch_in['input_mask'].append(feature.input_mask)
        batch_in['segment_ids'].append(feature.segment_ids)
        batch_in['label_ids'].append(feature.label_id)

    batch_dict = {
        'input_ids': np.array(batch_in['input_ids']),
        'input_mask': np.array(batch_in['input_mask']),
        'segment_ids': np.array(batch_in['segment_ids']),
        'label_ids': np.array(batch_in['label_ids']),
    }
    x = {
        'input_ids': np.array([feature.input_ids]),
        'input_mask': np.array([feature.input_mask]),
        'segment_ids': np.array([feature.segment_ids]),
        'label_ids': np.array([feature.label_id])
    }

    shaps = explain_shapley_seperate(predict, x, batch_dict, key_to_idx)

    return shaps[(0, 0)], shaps[(0, 1)]




