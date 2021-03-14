import numpy as np
import tensorflow as tf
import os
import math
import itertools
import copy
from build_model import TextModel
from tensorflow.python.framework import ops
from shapley import *
from preprocess import tokenization, extract
from keras.utils import to_categorical
from matplotlib import pyplot as plt

def compute_scores_seperate_elmo(positions_dict, embedding, label, predictor, layer_vector=None):
    with predictor.input.graph.as_default():
        embedding_samplei = []
        embedding_sample = []
        for positions in positions_dict[(0, 1)]:
            embedding_samplei.append(embedding * positions.reshape(-1, 1))
        embedding_samplei = np.array(embedding_samplei)
        # batch_dict = {"input_1": embedding_samplei}
        logits_i = predictor.predict(embedding_samplei)
        for positions in positions_dict[(0, 0)]:
            embedding_sample.append(embedding * positions.reshape(-1, 1))
        embedding_sample = np.array(embedding_sample)
        # batch_dict = {"input_1": embedding_sample}
        logits = predictor.predict(embedding_sample)
    # return logits[:, label], logits_i[:, label]
    if layer_vector is None:
        return logits[:, label], logits_i[:, label]
    else:
        return np.sum(logits*layer_vector, axis=1), np.sum(logits_i*layer_vector, axis=1)


def get_all_coalitions(seg):
    clists = []

    def generate_new_coalition(divide, a, seg):
        if a < seg[1] - 1:
            divide.append(a)
            generate_new_coalition(divide, a + 1, seg)
            divide.pop()
            generate_new_coalition(divide, a + 1, seg)
        elif a == seg[1] - 1:
            clist = [i for i in range(seg[2])]
            tmp_divide = divide[:]
            if len(tmp_divide) > 0:
                cnt = 0
                for i in tmp_divide:
                    tmp = []
                    if isinstance(clist[i - cnt], list):
                        tmp.extend(clist[i - cnt])
                    else:
                        tmp.append(clist[i - cnt])
                    tmp.append(clist[i - cnt + 1])
                    del clist[i - cnt]
                    del clist[i - cnt]
                    clist.insert(i - cnt, tmp)
                    cnt += 1
            clists.append(clist)

    generate_new_coalition([], seg[0], seg)
    return clists


def shapley_sample(seg, coal):
    """

    :param seg:
    :param clist:
    :return:
    """

    a_len = seg[-1]
    seg_len = seg[1] - seg[0]
    other_len = a_len - seg_len

    other_list = []
    for i in range(a_len):
        if i < seg[0]:
            other_list.append(i)
        if i >= seg[1]:
            other_list.append(i)

    masks = []
    masks_i = []
    weight_list = []

    for num in range(other_len+1):
        comb = list(itertools.combinations(other_list, num))
        for item in comb:
            mask = np.zeros_like(list(range(a_len)))
            mask_i = np.zeros_like(list(range(a_len)))
            if isinstance(coal, list):
                for i in coal:
                    mask_i[i] = 1
            else:
                mask_i[coal] = 1
            for i in item:
                mask[i] = 1
                mask_i[i] = 1
            weight = 1.0 * math.factorial(num) * math.factorial(other_len-num) / math.factorial(other_len+1)
            weight_list.append(weight)
            masks.append(mask)
            masks_i.append(mask_i)
    # print(clist)
    # print(masks)
    # print(masks_i)
    weight_list = np.array(weight_list)
    position_dict = {
        (0, 0): masks,
        (0, 1): masks_i
    }
    return position_dict, weight_list


def calculate_shap_sum(clist, seg, embedding, label, predictor_model, print_info=False, layer_vector=None):
    if print_info:
        print(clist, seg)
        print(clist)
    a_len = seg[-1]
    cIdx = seg[0]
    shap_sum = 0
    while (isinstance(clist[cIdx], list) and clist[cIdx][0]>=seg[0] and clist[cIdx][0]<seg[1])or ((isinstance(clist[cIdx], int) and clist[cIdx] < seg[1])):
        if print_info:
            print(clist, cIdx)

        positions_dict, weight_list = shapley_sample(seg, clist[cIdx])

        scores_c_s, scores_c_si = compute_scores_seperate_elmo(positions_dict, embedding, label, predictor_model, layer_vector=layer_vector)
        scores_c = scores_c_si - scores_c_s
        # scores_c = compute_scores(positions_dict, feature, a_len, model.predict)
        scores_c = scores_c * weight_list
        cIdx += 1
        shap_sum += np.sum(scores_c)
        if cIdx >= len(clist):
            break
    if print_info:
        print(clist, shap_sum)
    return shap_sum


def get_min_max_shap(seg, embedding, label, model, layer_vector=None):
    clists = get_all_coalitions(seg)

    max_clist = []
    min_clist = []
    max_shap = -999999
    min_shap = 999999
    for clist in clists:
        print(clist)
        shap_sum = calculate_shap_sum(clist, seg, embedding, label, model, print_info = True, layer_vector=layer_vector)

        if shap_sum > max_shap:
            max_shap = shap_sum
            max_clist = clist[:]
        if shap_sum < min_shap:
            min_shap = shap_sum
            min_clist = clist[:]
        # print(clist, shap_sum)
    return min_shap, max_shap, min_clist, max_clist
