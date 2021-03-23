import numpy as np
import tensorflow as tf
import os
import copy
import random


def exclude_mask(position_dict, coal, seg):

    if isinstance(coal, int):
        coal = [coal]
    masks = np.array(position_dict[(0, 0)])
    masks_i = np.array(position_dict[(0, 1)])

    masks[:, seg[0]:seg[1]] = 0
    masks_i[:, seg[0]:seg[1]] = 0
    masks_i[:, coal[0]:coal[-1]+1] = 1

    new_position_dict = {
        (0, 0): masks,
        (0, 1): masks_i
    }
    return new_position_dict


def get_new_list(coalition, clist):
    """
    Divide all coalitions into pixels/words except the target coalition
    :params coalition: the target coalition to compute Shapley value
    :params clist: coalition list generate from p_ij
    :return new_list: coalition list where the only coalition is the target coalition
    """
    new_list = []
    pos = -1
    for j in range(len(clist)):
        if clist[j] != coalition and isinstance(clist[j], list):
            new_list.extend(clist[j])
        elif clist[j] == coalition:
            pos = len(new_list)
            new_list.append(clist[j])
        else:
            new_list.append(clist[j])
    return new_list, pos


def get_start_end(slist, cIdx, a_len, neighbour=1):
    # l, r in new_list
    l = 0
    r = len(slist) - 1
    if (cIdx + neighbour) < len(slist):
        r = cIdx + neighbour
    if (cIdx - neighbour) >= 0:
        l = cIdx - neighbour
    # l, r in a_len list
    # 
    # if l == 0: #
    #     l = 1
    r = r + len(slist[cIdx])

    # [l, r)
    return l, r


def pij_coals(pij, threshold = 0.5, seg = None):
    '''
    threshold: needed to decide coalitions
    seg: the chosen seg
    return: a two-dimension list, denoting set of coalitions
    '''
    coalitions = []
    players = [i for i in range(pij.shape[0]+1)]
    # players = [i+seg[0] for i in range(pij.shape[0]+1)]
    start = 0
    end = 1
    # for i in range(0, seg[0]):
    #     coalitions.append([i])
    while end < len(players):
        if pij[end-1] > threshold:
            end += 1
        else:
            coalitions.append(players[start:end])
            start = end
            end = start+1
    coalitions.append(players[start:end])
    # for i in range(seg[1],seg[2]):
    #     coalitions.append([i])
    return coalitions


def g_sample_bern(pij):
    # 
    sample_res = np.zeros_like(pij)
    for i in range(len(pij)):
        if random.random() < pij[i]:
            sample_res[i] = 1
    return sample_res


def measure_coalition(g_sample):
    #
    """
    :param g_sample: sampled g
    :return: lambda i
    """
    sizes = [1 for i in range(len(g_sample)+1)]
    l = 0
    size = 0
    for i in range(len(g_sample)):
        if g_sample[i] > 0:
            size += 1
        else:
            for j in range(l,i+1):
                sizes[j] += size
            size = 0
            l = i+1
    for j in range(l, len(sizes)):
        sizes[j] += size
    return sizes


def compute_sum(scores_c_s, scores_c_si, g_sample, idx):

    exp = [0.0, 0.0, 0.0, 0.0]
    if idx > 0 and g_sample[idx-1] > 0:
        exp[0] += scores_c_si
        exp[2] += scores_c_s
    else:
        exp[1] += scores_c_si
        exp[3] += scores_c_s
    return exp


def cal_overall_exp(score_list):
    score_list = np.array(score_list)
    overall_sum = np.sum(score_list, axis=0)
    overall_count = np.sum(score_list!= 0, axis=0)
    overall_exp = overall_sum / (overall_count+1e-10)
    return overall_exp


def get_interactions(max_score, min_score):
    results = [0 for _ in range(len(min_score))]
    interactions = [[] for _ in range(len(min_score))]
    for i in range(len(min_score)):
        for j in range(len(min_score[i])):
            for k in range(len(max_score[i])):
                interaction = max_score[i][k] - min_score[i][j]
                interactions[i].append(interaction)
        count = 0
        for j in range(len(interactions[i])):
            for k in range(j + 1, len(interactions[i])):
                count += 1
                results[i] += (abs(interactions[i][j] - interactions[i][k]) / \
                           (0.5 * (abs(interactions[i][j]) + abs(interactions[i][k]))))
        results[i] = results[i] / count
    return results


def get_interactions_2(max_score, min_score):
    results = [0 for _ in range(len(min_score))]
    interactions = [[] for _ in range(len(min_score))]
    for i in range(len(min_score)):
        for j in range(len(min_score[i])):
            for k in range(len(max_score[i])):
                interaction = max_score[i][k] - min_score[i][j]
                interactions[i].append(interaction)
        count = 0
        for j in range(len(interactions[i])):
            for k in range(j + 1, len(interactions[i])):
                count += 1
                results[i] += (abs(interactions[i][j] - interactions[i][k]))
        abs_mean = np.mean(np.abs(np.array(interactions[i])))
        results[i] = results[i] / abs_mean
        results[i] = results[i] / count
    return results


if __name__ == '__main__':
    get_interactions_2([[1]], [[2]])
