import numpy as np
import tensorflow as tf
import os
import copy
import json
import random
from tensorflow.python.framework import ops
from shapley import *
from coalition_utils import *
from keras.utils import to_categorical
from matplotlib import pyplot as plt

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "maximize_shap", True,
    "True: maximize the sum of Shapley value; False: minimize the sum of Shapley value")

flags.DEFINE_integer(
    "epoch_num", 200, "Total number of training epochs."
)

flags.DEFINE_float(
    "lr", 1, "The learning rate."
)

flags.DEFINE_integer(
    "m_cnt", 50, "Sample number for each coalition"
)

flags.DEFINE_integer(
    "g_sample_num", 10, "Coalition sample number"
)

flags.DEFINE_integer(
    "min_seg_len", 2, "min length of selected segmentaion"
)

flags.DEFINE_integer(
    "max_seg_len", 5, "max length of selected segmentaion"
)

flags.DEFINE_string(
    "data_path", "toy_dataset/datasets/toy_dataset_exp.json", "path to the dataset"
)

flags.DEFINE_string(
    "resume_path", None, "path to a previous result, resume from it"
)

flags.DEFINE_string(
    "output_path", FLAGS.data_path.replace("datasets", "results"), "path to saving the results"
)


weight_decay = 0  # no use, set as zero
threshold = 0.5

resume = False if FLAGS.resume_path is None else True

seg_range = (FLAGS.min_seg_len, FLAGS.max_seg_len + 1)
m_cnt = FLAGS.m_cnt
g_sample_num = FLAGS.g_sample_num

random.seed(FLAGS.epoch_num + m_cnt * 100)
np.random.seed(FLAGS.epoch_num + m_cnt * 100)
tf.set_random_seed(FLAGS.epoch_num + m_cnt * 100)


def compute_score_toy(masks, masks_i, model_str, a_len):
    ones = np.ones(a_len)
    scores = []
    scores_i = []
    for mask in masks:
        a = mask * ones
        a = a.astype(np.int32)
        scores.append(eval(model_str))
    for mask_i in masks_i:
        a = mask_i * ones
        a = a.astype(np.int32)
        scores_i.append(eval(model_str))
    return scores, scores_i


def manage_an_item(seg, model_str):
    a_len = seg[-1]
    seg_len = seg[1] - seg[0]
    p_mask = np.zeros(a_len - 1)
    p_mask[seg[0]:seg[1] - 1] = 1
    # print("\nCurrent words:", tokens_a[seg[0]:seg[1]])

    # =================================================================================================
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session()

        tmp = [0.0] * (a_len - 1)
        pij_weights = tf.Variable(tmp)  # initial value of p, before sigmoid

        # pij_weights = tf.Variable(tf.random.normal([a_len-1]))
        pij_weights_ = tf.sigmoid(pij_weights)  # add sigmoid

        pij_masked = tf.where(p_mask > 0, pij_weights_, tf.zeros_like(pij_weights_))  # freeze p out of selected seg

        tf.summary.histogram("pij", pij_masked[seg[0]:seg[1]])
        for i in range(seg_len - 1):
            tf.summary.scalar("p_%d" % i, pij_masked[seg[0] + i])

        p_c = pij_masked[seg[0]:seg[1] - 1]
        p_seg = tf.concat([[[0.0]], [p_c]], axis=1)[0, :]  # ensure number of ps same as number of words

        overall_expected = tf.placeholder(shape=[seg_len, 4], dtype=tf.float32)

        phi_c = overall_expected[:, 0] * p_seg \
                + overall_expected[:, 1] * (1 - p_seg) \
                - overall_expected[:, 2] * p_seg \
                - overall_expected[:, 3] * (1 - p_seg)
        g_score = tf.reduce_sum(phi_c)

        mean, variance = tf.nn.moments(pij_masked[seg[0]:seg[1] - 1], axes=0)
        var_loss = -weight_decay * variance
        tf.summary.scalar("variance", var_loss)

        if FLAGS.maximize_shap:
            loss = tf.negative(g_score)
            totloss = loss + var_loss
        else:
            loss = g_score
            totloss = loss + var_loss

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.lr, global_step, 10, 1)
        my_opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.4)
        train_step = my_opt.minimize(totloss, global_step=global_step)

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("total_loss", totloss)

        merged_summary_op = tf.summary.merge_all()

        # =======================================================================================================================

        init = tf.global_variables_initializer()
        sess.run(init)

        item_list = [i for i in range(a_len)]

        for epoch in range(FLAGS.epoch_num):
            # print(p_mask)
            pij = sess.run(pij_masked)  # numpy ndarray
            # pij = 1 / (1 + np.exp(-pij))
            # clist = pij_coals(pij)

            end_flag = True
            for item in pij[seg[0]:seg[1]-1]:
                if item < 0.9 and item > 0.1:
                    end_flag = False
                    break
            if end_flag:
                break

            clist = pij_coals(pij, seg=seg)

            words = []
            for coal in clist:
                if len(coal) == 1:
                    item = coal[0]
                else:
                    item = coal
                if isinstance(item, int):
                    words.append(item)
                else:
                    tmp = []
                    for id in item:
                        tmp.append(id)
                    words.append(tmp)

            print('pij', pij, clist)
            print("coalition:", words)

            score_exp_list = []
            coal_size_list = []
            for g_ in range(g_sample_num):
                g_sample = g_sample_bern(pij)  # sample g

                coal_size = measure_coalition(g_sample)

                coal_size_list.append(coal_size)

                g_clist = pij_coals(g_sample, seg=seg)  # partition the coalition based on sampled g

                score_exp_items = []
                score_item = [0.0, 0.0]

                for cIdx, coal in enumerate(g_clist):
                    # new_list, cIdx = get_new_list(item, item_list)

                    if coal[0] < seg[0] or coal[0] >= seg[1]:  # whether out of the seg
                        continue

                    positions_dict = get_masks_sampleshapley(g_clist, cIdx, a_len, m_cnt)  # sample S
                    positions_dict = exclude_mask(positions_dict, coal, seg)

                    scores_c_s, scores_c_si = compute_score_toy(positions_dict[(0, 0)], positions_dict[(0, 1)], model_str, a_len)

                    score_item[0] += np.mean(scores_c_si)
                    score_item[1] += np.mean(scores_c_s)

                score_item[0] /= seg_len
                score_item[1] /= seg_len

                for idx, item in enumerate(item_list[seg[0]:seg[1]]):
                    score_exp = compute_sum(score_item[1], score_item[0], g_sample, item)
                    score_exp_items.append(score_exp)

                score_exp_list.append(score_exp_items)

            overall_exp_score = cal_overall_exp(score_exp_list)

            in_dict = {
                overall_expected: overall_exp_score
            }

            _, _loss, summary_str, lr, var, g_score_ = sess.run(
                [train_step, loss, merged_summary_op, learning_rate, var_loss, g_score], feed_dict=in_dict)

            print('epoch:', epoch, '-->loss:', _loss, '-->variance_loss:', var, '-->learning_rate:', lr, "\n")
        res_p = sess.run(pij_masked)
        sess.close()
    return res_p


def thre(p, t=0.5):
    p = np.array(p)
    p[p >= t] = 1
    p[p < t] = 0
    return list(p)


def eval_res(p, gt, seg):
    print(p)
    count = np.array([0., 0.])
    correct = np.array([0., 0.])
    acc = np.array([0., 0.])
    for i in range(seg[0], seg[1]-1):
        count[gt[i]] += 1
        if p[i] == gt[i]:
            correct[gt[i]] += 1
    assert (correct <= count).any()
    if count[0] == 0:
        acc[0] = 1
    else:
        acc[0] = correct[0] / count[0]
    if count[1] == 0:
        acc[1] = 1
    else:
        acc[1] = correct[1] / count[1]
    acc_tot = np.sum(correct) / np.sum(count)
    print(count)
    print(correct)
    return count, correct, acc[0], acc[1], acc_tot


def main(_):
    # -------------------- configuration ------------------------- #
    tf.logging.set_verbosity(tf.logging.INFO)

    with open(FLAGS.data_path, "r") as file:
        data = json.load(file)

    results = []

    accs_one = []
    accs_zero = []
    accs_total = []

    count_tot = np.array([0., 0.])
    correct_tot = np.array([0., 0.])

    if resume:
        with open(FLAGS.resume_path, "r") as f:
            results = json.load(f)
        start = len(results)
        for item in results[:start]:
            accs_one.append(item["Current"]["Zero"])
            accs_zero.append(item["Current"]["One"])
            accs_total.append(item["Current"]["Total"])
        count_tot += np.array(results[start-1]["count"])
        correct_tot += np.array(results[start-1]["correct"])
    else:
        start = 0

    for id_, item in enumerate(data[start:]):
        id = id_ + start
        print("id:", id)
        model_str = item[0]
        gt = item[1]

        print("model:", model_str)

        a_len = len(gt) + 1

        seg = [0, 0, a_len]
        seg_len = random.randint(seg_range[0], min(a_len//2, seg_range[1]))

        seg[0] = random.choice(range(a_len-seg_len))
        seg[1] = seg[0] + seg_len

        res_p = manage_an_item(seg, model_str)
        thre_p = thre(res_p, threshold)

        count, correct, acc_zero, acc_one, acc_total = eval_res(thre_p, gt, seg)
        count_tot += count
        correct_tot += correct
        accs_zero.append(acc_zero)
        accs_one.append(acc_one)
        accs_total.append(acc_total)

        print("Current Accuracy:")
        print("Zero:", acc_zero)
        print("One:", acc_one)
        print("Total:", acc_total)
        print("\n")

        cur = {"Zero": acc_zero, "One": acc_one, "Total": acc_total}
        ave = {"Zero": np.mean(accs_zero), "One": np.mean(accs_one), "Total": np.mean(accs_total)}
        results.append({"id": id_, "Current": cur, "Average": ave, "count": list(count_tot),
                        "correct": list(correct_tot), "p": res_p.tolist(), "seg": seg, "gt": gt, "thre": threshold})

        acc = np.sum(correct_tot) / np.sum(count_tot)
        pre = correct_tot[1] / (correct_tot[1] + (count_tot[0] - correct_tot[0]))
        recall = correct_tot[1] / count_tot[1]

        print(correct_tot, count_tot)
        print("Accuracy: %f" % (acc))
        print("Precision: %f" % (pre))
        print("Recall: %f" % (recall))
        print("F value: %f" % (pre * recall * 2 / (pre + recall)))

        with open(FLAGS.output_path, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    # parse flags and run main()
    tf.app.run()
