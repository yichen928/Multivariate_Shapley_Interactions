import numpy as np
import tensorflow as tf
import os
import copy
import random
import sys
sys.path.append("../..")
import data_helpers
from tensorflow.contrib import learn
from build_cnn import TextModel
from shapley_cnn import compute_scores_seperate_cnn, get_masks_sampleshapley
from coalition_utils import *
import time
import json

flags = tf.flags
FLAGS = flags.FLAGS

# Parameters =======================
# Data Parameters
flags.DEFINE_string("task_name", "sst-2", "Data source")
flags.DEFINE_string("positive_data_file", "../../CNN_model/cnn-text/data/sst2_val.pos", "Data source for the positive data.")
flags.DEFINE_string("negative_data_file", "../../CNN_model/cnn-text/data/sst2_val.neg", "Data source for the negative data.")

# Eval Parameters
flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
flags.DEFINE_string("checkpoint_dir", "../../CNN_model/cnn-text/runs/SST2/checkpoints/", "Checkpoint directory rom training run")

# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

flags.DEFINE_integer(
    "epoch_num", 100, "Total number of training epochs."
)

flags.DEFINE_float(
    "lr", 2.0, "The learning rate."
)

# ===================== hyperparameters for computing instability =======
flags.DEFINE_integer(
    "sentence_num", 20, "The number of sentences used for the computation of instability."
)

flags.DEFINE_integer(
    "min_len", 8, "The minimal length of the sentence in the computation of the instability"
)

flags.DEFINE_integer(
    "max_len", 12, "The maximal length of the sentence in the computation of the instability"
)

flags.DEFINE_integer(
    "test_num", 1, "The number of tests to compute the instability repeatedly."
)

sentence_num = FLAGS.sentence_num
min_len = FLAGS.min_len
max_len = FLAGS.max_len
test_num = FLAGS.test_num


def self_tokenizer(docs):
    ans = []
    for doc in docs:
        ans.append(doc.split())
    return ans


# load datas and labels
X_raw, Y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
Y_test = np.argmax(Y_test, axis=1)
# map data to vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
for i in range(len(X_raw)):
    X_raw[i] = X_raw[i].lower()

# transform to word vectors
X_test = np.array(list(vocab_processor.transform(X_raw)))
print('X raw len:', len(X_raw))
print('X test shape:', X_test.shape)


def main(_):
    # -------------------- configuration ------------------------- #
    tf.set_random_seed(FLAGS.epoch_num + sentence_num * 100)
    random.seed(FLAGS.epoch_num + sentence_num * 100)
    np.random.seed(FLAGS.epoch_num + sentence_num * 100)
    tf.logging.set_verbosity(tf.logging.INFO)
    task_name = FLAGS.task_name.lower()
    model_name = "cnn"
    if task_name not in ['sst-2', 'cola']:
        raise ValueError("Task not found: %s" % (task_name))
    date = time.strftime("%Y-%m-%d", time.localtime())

    # --------------Result-----------------------------------------
    res = []
    res.append({"lr": FLAGS.lr, "epoch_num": FLAGS.epoch_num})

    result_path = "../result/interaction/%s/%s/%s/" % (date, model_name, task_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # ------------------- preprocess dataset -------------------- #
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(X_raw))

    # ----------------------- build cnn predict model --------------------- #
    # sess1
    sess_config = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)

    cnn_model = TextModel(FLAGS.checkpoint_dir, sess_config)
    # cnn_model.start_session()

    # ------------------------------------------- #
    print('Making explanations...')
    count = 0
    for (i, sentence) in enumerate(X_raw):
        tokens_a = sentence.split()
        a_len = len(tokens_a)
        print(tokens_a)
        print('a_len is:', a_len)
        if a_len < min_len or a_len > max_len:
            continue
        count += 1
        if count > sentence_num:
            break

        feature = {
            'input_x': X_test[i],
            'input_y': Y_test[i],
        }

        left_pos = random.randint(0, a_len // 2)
        right_pos = min(a_len - 1, left_pos + random.randint(4, 6))
        seg = (left_pos, right_pos, a_len)  #
        seg_len = seg[1] - seg[0]
        p_mask = np.zeros(a_len - 1)
        p_mask[seg[0]:seg[1] - 1] = 1
        print("\nCurrent words:", tokens_a[seg[0]:seg[1]])

        sentence_result = {}
        sentence_result["id"] = i
        sentence_result["seg"] = seg
        sentence_result["loss"], sentence_result["pij"] = {"True": [], "False": []}, {"True": [], "False": []}

        m_cnts = [50]
        for m_cnt in m_cnts:
            # m_cnt = 50  #
            # n_steps = 10
            # m_cnt = int(m_cnt/(2*n_steps-1))*(2*n_steps-1)
            weight_decay = 0  #

            g_sample_num = 50  # 采样g的次数

            #==========================================================================
            for t in range(test_num):
                for mode in [True, False]:
                    sentence_result["m_cnt"] = m_cnt
                    sentence_result["g_sample_num"] = g_sample_num
                    g = tf.Graph()
                    with g.as_default():
                        sess = tf.Session()

                        tmp = [0.0] * (a_len - 1)
                        pij_weights = tf.Variable(tmp)  #

                        # pij_weights = tf.Variable(tf.random.normal([a_len-1]))
                        pij_weights_ = tf.sigmoid(pij_weights)  # add sigmoid

                        pij_masked = tf.where(p_mask > 0, pij_weights_, tf.zeros_like(pij_weights_))  #

                        tf.summary.histogram("pij", pij_masked[seg[0]:seg[1]])
                        for ii in range(seg_len - 1):
                            tf.summary.scalar("p_%d" % ii, pij_masked[seg[0] + ii])

                        p_c = pij_masked[seg[0]:seg[1] - 1]
                        p_seg = tf.concat([[[0.0]], [p_c]], axis=1)[0, :]  #
                        overall_expected = tf.placeholder(shape=[seg_len, 4], dtype=tf.float32)

                        phi_c = overall_expected[:, 0] * p_seg \
                                + overall_expected[:, 1] * (1 - p_seg) \
                                - overall_expected[:, 2] * p_seg \
                                - overall_expected[:, 3] * (1 - p_seg)
                        g_score = tf.reduce_sum(phi_c)

                        mean, variance = tf.nn.moments(pij_masked[seg[0]:seg[1] - 1], axes=0)
                        var_loss = -weight_decay * variance
                        tf.summary.scalar("variance", var_loss)

                        #
                        if mode:
                            # loss = tf.negative(tf.reduce_mean(g_score))
                            loss = tf.negative(g_score)
                            totloss = loss + var_loss
                        else:
                            # loss = tf.reduce_mean(g_score)
                            loss = g_score
                            totloss = loss + var_loss

                        global_step = tf.Variable(0, trainable=False)
                        learning_rate = tf.train.exponential_decay(FLAGS.lr, global_step, 10, 0.95)
                        # my_opt = tf.train.GradientDescentOptimizer(learning_rate)
                        my_opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
                        train_step = my_opt.minimize(totloss, global_step=global_step)

                        tf.summary.scalar("loss", loss)
                        tf.summary.scalar("total_loss", totloss)

                        merged_summary_op = tf.summary.merge_all()

                        # =======================================================================================================================

                        init = tf.global_variables_initializer()
                        sess.run(init)
                        # loss_list = []

                        item_list = [ii for ii in range(a_len)]

                        for epoch in range(FLAGS.epoch_num):
                            pij = sess.run(pij_masked)  # numpy ndarray
                            # pij = 1 / (1 + np.exp(-pij))
                            # clist = pij_coals(pij)

                            clist = pij_coals(pij, seg=seg)  #

                            words = []
                            for coal in clist:
                                if len(coal) == 1:
                                    item = coal[0]
                                else:
                                    item = coal
                                if isinstance(item, int):
                                    words.append(tokens_a[item])
                                else:
                                    tmp = []
                                    for id in item:
                                        tmp.append(tokens_a[id])
                                    words.append(tmp)

                            print('pij', pij, clist)
                            print("coalition:", words)

                            score_exp_list = []
                            coal_size_list = []
                            for g_ in range(g_sample_num):
                                g_sample = g_sample_bern(pij)  #

                                g_clist = pij_coals(g_sample, seg=seg)  #
                                score_exp_items = []
                                score_item = [0.0, 0.0]
                                positions_dicts = {}
                                positions_dicts[(0, 0)] = []
                                positions_dicts[(0, 1)] = []

                                for cIdx, coal in enumerate(g_clist):
                                    # new_list, cIdx = get_new_list(item, item_list)

                                    if coal[0] < seg[0] or coal[0] >= seg[1]:  #
                                        continue

                                    positions_dict = get_masks_sampleshapley(g_clist, cIdx, a_len, m_cnt)  #
                                    positions_dict = exclude_mask(positions_dict, coal, seg)

                                    positions_dicts[(0, 0)].extend(positions_dict[(0, 0)])
                                    positions_dicts[(0, 1)].extend(positions_dict[(0, 1)])
                                scores_c_s, scores_c_si = compute_scores_seperate_cnn(positions_dict, feature, a_len,
                                                                                  cnn_model.predict)
                                # score_diff = scores_c_si - scores_c_s

                                score_item[0] += np.mean(scores_c_si)
                                score_item[1] += np.mean(scores_c_s)

                                score_item[0] /= seg_len  #
                                score_item[1] /= seg_len

                                for idx, item in enumerate(item_list[seg[0]:seg[1]]):  #
                                    score_exp = compute_sum(score_item[1], score_item[0], g_sample, item)
                                    score_exp_items.append(score_exp)

                                score_exp_list.append(score_exp_items)

                            overall_exp_score = cal_overall_exp(score_exp_list)

                            in_dict = {
                                overall_expected: overall_exp_score
                            }

                            _, _loss, summary_str, lr, var, g_score_ = sess.run(
                                [train_step, loss, merged_summary_op, learning_rate, var_loss, g_score],
                                feed_dict=in_dict)

                            sentence_result["loss"][str(mode)].append(_loss)
                            sentence_result["pij"][str(mode)].append(list(pij))

                            print('epoch:', epoch, '-->loss:', _loss, '-->variance_loss:', var, '-->learning_rate:', lr, "\n")

        res.append(sentence_result)

    np.save('%s/result_%s_cnn.npy' % (result_path, FLAGS.task_name), res)
    print("total number: %d" % count)


if __name__ == "__main__":
    tf.app.run()



