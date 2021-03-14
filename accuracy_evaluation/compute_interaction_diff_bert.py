import numpy as np
import tensorflow as tf
import os
import copy
import random
import pickle
import json
from tensorflow.python.framework import ops
from keras.utils import to_categorical
import sys
import datetime

sys.path.append("..")
from truth_interaction_GT import calculate_shap_sum, get_min_max_shap
from build_model import TextModel
from shapley import *
from preprocess import tokenization, extract

bert_dir = "../models/uncased_L-12_H-768_A-12"

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", bert_dir + "/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "sst-2", "The name of the task to train.")
assert FLAGS.task_name in ["sst-2", "cola"]

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

if FLAGS.data_dir is None:
    if FLAGS.task_name == "sst-2":
        FLAGS.data_dir = "../GLUE_data/SST-2"
    else:
        FLAGS.data_dir = "../GLUE_data/CoLA"

if FLAGS.init_checkpoint is None:
    if FLAGS.task_name == "sst-2":
        FLAGS.init_checkpoint = FLAGS.data_dir + "/model/model.ckpt-6313"
    else:
        FLAGS.init_checkpoint = FLAGS.data_dir + "/model/model.ckpt-801"

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string("vocab_file", bert_dir + "/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "maximize_shap", True,
    "True: maximize the sum of Shapley value; False: minimize the sum of Shapley value")

flags.DEFINE_integer(
    "epoch_num", 100, "Total number of training epochs."
)

flags.DEFINE_float(
    "lr", 1.0, "The learning rate."
)

flags.DEFINE_integer(
    "m_cnt", 25, "Sample number for each coalition"
)

flags.DEFINE_integer(
    "g_sample_num", 10, "Coalition sample number"
)

flags.DEFINE_integer(
    "min_len", 8, "min length of sentence"
)

flags.DEFINE_integer(
    "max_len", 12, "max length of sentence"
)

flags.DEFINE_string(
    "seg_len_range", "3,4", "length range of chosen seg"
)

nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

flags.DEFINE_string(
    "tenboard_dir", f"./logs/{nowTime}/", "directory of tensorboard"
)
tenboard_dir = FLAGS.tenboard_dir

m_cnt = FLAGS.m_cnt
g_sample_num = FLAGS.g_sample_num
l_step = 1
min_len = FLAGS.min_len
max_len = FLAGS.max_len
seg_len_range = eval(FLAGS.seg_len_range)

random.seed(FLAGS.epoch_num + FLAGS.m_cnt * 100)
np.random.seed(FLAGS.epoch_num + FLAGS.m_cnt * 100)
tf.set_random_seed(FLAGS.epoch_num + FLAGS.m_cnt * 100)


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
    # sample p based on g
    sample_res = np.zeros_like(pij)
    for i in range(len(pij)):
        if random.random() < pij[i]:
            sample_res[i] = 1
    return sample_res


def measure_coalition(g_sample):
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


def manage_a_sentence(tokens_a, seg, feature, model):

    seg_len = seg[1] - seg[0]
    a_len = seg[2]

    p_mask = np.zeros(a_len-1)
    p_mask[seg[0]:seg[1]-1] = 1

    # =================================================================================================
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session()

        tmp = [0.0] * (a_len - 1)
        pij_weights = tf.Variable(tmp)  # initial value of p before sigmoid

        # pij_weights = tf.Variable(tf.random.normal([a_len-1]))
        pij_weights_ = tf.sigmoid(pij_weights)  # add sigmoid

        pij_masked = tf.where(p_mask > 0, pij_weights_, tf.zeros_like(pij_weights_))  # freeze ps out of seg

        tf.summary.histogram("pij", pij_masked[seg[0]:seg[1]])
        for i in range(seg_len - 1):
            tf.summary.scalar("p_%d" % i, pij_masked[seg[0] + i])

        p_c = pij_masked[seg[0]:seg[1] - 1]
        p_seg = tf.concat([[[0.0]], [p_c]], axis=1)[0, :]  # ensure that the number of ps is same as the number of words

        overall_expected = tf.placeholder(shape=[seg_len, 4], dtype=tf.float32)

        phi_c = overall_expected[:, 0] * p_seg \
                + overall_expected[:, 1] * (1 - p_seg) \
                - overall_expected[:, 2] * p_seg \
                - overall_expected[:, 3] * (1 - p_seg)
        g_score = tf.reduce_sum(phi_c)

        if FLAGS.maximize_shap:
            totloss = tf.negative(g_score)
        else:
            totloss = g_score

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.lr, global_step, 10, 0.95)
        my_opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_step = my_opt.minimize(totloss, global_step=global_step)

        tf.summary.scalar("total_loss", totloss)

        merged_summary_op = tf.summary.merge_all()

        # =======================================================================================================================

        init = tf.global_variables_initializer()
        sess.run(init)
        # loss_list = []

        item_list = [i for i in range(a_len)]

        res = []

        for epoch in range(FLAGS.epoch_num):
            pij = sess.run(pij_masked)  # numpy ndarray

            clist = pij_coals(pij, seg=seg)

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
            # print("coalition:", words)

            score_exp_list = []
            for g_ in range(g_sample_num):
                g_sample = g_sample_bern(pij)  # sample g
                g_clist = pij_coals(g_sample, seg=seg)  # partition the coalition based on g

                score_exp_items = []
                score_item = [0.0, 0.0]

                for cIdx, coal in enumerate(g_clist):

                    if coal[0] < seg[0] or coal[0] >= seg[1]:  # out of seg?
                        continue

                    positions_dict = get_masks_sampleshapley(g_clist, cIdx, a_len, m_cnt)  # sample S
                    positions_dict = exclude_mask(positions_dict, coal, seg)

                    scores_c_s, scores_c_si = compute_scores_seperate(positions_dict, feature, a_len,
                                                                      model.predict)

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

            _, _loss, summary_str, lr = sess.run(
                [train_step, totloss, merged_summary_op, learning_rate], feed_dict=in_dict)

            print('epoch:', epoch, '-->loss:', _loss,  '-->learning_rate:', lr, "\n")

            res.append({"p": pij.tolist(), "loss": float(_loss)})
        return res


def main(_):
    # -------------------- configuration ------------------------- #
    tf.logging.set_verbosity(tf.logging.INFO)
    task_name = FLAGS.task_name.lower()
    processors = {
        "sst-2": extract.Sst2Processor,
        "cola": extract.ColaProcessor,
    }
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    # ------------------- preprocess dataset -------------------- #
    label_list = processor.get_labels()
    num_labels = len(label_list)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    max_seq_length = FLAGS.max_seq_length

    # prepare valid dataset
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    if not os.path.exists(eval_file):
        extract.save_tfrecord(eval_examples, label_list, max_seq_length, tokenizer, eval_file)
    else:
        print('eval_tfrecord exists')

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(eval_examples))
    # ----------------------- build model --------------------- #

    # sess1
    bert_model = TextModel(FLAGS.bert_config_file, FLAGS.init_checkpoint, max_seq_length, num_labels)
    bert_model.start_session()

    print('Making explanations...')
    # for (i, example) in enumerate(eval_examples[:1]):
    # ==============================================================================

    res = []
    res.append({"lr":FLAGS.lr, "g_sample_num":g_sample_num, "m_cnt":m_cnt, "epoch_num": FLAGS.epoch_num, "maximize": FLAGS.maximize_shap})

    count = 0
    for id, sentence in enumerate(eval_examples):
        dic = {}
        tokens_a = tokenizer.tokenize(sentence.text_a)

        dic["id"] = id
        dic["tokens"] = tokens_a

        a_len = len(tokens_a)
        if a_len < min_len or a_len > max_len:
            continue
        count += 1
        print(count)
        # print(count)

        print(id, tokens_a)

        seg_len = random.choice(seg_len_range)
        seg = [0, 0, a_len]
        seg[0] = random.choice(range(a_len-seg_len))
        seg[1] = seg[0] + seg_len

        # print(seg,"\n\n\n\n")

        # input feature of the sentence to BERT model
        feature = extract.convert_single_example(0, sentence, label_list, max_seq_length, tokenizer)

        # print("\nCurrent words:", tokens_a[seg[0]:seg[1]])

        dic["seg"] = seg

        FLAGS.maximize_shap = True
        opt_res_1 = manage_a_sentence(tokens_a, seg, feature, bert_model)
        FLAGS.maximize_shap = False
        opt_res_2 = manage_a_sentence(tokens_a, seg, feature, bert_model)

        opt_res = []
        for i in range(len(opt_res_1)):
            item = {"p_max": opt_res_1[i]["p"],
                    "p_min": opt_res_2[i]["p"],
                    "loss": -1 * opt_res_1[i]["loss"] - opt_res_2[i]["loss"]
                    }
            opt_res.append(item)

        dic["opt_res"] = opt_res

        min_gt_score, max_gt_score, min_gt_part, max_gt_part = get_min_max_shap(seg, feature, bert_model)
        gt_score = max_gt_score - min_gt_score
        dic["gt_score"] = gt_score

        difference = []
        for i in range(FLAGS.epoch_num//l_step):

            opt_score = 0
            for j in range(i*l_step,(i+1)*l_step):
                opt_score += abs(opt_res[j]["loss"])
            opt_score /= l_step

            difference.append(abs(gt_score-opt_score))

        dic["difference"] = difference
        res.append(dic)

        print("gt_score:", gt_score)
        with open('difference_%s.json'%FLAGS.task_name, 'w') as f:
            json.dump(res, f)


if __name__ == "__main__":
    # parse flags and run main()
    tf.app.run()
