import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
import copy
import random
import pickle
import json
from tensorflow.python.framework import ops
from keras.utils import to_categorical
from keras.layers import Input
from keras.models import load_model
from keras.models import Model
from tensorflow.python.keras.backend import set_session

import sys
sys.path.append("..")
from truth_interaction_GT_elmo import calculate_shap_sum, get_min_max_shap
from elmo.elmo_data import _load_shard_cola, _load_shard_sst
from preprocess import tokenization, extract
from shapley import *
from coalition_utils import *
from build_model import TextModel

resume = False
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("task_name", "sst-2", "The name of the task to train. [cola, sst-2]")
assert FLAGS.task_name in ["sst-2", "cola"]

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

if FLAGS.data_dir is None:
    if FLAGS.task_name == "sst-2":
        FLAGS.data_dir = "../GLUE_data/SST-2/dev.tsv"
    else:
        FLAGS.data_dir = "../GLUE_data/CoLA/dev.tsv"

flags.DEFINE_string(
    "load_path", None, "trained elmo model path"
)

if FLAGS.load_path is None:
    if FLAGS.task_name == "sst-2":
        FLAGS.load_path = "../models/Elmo/sst-2/model/my_weight_sst2.h5"
    else:
        FLAGS.load_path = "../models/Elmo/cola/model/my_weight_cola.h5"

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

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

flags.DEFINE_string(
    "resume_path", None, "resume form previous result"
)

if FLAGS.resume_path is None:
    resume = False
else:
    resume = True

m_cnt = FLAGS.m_cnt
g_sample_num = FLAGS.g_sample_num
l_step = 1
min_len = FLAGS.min_len
max_len = FLAGS.max_len
seg_len_range = eval(FLAGS.seg_len_range)

random.seed(FLAGS.epoch_num + m_cnt * 100)
np.random.seed(FLAGS.epoch_num + m_cnt * 100)
tf.set_random_seed(FLAGS.epoch_num + m_cnt * 100)


def compute_scores_seperate_elmo(positions_dict, embedding, label, predictor):
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
    return logits[:, label], logits_i[:, label]


def manage_a_sentence(seg, embedding, label, predictor_model):

    seg_len = seg[1] - seg[0]
    a_len = seg[2]

    p_mask = np.zeros(a_len-1)
    p_mask[seg[0]:seg[1]-1] = 1

    # =================================================================================================
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session()

        tmp = [0.0] * (a_len - 1)
        pij_weights = tf.Variable(tmp)  # initial value of p, before sigmoid

        # pij_weights = tf.Variable(tf.random.normal([a_len-1]))
        pij_weights_ = tf.sigmoid(pij_weights)  # add sigmoid

        pij_masked = tf.where(p_mask > 0, pij_weights_, tf.zeros_like(pij_weights_))  # freeze ps out of the seg

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
        # my_opt = tf.train.GradientDescentOptimizer(learning_rate)
        my_opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_step = my_opt.minimize(totloss, global_step=global_step)

        tf.summary.scalar("total_loss", totloss)

        merged_summary_op = tf.summary.merge_all()

        # =======================================================================================================================

        init = tf.global_variables_initializer()
        sess.run(init)

        item_list = [i for i in range(a_len)]

        res = []

        for epoch in range(FLAGS.epoch_num):
            pij = sess.run(pij_masked)  # numpy ndarray

            clist = pij_coals(pij, seg=seg)  # partition the coalition based on g

            print('pij', pij, clist)
            # print("coalition:", words)

            score_exp_list = []
            for g_ in range(g_sample_num):
                g_sample = g_sample_bern(pij)  # 采样g
                g_clist = pij_coals(g_sample, seg=seg)  # 根据g采样结果划分coalition
                # print("g_sample:", g_sample)
                # print("coal_size:", coal_size)

                score_exp_items = []
                score_item = [0.0, 0.0]

                for cIdx, coal in enumerate(g_clist):
                    # new_list, cIdx = get_new_list(item, item_list)

                    if coal[0] < seg[0] or coal[0] >= seg[1]:  # 是否在seg内
                        continue

                    positions_dict = get_masks_sampleshapley(g_clist, cIdx, a_len, m_cnt)  # 采样S
                    positions_dict = exclude_mask(positions_dict, coal, seg)

                    scores_c_s, scores_c_si = compute_scores_seperate_elmo(positions_dict, embedding, label,
                                                                           predictor_model)

                    # score_diff = scores_c_si - scores_c_s

                    score_item[0] += np.mean(scores_c_si)
                    score_item[1] += np.mean(scores_c_s)

                score_item[0] /= seg_len  # 计算每个i均值
                score_item[1] /= seg_len

                for idx, item in enumerate(item_list[seg[0]:seg[1]]):  # 求每个i对应期望
                    score_exp = compute_sum(score_item[1], score_item[0], g_sample, item)
                    score_exp_items.append(score_exp)

                score_exp_list.append(score_exp_items)

            overall_exp_score = cal_overall_exp(score_exp_list)
            # print(overall_exp_score)

            in_dict = {
                overall_expected: overall_exp_score
            }

            _, _loss, summary_str, lr = sess.run(
                [train_step, totloss, merged_summary_op, learning_rate], feed_dict=in_dict)

            print('epoch:', epoch, '-->loss:', _loss, '-->learning_rate:', lr, "\n")

            res.append({"p": pij.tolist(), "loss": float(_loss)})
        return res


def main(_):
    # -------------------- configuration ------------------------- #
    tf.logging.set_verbosity(tf.logging.INFO)
    task_name = FLAGS.task_name.lower()
    model_name = "elmo"
    processors = {
        "sst-2": extract.Sst2Processor,
        "cola": extract.ColaProcessor,
    }
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    # ------------------- preprocess dataset -------------------- #
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if FLAGS.task_name == 'sst-2':
        sentences, labels = _load_shard_sst(FLAGS.data_dir)
    if FLAGS.task_name == 'cola':
        sentences, labels = _load_shard_cola(FLAGS.data_dir)
    sentences_input = np.array(sentences)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(sentences))
    # ----------------------- build model --------------------- #

    # sess1
    elmo = hub.Module(spec='../elmo/tf_module', trainable=False)
    Elmo_model = load_model(FLAGS.load_path, custom_objects={'elmo': elmo, "tf": tf})
    print(Elmo_model.summary())

    dense_layer_model = Model(inputs=Elmo_model.input, outputs=Elmo_model.get_layer('dense_2').output)
    output_logits = dense_layer_model.predict(sentences_input)

    embedding_model = Model(inputs=Elmo_model.input, outputs=Elmo_model.get_layer('lambda_1').output)
    # embeddings = embedding_model.predict(sentences_input)
    count = 0

    print('Making explanations...')
    # for (i, example) in enumerate(eval_examples[:1]):
    # ==============================================================================

    res = []
    res.append({"lr":FLAGS.lr, "g_sample_num":g_sample_num, "m_cnt":m_cnt, "epoch_num": FLAGS.epoch_num, "maximize": FLAGS.maximize_shap})

    if resume:
        with open(FLAGS.resume_path,"r") as f:
            res = json.load(f)
        count = len(res) - 1
        start = res[-1]["id"] + 1
    else:
        start = 0
        count = 0

    for i, sentence in enumerate(sentences[start:]):
        id = i + start
        dic = {}
        # sentence = eval_examples[0] # 分析的句子
        # tokens_a = tokenizer.tokenize(sentence.text_a)
        label = int(labels[id])
        sentence = sentences[id]
        splitted = sentence.split()
        logit = output_logits[id]
        embedding = embedding_model.predict(np.array([sentence, ""]))[0]  # cannot set batchsize=1? why?
        time_step = embedding.shape[0]

        # ========== predictor model =======================================
        idx = 2
        layer_input = Input(shape=(time_step, 1024))
        # print(layer_input)
        x = layer_input
        for l, layer in enumerate(Elmo_model.layers[idx:-1]):
            x = layer(x)
            print(l, x.shape)
        predictor_model = Model(layer_input, x)
        # _ = predictor_model.predict(np.random.randn(10, time_step, 1024))
        with predictor_model.input.graph.as_default():
            predictor_model.sess = tf.Session(graph=predictor_model.input.graph)
            predictor_model.sess.run(tf.global_variables_initializer())
        print(predictor_model.summary())
        # ===================================================================


        dic["id"] = id
        dic["tokens"] = splitted

        a_len = len(splitted)
        if a_len < min_len or a_len > max_len:
            continue
        count += 1
        print(count)
        # print(count)

        print(id, splitted)

        seg_len = random.choice(seg_len_range)
        seg = [0, 0, a_len]
        seg[0] = random.choice(range(a_len-seg_len))
        seg[1] = seg[0] + seg_len

        dic["seg"] = seg

        # opt_res = manage_a_sentence(seg, embedding, label, predictor_model)
        # # print(res)

        FLAGS.maximize_shap = True
        opt_res_1 = manage_a_sentence(seg, embedding, label, predictor_model)
        FLAGS.maximize_shap = False
        opt_res_2 = manage_a_sentence(seg, embedding, label, predictor_model)

        opt_res = []
        for i in range(len(opt_res_1)):
            item = {"p_max": opt_res_1[i]["p"],
                    "p_min": opt_res_2[i]["p"],
                    "loss": -1 * opt_res_1[i]["loss"] - opt_res_2[i]["loss"]
                    }
            opt_res.append(item)

        dic["opt_res"] = opt_res
        min_gt_score, max_gt_score, min_gt_part, max_gt_part = get_min_max_shap(seg, embedding, label, predictor_model)
        gt_score = max_gt_score - min_gt_score
        dic["gt_score"] = gt_score

        difference = []
        for i in range(FLAGS.epoch_num//l_step):

            opt_score = 0
            for j in range(i*l_step,(i+1)*l_step):
                if FLAGS.maximize_shap:
                    opt_score += -1* opt_res[j]["loss"]
                else:
                    opt_score += opt_res[j]["loss"]
            opt_score /= l_step
            # step_dict = {"gt_score": gt_score, "diff": abs(gt_score-opt_score)}

            difference.append(abs(gt_score-opt_score))

        dic["difference"] = difference
        res.append(dic)
        print("gt_score:", gt_score)
        with open('difference_%s_elmo.json'%FLAGS.task_name, 'w') as f:
            json.dump(res, f)
    print(sentences)


if __name__ == "__main__":
    # parse flags and run main()
    tf.app.run()
