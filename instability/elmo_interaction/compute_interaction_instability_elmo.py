import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
import copy
import random

import sys
sys.path.append("../..")
from build_model import TextModel
from shapley import *
from preprocess import tokenization, extract
from elmo.elmo_data import _load_shard_cola, _load_shard_sst

from tensorflow.python.framework import ops
from keras.utils import to_categorical
from keras.layers import Input
from keras.models import load_model
from keras.models import Model
from tensorflow.python.keras.backend import set_session
import time

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("task_name", "cola", "Data source")

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

if FLAGS.data_dir is None:
    if FLAGS.task_name == "cola":
        FLAGS.data_dir = "../../GLUE_data/CoLA/dev.tsv"
    else:
        FLAGS.data_dir = "../../GLUE_data/SST-2/dev.tsv"

flags.DEFINE_string(
    "load_path", None,
    "The pretrained model for Elmo."
)

if FLAGS.load_path is None:
    if FLAGS.task_name == "cola":
        FLAGS.load_path = "../../models/Elmo/cola/model/my_weight_cola.h5"
    else:
        FLAGS.load_path = "../../models/Elmo/sst-2/model/my_weight_sst2.h5"

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
    "epoch_num", 50, "Total number of training epochs."
)

flags.DEFINE_float(
    "lr", 2.0, "The learning rate."
)

# ===================== hyperparameters for computing instability =======
flags.DEFINE_integer(
    "sentence_num", 10, "The number of sentences used for the computation of instability."
)

flags.DEFINE_integer(
    "min_len", 8, "The minimum length of the sentence in the computation of the instability"
)

flags.DEFINE_integer(
    "max_len", 12, "The maximum length of the sentence in the computation of the instability"
)

flags.DEFINE_integer(
    "test_num", 5, "The number of tests to compute the instability repeatedly."
)

flags.DEFINE_string(
    "inter_date", time.strftime("%Y-%m-%d", time.localtime()), "The date folder that contains the interaction result %yyyy-%mm-%dd"
)

inter_date = FLAGS.inter_date

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


def exclude_mask(position_dict, coal, seg):

    if isinstance(coal, int):
        coal = [coal]
    masks = copy.deepcopy(position_dict[(0, 0)])
    masks_i = copy.deepcopy(position_dict[(0, 1)])
    for mask in masks:
        for i in range(seg[0], seg[1]):
            mask[i] = 0
    for mask_i in masks_i:
        for i in range(seg[0], seg[1]):
            if i not in coal:
                mask_i[i] = 0
    new_position_dict = {
        (0, 0): masks,
        (0, 1): masks_i
    }
    return new_position_dict

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
    # ??????p??????g
    sample_res = np.zeros_like(pij)
    for i in range(len(pij)):
        if random.random() < pij[i]:
            sample_res[i] = 1
    return sample_res

#
# def measure_coalition(g_sample):
#     # ????????????coalition???????????????lambda_i??????
#     """
#     :param g_sample: sampled g
#     :return: lambda i
#     """
#     sizes = [1 for i in range(len(g_sample)+1)]
#     l = 0
#     size = 0
#     for i in range(len(g_sample)):
#         if g_sample[i] > 0:
#             size += 1
#         else:
#             for j in range(l,i+1):
#                 sizes[j] += size
#             size = 0
#             l = i+1
#     for j in range(l, len(sizes)):
#         sizes[j] += size
#     return sizes


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


def main(_):
    # -------------------- configuration ------------------------- #
    sentence_num = FLAGS.sentence_num
    min_len = FLAGS.min_len
    max_len = FLAGS.max_len
    test_num = FLAGS.test_num
    m_cnts = [50]
    g_sample_nums = [50, 40, 30, 20, 10]  # Different sampling numbers
    tf.set_random_seed(FLAGS.epoch_num + sentence_num * 100)
    random.seed(FLAGS.epoch_num + sentence_num * 100)
    np.random.seed(FLAGS.epoch_num + sentence_num * 100)
    tf.logging.set_verbosity(tf.logging.INFO)
    task_name = FLAGS.task_name.lower()
    model_name = "elmo"
    processors = {
        "sst-2": extract.Sst2Processor,
        "cola": extract.ColaProcessor,
    }
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    date = time.strftime("%Y-%m-%d", time.localtime())

    # --------------Result-----------------------------------------
    res = []
    result_path = "../result/instability/%s/%s/%s/" % (date, model_name, task_name)
    interaction_path = "../result/interaction/%s/%s/%s/" % (inter_date, model_name, task_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    res.append({"interaction_path": interaction_path, "m_cnts": m_cnts, "g_sample_nums": g_sample_nums})

    # ------------------- preprocess dataset -------------------- #
    if FLAGS.task_name == 'sst-2':
        sentences, labels = _load_shard_sst(FLAGS.data_dir)
    if FLAGS.task_name == 'cola':
        sentences, labels = _load_shard_cola(FLAGS.data_dir)
    sentences_input = np.array(sentences)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(sentences))

    # prepare interaction and trained pij --------------------------------#
    interaction = np.load(interaction_path + "result_%s_%s.npy" % (task_name, model_name), allow_pickle=True)
    # ----------------------- build model --------------------- #

    # sess1
    elmo = hub.Module(spec='../../elmo/tf_module', trainable=False)
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
    for (i, sentence) in enumerate(sentences):
        for j in range(1, len(interaction)):
            if interaction[j]['id'] == i:
                print(i)
                # sentence = eval_examples[0] # ???????????????
                # tokens_a = tokenizer.tokenize(sentence.text_a)
                label = int(labels[i])
                sentence = sentences[i]
                splitted = sentence.split()

                a_len = len(splitted)
                if a_len < min_len or a_len > max_len:
                    continue
                count += 1
                # continue

                logit = output_logits[i]
                embedding = embedding_model.predict(np.array([sentence, ""]))[0]  # cannot set batchsize=1? why?
                time_step = embedding.shape[0]

                # ========== predictor model =======================================
                idx = 2
                layer_input = Input(shape=(time_step, 1024))
                # print(layer_input)
                x = layer_input
                for layer in Elmo_model.layers[idx:-1]:
                    x = layer(x)
                predictor_model = Model(layer_input, x)
                # _ = predictor_model.predict(np.random.randn(10, time_step, 1024))
                with predictor_model.input.graph.as_default():
                    predictor_model.sess = tf.Session(graph=predictor_model.input.graph)
                    predictor_model.sess.run(tf.global_variables_initializer())
                print(predictor_model.summary())
                # ===================================================================

                # if count > sentence_num:
                #     break
                # input feature of the sentence to BERT model
                # feature = extract.convert_single_example(0, sentence, label_list, max_seq_length, tokenizer)
                seg = interaction[j]['seg']
                seg_len = seg[1] - seg[0]
                p_mask = np.zeros(a_len-1)
                p_mask[seg[0]:seg[1]-1] = 1
                print("\nCurrent words:", splitted[seg[0]:seg[1]])

                sentence_result = {}
                sentence_result["id"] = i
                sentence_result["seg"] = seg
                # sentence_result["result"] = {"True": [], "False": []}
                sentence_result["result"] = []

                weight_decay = 0  # ???????????????????????????0
                for m_cnt in m_cnts:
                    for g_sample_num in g_sample_nums:
                        sentence_result["result"].append({"m_cnt": m_cnt, "g_sample_num": g_sample_num, "True": [], "False": []})
                        #==========================================================================
                        for mode in [True, False]:
                            score_summary = [[], []]
                            g_sample_summary = []
                            for t in range(test_num):
                                g = tf.Graph()
                                with g.as_default():
                                    sess = tf.Session()

                                    # tmp = [0.0] * (a_len-1)
                                    tmp = interaction[j]["pij"][str(mode)][-1]
                                    pij_weights_ = tf.Variable(tmp)  # pi???????????????????????????sigmoid??????
                                    pij_masked = tf.where(p_mask>0, pij_weights_, tf.zeros_like(pij_weights_)) # ??????seg?????????pi


                                    tf.summary.histogram("pij", pij_masked[seg[0]:seg[1]])
                                    for ii in range(seg_len-1):
                                        tf.summary.scalar("p_%d"%ii, pij_masked[seg[0]+ii])

                                    p_c = pij_masked[seg[0]:seg[1]-1]
                                    p_seg = tf.concat([ [[0.0]], [p_c] ], axis=1)[0,:]  # ???seg????????????0?????????pi?????????i??????
                                    overall_expected = tf.placeholder(shape=[seg_len, 4], dtype=tf.float32)

                                    phi_c = overall_expected[:, 0] * p_seg\
                                            +overall_expected[:, 1] * (1 - p_seg)\
                                            -overall_expected[:, 2] * p_seg\
                                            -overall_expected[:, 3] * (1 - p_seg)
                                    g_score = tf.reduce_sum(phi_c)

                                    mean, variance = tf.nn.moments(pij_masked[seg[0]:seg[1]-1], axes=0)
                                    var_loss = -weight_decay * variance
                                    tf.summary.scalar("variance", var_loss)

                                    # ????????????Shapley value??????????????????loss???????????????minimize
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
                                    my_opt = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9)
                                    train_step = my_opt.minimize(totloss, global_step=global_step)

                                    tf.summary.scalar("loss", loss)
                                    tf.summary.scalar("total_loss", totloss)

                                    merged_summary_op = tf.summary.merge_all()

                                    #=======================================================================================================================

                                    init = tf.global_variables_initializer()
                                    sess.run(init)
                                    # loss_list = []

                                    item_list = [ii for ii in range(a_len)]

                                    pij = sess.run(pij_masked)  # numpy ndarray
                                    # pij = 1 / (1 + np.exp(-pij))
                                    # clist = pij_coals(pij)

                                    clist = pij_coals(pij, seg=seg)  # ??????pi??????????????????coalition????????????????????????

                                    words = []
                                    for coal in clist:
                                        if len(coal)==1:
                                            item = coal[0]
                                        else:
                                            item = coal
                                        if isinstance(item, int):
                                            words.append(splitted[item])
                                        else:
                                            tmp = []
                                            for id in item:
                                                tmp.append(splitted[id])
                                            words.append(tmp)

                                    print('pij', pij, clist)
                                    print("coalition:", words)

                                    score_exp_list = []
                                    coal_size_list = []

                                    for g_ in range(g_sample_num):
                                        score_exp_items = []
                                        if g_sample_nums.index(g_sample_num) == 0:
                                            g_sample = g_sample_bern(pij) # ??????g

                                            g_clist = pij_coals(g_sample, seg=seg)  # ??????g??????????????????coalition

                                            score_item = [0.0, 0.0]
                                            positions_dicts = {}
                                            positions_dicts[(0, 0)] = []
                                            positions_dicts[(0, 1)] = []

                                            for cIdx, coal in enumerate(g_clist):
                                                if coal[0] < seg[0] or coal[0] >= seg[1]: # ?????????seg???
                                                    continue

                                                positions_dict = get_masks_sampleshapley(g_clist, cIdx, a_len, m_cnt) # ??????S
                                                positions_dict = exclude_mask(positions_dict, coal, seg)

                                                positions_dicts[(0, 0)].extend(positions_dict[(0, 0)])
                                                positions_dicts[(0, 1)].extend(positions_dict[(0, 1)])

                                            scores_c_s, scores_c_si = compute_scores_seperate_elmo(positions_dict, embedding, label, predictor_model)
                                            # score_diff = scores_c_si - scores_c_s

                                            score_summary[0].extend(scores_c_si)
                                            score_summary[1].extend(scores_c_s)
                                            g_sample_summary.append(g_sample)
                                        else:
                                            score_summary = np.load(result_path + "/sample_pool_%s.npy" % str(mode), allow_pickle=True)
                                            g_sample_summary = score_summary[1]
                                            g_idx = np.random.choice(np.arange(len(g_sample_summary)), 1, replace=False)
                                            g_idx = int(g_idx)
                                            g_sample = g_sample_summary[g_idx]
                                            st = g_idx * m_cnt
                                            # idx = np.random.choice(np.arange(st, st + m_cnt), m_cnt, replace=False)
                                            scores_c_si = score_summary[0][0][st: st + m_cnt]
                                            scores_c_s = score_summary[0][1][st: st + m_cnt]
                                        score_item[0] += np.mean(scores_c_si)
                                        score_item[1] += np.mean(scores_c_s)


                                        score_item[0] /= seg_len # ????????????i??????
                                        score_item[1] /= seg_len

                                        for idx, item in enumerate(item_list[seg[0]:seg[1]]): # ?????????i????????????
                                            score_exp = compute_sum(score_item[1], score_item[0], g_sample, item)
                                            score_exp_items.append(score_exp)

                                        score_exp_list.append(score_exp_items)

                                    overall_exp_score = cal_overall_exp(score_exp_list)

                                    in_dict = {
                                            overall_expected: overall_exp_score
                                        }

                                    _, _loss, summary_str, lr, var, g_score_, phi_c_ = sess.run([train_step, loss, merged_summary_op, learning_rate, var_loss, g_score, phi_c], feed_dict=in_dict)

                                    sentence_result["result"][-1][str(mode)].append(_loss)

                                    print('test:', t, '-->loss:', _loss, '-->variance_loss:', var, '-->learning_rate:', lr, "\n")
                            if g_sample_nums.index(g_sample_num) == 0:
                                score_summary = np.array(score_summary)
                                g_sample_summary = np.array(g_sample_summary)
                                np.save(result_path + "/sample_pool_%s.npy" % str(mode), [score_summary, g_sample_summary])

                res.append(sentence_result)

    np.save('%s/result_%s_elmo.npy' % (result_path, FLAGS.task_name), res)


if __name__ == "__main__":
    # parse flags and run main()
    tf.app.run()
