import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
import copy
import random
from tensorflow.python.framework import ops
import sys

sys.path.append("../..")
from build_model import TextModel
from shapley import *
from coalition_utils import *
from preprocess import tokenization, extract
from keras.utils import to_categorical
from keras.layers import Input
from keras.models import load_model
from keras.models import Model
from elmo.elmo_data import _load_shard_cola, _load_shard_sst
from tensorflow.python.keras.backend import set_session
import time

# max_score = [[0 for _ in range(test_num)] for _ in range(5)]
# min_score = [[0 for _ in range(test_num)] for _ in range(5)]

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("task_name", "sst-2", "The name of the task to train.")
assert FLAGS.task_name in ["cola", "sst-2"]

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
    "sentence_num", 20, "The number of sentences used for the computation of instability."
)

flags.DEFINE_integer(
    "min_len", 8, "The minimum length of the sentence in the computation of the instability"
)

flags.DEFINE_integer(
    "max_len", 12, "The maximum length of the sentence in the computation of the instability"
)

flags.DEFINE_integer(
    "test_num", 1, "The number of tests to compute the instability repeatedly."
)

sentence_num = FLAGS.sentence_num
min_len = FLAGS.min_len
max_len = FLAGS.max_len
test_num = FLAGS.test_num


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


def main(_):
    # -------------------- configuration ------------------------- #
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
    res.append({"lr": FLAGS.lr, "epoch_num": FLAGS.epoch_num})

    result_path = "../result/interaction/%s/%s/%s/" % (date, model_name, task_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # ------------------- preprocess dataset -------------------- #
    if FLAGS.task_name == 'sst-2':
        sentences, labels = _load_shard_sst(FLAGS.data_dir)
    if FLAGS.task_name == 'cola':
        sentences, labels = _load_shard_cola(FLAGS.data_dir)
    sentences_input = np.array(sentences)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(sentences))
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
        print(i)
        # sentence = eval_examples[0] # 分析的句子
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

        if count > sentence_num:
            break
        # input feature of the sentence to BERT model
        # feature = extract.convert_single_example(0, sentence, label_list, max_seq_length, tokenizer)
        left_pos = random.randint(0, a_len // 2)
        right_pos = min(a_len - 1, left_pos + random.randint(4, 6))
        # seg = (2, 6, a_len)  #左闭右开
        seg = (left_pos, right_pos, a_len)  #左闭右开
        seg_len = seg[1] - seg[0]
        p_mask = np.zeros(a_len-1)
        p_mask[seg[0]:seg[1]-1] = 1
        print("\nCurrent words:", splitted[seg[0]:seg[1]])

        sentence_result = {}
        sentence_result["id"] = i
        sentence_result["seg"] = seg
        sentence_result["loss"], sentence_result["pij"] = {"True": [], "False": []}, {"True": [], "False": []}

        m_cnts = [50]
        for m_cnt in m_cnts:
            # m_cnt = 50  # 对于每个g，采样50次S
            # n_steps = 10
            # m_cnt = int(m_cnt/(2*n_steps-1))*(2*n_steps-1)
            weight_decay = 0  # 现在没有用到，设为0

            g_sample_num = 50  # 采样g的次数

            #==========================================================================
            for t in range(test_num):
                for mode in [True, False]:
                    sentence_result["m_cnt"] = m_cnt
                    sentence_result["g_sample_num"] = g_sample_num
                    g = tf.Graph()
                    with g.as_default():
                        sess = tf.Session()

                        tmp = [0.0] * (a_len-1)
                        pij_weights = tf.Variable(tmp) # pi初始值，之后会经过sigmoid计算

                        # pij_weights = tf.Variable(tf.random.normal([a_len-1]))
                        pij_weights_ = tf.sigmoid(pij_weights)  # add sigmoid

                        pij_masked = tf.where(p_mask>0, pij_weights_, tf.zeros_like(pij_weights_)) # 冻结seg之外的pi


                        tf.summary.histogram("pij", pij_masked[seg[0]:seg[1]])
                        for ii in range(seg_len-1):
                            tf.summary.scalar("p_%d"%ii, pij_masked[seg[0]+ii])

                        p_c = pij_masked[seg[0]:seg[1]-1]
                        p_seg = tf.concat([ [[0.0]], [p_c] ], axis=1)[0,:]  # 在seg最前方补0，使得pi个数与i相同
                        overall_expected = tf.placeholder(shape=[seg_len, 4], dtype=tf.float32)

                        phi_c = overall_expected[:, 0] * p_seg\
                                +overall_expected[:, 1] * (1 - p_seg)\
                                -overall_expected[:, 2] * p_seg\
                                -overall_expected[:, 3] * (1 - p_seg)
                        g_score = tf.reduce_sum(phi_c)

                        mean, variance = tf.nn.moments(pij_masked[seg[0]:seg[1]-1], axes=0)
                        var_loss = -weight_decay * variance
                        tf.summary.scalar("variance", var_loss)

                        # 若最大化Shapley value的和，则需取loss的相反数再minimize
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

                        for epoch in range(FLAGS.epoch_num):
                            pij = sess.run(pij_masked)  # numpy ndarray
                            # pij = 1 / (1 + np.exp(-pij))
                            # clist = pij_coals(pij)

                            clist = pij_coals(pij, seg=seg)  # 根据pi按照阈值划分coalition，仅作输出观察用

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
                                g_sample = g_sample_bern(pij) # 采样g

                                g_clist = pij_coals(g_sample, seg=seg)  # 根据g采样结果划分coalition
                                score_exp_items = []
                                score_item = [0.0, 0.0]
                                positions_dicts = {}
                                positions_dicts[(0, 0)] = []
                                positions_dicts[(0, 1)] = []

                                for cIdx, coal in enumerate(g_clist):
                                    # new_list, cIdx = get_new_list(item, item_list)

                                    if coal[0] < seg[0] or coal[0] >= seg[1]: # 是否在seg内
                                        continue

                                    positions_dict = get_masks_sampleshapley(g_clist, cIdx, a_len, m_cnt) # 采样S
                                    positions_dict = exclude_mask(positions_dict, coal, seg)

                                    positions_dicts[(0, 0)].extend(positions_dict[(0, 0)])
                                    positions_dicts[(0, 1)].extend(positions_dict[(0, 1)])

                                scores_c_s, scores_c_si = compute_scores_seperate_elmo(positions_dict, embedding, label, predictor_model)
                                # score_diff = scores_c_si - scores_c_s

                                score_item[0] += np.mean(scores_c_si)
                                score_item[1] += np.mean(scores_c_s)


                                score_item[0] /= seg_len # 计算每个i均值
                                score_item[1] /= seg_len

                                for idx, item in enumerate(item_list[seg[0]:seg[1]]): # 求每个i对应期望
                                    score_exp = compute_sum(score_item[1], score_item[0], g_sample, item)
                                    score_exp_items.append(score_exp)

                                score_exp_list.append(score_exp_items)

                            overall_exp_score = cal_overall_exp(score_exp_list)

                            in_dict = {
                                    overall_expected: overall_exp_score
                                }

                            _, _loss, summary_str, lr, var, g_score_ = sess.run([train_step, loss, merged_summary_op, learning_rate, var_loss, g_score], feed_dict=in_dict)

                            sentence_result["loss"][str(mode)].append(_loss)
                            sentence_result["pij"][str(mode)].append(pij)

                            print('epoch:', epoch, '-->loss:', _loss, '-->variance_loss:', var, '-->learning_rate:', lr, "\n")
                    # if mode:
                    #     max_score[m_cnt // 20 - 1][t] = 0 - _loss
                    # else:
                    #     min_score[m_cnt // 20 - 1][t] = _loss
        res.append(sentence_result)

    np.save('%s/result_%s_elmo.npy' % (result_path, FLAGS.task_name), res)
    print("total number: %d" % count)


if __name__ == "__main__":
    # parse flags and run main()
    tf.app.run()
