import numpy as np
import tensorflow as tf
import os
import copy
import random
from build_model import TextModel
from tensorflow.python.framework import ops
from preprocess import tokenization, extract
from shapley import *
from coalition_utils import *
import datetime

bert_dir = "./models/uncased_L-12_H-768_A-12"

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", bert_dir + "/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "sst-2", "The name of the task to train. [cola or sst-2]")

flags.DEFINE_string(
    "init_checkpoint", "",
    "Initial checkpoint (usually from a pre-trained BERT model).")

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
    "m_cnt", 50, "Sample number for each coalition"
)

flags.DEFINE_integer(
    "g_sample_num", 10, "Coalition sample number"
)

nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

flags.DEFINE_string(
    "tenboard_dir", f"./logs/{nowTime}/", "directory of tensorboard"
)

flags.DEFINE_integer(
    "sentence_idx", 171, "The index of input sentence"
)

flags.DEFINE_integer(
    "seg_start", 2, "The beginning position of segmentation"
)

flags.DEFINE_integer(
    "seg_end", 5, "The end position of segmentation"
)

tenboard_dir = FLAGS.tenboard_dir
summary_writer = tf.summary.FileWriter(tenboard_dir)


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

    if FLAGS.init_checkpoint == "":
        if FLAGS.task_name == "sst-2":
            data_dir = "./GLUE_data/SST-2"
            FLAGS.init_checkpoint = data_dir + "/model/model.ckpt-6313"
        else:
            data_dir = "./GLUE_data/CoLA"
            FLAGS.init_checkpoint = data_dir + "/model/model.ckpt-801"

    processor = processors[task_name]()

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    # ------------------- preprocess dataset -------------------- #
    label_list = processor.get_labels()
    num_labels = len(label_list)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    max_seq_length = FLAGS.max_seq_length

    # prepare valid dataset
    eval_examples = processor.get_dev_examples(data_dir)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(eval_examples))
    # ----------------------- build model --------------------- #

    # sess1
    bert_model = TextModel(FLAGS.bert_config_file, FLAGS.init_checkpoint, max_seq_length, num_labels)
    bert_model.start_session()

    print('Making explanations...')
    # for (i, example) in enumerate(eval_examples[:1]):
    # ==============================================================================

    sentence = eval_examples[FLAGS.sentence_idx] # the input sentence
    tokens_a = tokenizer.tokenize(sentence.text_a)

    a_len = len(tokens_a)
    # input feature of the sentence to BERT model
    feature = extract.convert_single_example(0, sentence, label_list, max_seq_length, tokenizer)

    seg = (FLAGS.seg_start, FLAGS.seg_end, a_len)
    seg_len = seg[1] - seg[0]
    p_mask = np.zeros(a_len-1)
    p_mask[seg[0]:seg[1]-1] = 1
    print("\nCurrent words:", tokens_a[seg[0]:seg[1]])

    m_cnt = FLAGS.m_cnt
    g_sample_num = FLAGS.g_sample_num

#=================================================================================================
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session()

        summary_writer.add_graph(sess.graph)

        tmp = [0.0] * (a_len-1)
        pij_weights = tf.Variable(tmp)  # initial value of ps, before sigmoid

        # pij_weights = tf.Variable(tf.random.normal([a_len-1]))
        pij_weights_ = tf.sigmoid(pij_weights)  # add sigmoid

        pij_masked = tf.where(p_mask>0, pij_weights_, tf.zeros_like(pij_weights_))  # freeze pi out of selected seg


        tf.summary.histogram("pij", pij_masked[seg[0]:seg[1]])
        for i in range(seg_len-1):
            tf.summary.scalar("p_%d"%i, pij_masked[seg[0]+i])

        p_c = pij_masked[seg[0]:seg[1]-1]
        p_seg = tf.concat([ [[0.0]], [p_c] ], axis=1)[0,:]  # ensure the number of ps same as the number of words

        overall_expected = tf.placeholder(shape=[seg_len, 4], dtype=tf.float32)

        phi_c = overall_expected[:, 0] * p_seg\
                +overall_expected[:, 1] * (1 - p_seg)\
                -overall_expected[:, 2] * p_seg\
                -overall_expected[:, 3] * (1 - p_seg)
        g_score = tf.reduce_sum(phi_c)

        if FLAGS.maximize_shap:
            totloss = tf.negative(g_score)
        else:
            totloss = g_score

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.lr, global_step, 10, 1)

        my_opt = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9)
        train_step = my_opt.minimize(totloss, global_step=global_step)

        tf.summary.scalar("total_loss", totloss)

        merged_summary_op = tf.summary.merge_all()

#=======================================================================================================================

        init = tf.global_variables_initializer()
        sess.run(init)
        # loss_list = []

        item_list = [i for i in range(a_len)]

        for epoch in range(FLAGS.epoch_num):
            pij = sess.run(pij_masked)  # numpy ndarray

            clist = pij_coals(pij, seg=seg)

            words = []
            for coal in clist:
                if len(coal)==1:
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
            for g_ in range(g_sample_num):
                g_sample = g_sample_bern(pij)  # sample g

                g_clist = pij_coals(g_sample, seg=seg)  # partition the coalition based on g
                score_exp_items = []
                score_item = [0.0, 0.0]

                for cIdx, coal in enumerate(g_clist):
                    # new_list, cIdx = get_new_list(item, item_list)

                    if coal[0] < seg[0] or coal[0] >= seg[1]:  # out of the seg
                        continue

                    positions_dict = get_masks_sampleshapley(g_clist, cIdx, a_len, m_cnt)  # sample S
                    positions_dict = exclude_mask(positions_dict, coal, seg)

                    scores_c_s, scores_c_si = compute_scores_seperate(positions_dict, feature, a_len, bert_model.predict)

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

            _, _loss, summary_str, lr, g_score_ = sess.run([train_step, totloss, merged_summary_op, learning_rate, g_score], feed_dict=in_dict)

            summary_writer.add_summary(summary_str, epoch)

            print('epoch:', epoch, '-->loss:', _loss, '-->learning_rate:', lr, "\n")


if __name__ == "__main__":
    tf.app.run()
