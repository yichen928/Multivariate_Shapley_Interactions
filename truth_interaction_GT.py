import numpy as np
import tensorflow as tf
import os
import math
import itertools
import copy
from build_model import TextModel
from shapley import *
from preprocess import tokenization, extract

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
    players = [i+seg[0] for i in range(pij.shape[0]+1)]
    start = 0
    end = 1
    for i in range(0, seg[0]):
        coalitions.append([i])
    while end < len(players):
        if pij[end-1] > threshold:
            end += 1
        else:
            coalitions.append(players[start:end])
            start = end
            end = start+1
    coalitions.append(players[start:end])
    for i in range(seg[1],seg[2]):
        coalitions.append([i])
    return coalitions


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


def sharpley_sample(seg, coal):
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

    weight_list = np.array(weight_list)
    position_dict = {
        (0, 0): masks,
        (0, 1): masks_i
    }
    return position_dict, weight_list


def calculate_shap_sum(clist, seg, feature, model, print_info=False):
    print(clist, seg)
    if print_info:
        print(clist)
    a_len = seg[-1]
    cIdx = seg[0]
    shap_sum = 0
    while (isinstance(clist[cIdx], list) and clist[cIdx][0]>=seg[0] and clist[cIdx][0]<seg[1])or ((isinstance(clist[cIdx], int) and clist[cIdx] < seg[1])):
        if print_info:
            print(clist, cIdx)

        positions_dict, weight_list = sharpley_sample(seg, clist[cIdx])

        scores_c = compute_scores(positions_dict, feature, a_len, model.predict)
        scores_c = scores_c * weight_list
        cIdx += 1
        shap_sum += np.sum(scores_c)
        if cIdx >= len(clist):
            break
    if print_info:
        print(clist, shap_sum)
    return shap_sum


def get_min_max_shap(seg, feature, model):
    clists = get_all_coalitions(seg)

    max_clist = []
    min_clist = []
    max_shap = -999999
    min_shap = 999999
    for clist in clists:
        shap_sum = calculate_shap_sum(clist, seg, feature, model, print_info = True)

        if shap_sum > max_shap:
            max_shap = shap_sum
            max_clist = clist[:]
        if shap_sum < min_shap:
            min_shap = shap_sum
            min_clist = clist[:]
    return min_shap, max_shap, min_clist, max_clist


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
    tf.gfile.MakeDirs(FLAGS.output_dir)

    # ------------------- preprocess dataset -------------------- #
    label_list = processor.get_labels()
    num_labels = len(label_list)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    max_seq_length = FLAGS.max_seq_length

    # prepare valid dataset
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
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

    sentence = eval_examples[FLAGS.sentence_idx] # the input sentence
    tokens_a = tokenizer.tokenize(sentence.text_a)

    a_len = len(tokens_a)
    # input feature of the sentence to BERT model
    feature = extract.convert_single_example(0, sentence, label_list, max_seq_length, tokenizer)

    seg = (FLAGS.seg_start, FLAGS.seg_end, a_len)  #左闭右开

    print(tokens_a)
    min_shap, max_shap, min_clist, max_clist = get_min_max_shap(seg, feature, bert_model)

    print(sentence.text_a)
    print(tokens_a)
    print("MAX:", max_shap, max_clist)
    print("MIN:", min_shap, min_clist)
    print("seg:","(%d, %d)"%(seg[0],seg[1]-1)) # 左右都包含



if __name__ == "__main__":
    flags = tf.flags
    FLAGS = flags.FLAGS
    bert_dir = "./models/uncased_L-12_H-768_A-12"
    data_dir = "./GLUE_data/SST-2"

    # parse flags and run main()
    flags.DEFINE_string("task_name", "sst-2", "The name of the task to train. [sst-2, cola]")

    if FLAGS.task_name == "sst-2":
        data_dir = "./GLUE_data/SST-2"
    else:
        assert FLAGS.task_name == "cola"
        data_dir = "./GLUE_data/CoLA"

    flags.DEFINE_string(
        "bert_config_file", bert_dir + "/bert_config.json",
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

    flags.DEFINE_string(
        "data_dir", data_dir,
        "The input data dir. Should contain the .tsv files (or other data files) "
        "for the task.")

    flags.DEFINE_string(
        "init_checkpoint", None,
        "Initial checkpoint (usually from a pre-trained BERT model).")

    flags.DEFINE_integer(
        "max_seq_length", 128,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded.")

    flags.DEFINE_string("vocab_file", bert_dir + "/vocab.txt",
                        "The vocabulary file that the BERT model was trained on.")

    flags.DEFINE_string(
        "output_dir", "new_output",
        "The output directory where the dataset tfrecord will be written.")

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
        "lr", 1e-1, "The learning rate."
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

    if FLAGS.init_checkpoint is None:
        if FLAGS.task_name == "sst-2":
            FLAGS.init_checkpoint = data_dir + "/model/model.ckpt-6313"
        else:
            assert FLAGS.task_name == "cola"
            FLAGS.init_checkpoint = data_dir + "/model/model.ckpt-801"

    tf.app.run()

