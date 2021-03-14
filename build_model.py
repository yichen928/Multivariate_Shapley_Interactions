# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
from preprocess import modeling


class TextModel(object):
    def __init__(self, bert_config_file, checkpoint, max_seq_length, num_labels):
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        self.graph = tf.Graph()
        with self.graph.as_default():
            # tf.set_random_seed(-1)
            self.max_seq_length = max_seq_length
            self.num_labels = num_labels
            # [batch_size, max_seq_length]
            self.input_ids = tf.placeholder(tf.int32, shape=(None, max_seq_length), name='input_ids')
            self.input_mask = tf.placeholder(tf.int32, shape=(None, max_seq_length), name='input_mask')
            self.segment_ids = tf.placeholder(tf.int32, shape=(None, max_seq_length), name='segment_ids')
            self.label_ids = tf.placeholder(tf.int32, shape=(None,), name='label_ids')

            print(os.path.dirname(modeling.__file__))  # bugfix
            self.model = modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=self.input_ids,
                input_mask=self.input_mask,
                token_type_ids=self.segment_ids,
                use_one_hot_embeddings=False
            )
            output_layer = self.model.get_pooled_output()  # [bs, hidden_size]
            hidden_size = output_layer.shape[-1].value  # 768

            output_weights = tf.get_variable(
                "output_weights", [self.num_labels, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "output_bias", [self.num_labels], initializer=tf.zeros_initializer())

            with tf.variable_scope("loss"):
                is_training = False
                if is_training:
                    # I.e., 0.1 dropout
                    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

                logits = tf.matmul(output_layer, output_weights, transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)   # [bs, 2]
                probs = tf.nn.softmax(logits, axis=-1)
                log_probs = tf.nn.log_softmax(logits, axis=-1)

                one_hot_labels = tf.one_hot(self.label_ids, depth=self.num_labels, dtype=tf.float32)

                per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
                loss = tf.reduce_mean(per_example_loss)

                # eval metrics
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                e = tf.equal(predictions, self.label_ids)
                e_accuracy = tf.reduce_mean(tf.cast(e, tf.float32))

            tvars = tf.trainable_variables()
            (assignment_map,
             initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars, checkpoint)
            # values are not load immediately but when the global initializer is run
            tf.train.init_from_checkpoint(checkpoint, assignment_map)
            tf.logging.info("**** Pre-trained Trainable Variables ****")
            # for var in tvars:
            #     init_string = ""
            #     if var.name in initialized_variable_names:
            #         init_string = ", *INIT_FROM_CKPT*"
            #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
            #                     init_string)

        self.all_layers = self.model.get_all_encoder_layers()  # [12，batch_size, seq_length, hidden_size]
        self.embedding = self.model.get_embedding_output()  # [batch_size, seq_length, embedding_size]
        self.logits = logits
        self.probs = probs
        self.sess = None

    def start_session(self):
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(tf.global_variables_initializer())

    def close_session(self):
        self.sess.close()

    def predict(self, batch_dict, bs=1000):
        input_ids = batch_dict['input_ids']
        input_mask = batch_dict['input_mask']
        segment_ids = batch_dict['segment_ids']
        label_ids = batch_dict['label_ids']

        tot = input_ids.shape[0]
        lots = []
        with self.graph.as_default():
            for i in range(0, tot, bs):
                st = i
                if st+bs < tot:
                    ed = st+bs
                else:
                    ed = tot

                mini_input_ids = input_ids[st:ed]
                mini_input_mask = input_mask[st:ed]
                mini_segment_ids = segment_ids[st:ed]
                mini_label_ids = label_ids[st:ed]
                logits_, probs_ = self.sess.run([self.logits, self.probs],
                                                feed_dict={self.input_ids: mini_input_ids, self.input_mask: mini_input_mask,
                                                self.segment_ids: mini_segment_ids, self.label_ids: mini_label_ids})
                lots.append(logits_)

        return np.concatenate(lots, axis=0)


class CoalitionModel(object):
    def __init__(self, element_num: int, neighbour_range: int = 3):
        """
            element_num: number of words in the sentence or pixels in the subarea

        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.config = config
        self.graph = tf.Graph()

        # add parameter if needed
        self.element_num = element_num
        self.lamada = (neighbour_range - 1) // 2
        self.build_model(self.graph)

    def build_model(self, graph):
        with graph.as_default():
            # self.weight_decay = 0
            # placeholder inputs
            self.lr = tf.placeholder("float", shape=[])
            self.mask = tf.placeholder("bool", shape=[self.element_num])  # mask value
            self.coalition = tf.placeholder("bool",
                                            shape=[self.element_num])  # True: in coalition, False: not in coalition
            self.coalition_neighbour = tf.placeholder("bool", shape=[
                self.element_num])  # True: in coalition or its neighbour, False: not in coalition or its neighbour

            # learnable params
            self.p_weights = tf.Variable(tf.random.uniform(self.element_num - 1))

            # optimizer
            self.optimizer = tf.train.AdamOptimizer(self.lr)

            self.prob_c = self.get_coalition_prob(self.p_weights, self.mask, self.coalition_neighbour)

            # TODO ----------------------
            loss = 0
            self.train_step = self.optimizer.minimize(loss)

    def get_coalition_prob(self, p, mask, coalition_neighbour):
        mask_equ = tf.abs(mask[:, 1:] - mask[:, :-1])  # judge whether the mask equal to its upstream neighbour
        p_up = tf.Variable(tf.ones(
            [self.element_num - 1])) - mask_equ * p  # from the second element because the first one has no upstream
        Prob_c = tf.reduce_prod(p_up * coalition_neighbour)
        return Prob_c

    def train(self, save_path='./saves', load_vars_from_ckpt=False, lr=1e-5, epochs=100):
        with self.graph.as_default():
            if self.sess is None:
                self.sess = tf.Session(config=self.config, graph=self.graph)
            if not load_vars_from_ckpt:
                self.sess.run(tf.global_variables_initializer())
            else:
                self.load_variables(save_path + '/epoch10_ckpt')

            lr_decay = 0.01
            for epoch in range(epochs):
                print('--------Epoch %d--------' % epoch)
                curlr = lr * pow(lr_decay, epoch * 1.0 / (epochs - 1))
                train_steps = 1
                epoch_loss = []

                batch_loss = []
                for b in range(train_steps):
                    in_dict = {
                        self.lr: curlr,
                        self.mask: tf.random_normal([self.element_num]),
                        self.coalition: tf.random_normal([self.element_num]),
                        self.coalition_neighbour: tf.random_normal([self.element_num]),
                    }
                    _train_step, loss = self.sess.run([self.train_step, self.loss], feed_dict=in_dict)
                    batch_loss.append(loss)

                train_loss = np.mean(np.array(batch_loss))
                print('train loss:', train_loss)
                epoch_loss.append(train_loss)

                if (epoch + 1) % 10 == 0:
                    savepath = save_path + "/epoch{}_ckpt".format(epoch + 1)
                    np.savetxt(savepath + '_train_loss.txt', np.array(epoch_loss))
                    self.save_variables(savepath)

    def load_variables(self, path='./coal_ckpt'):
        saver = tf.train.Saver()
        print('loading variables...')
        saver.restore(self.sess, path)

    def save_variables(self, path='./coal_ckpt'):
        saver = tf.train.Saver()
        print('saving variables...')
        saver.save(self.sess, path)

    def close_session(self):
        self.sess.close()



