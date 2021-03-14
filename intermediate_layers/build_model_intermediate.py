
import numpy as np
import tensorflow as tf
import os
import json
import random
import math
from preprocess import modeling


class TextModel(object):
    def __init__(self, bert_config_file, checkpoint, max_seq_length, num_labels, layer_id=11):
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

            all_layers = self.model.get_all_encoder_layers()  # [12，batch_size, seq_length, hidden_size]
            mid_layer = all_layers[layer_id]  # [batch_size, seg_length, hidden_size]
            mid_first_token_tensor = tf.squeeze(mid_layer[:, 0:1, :], axis=1)

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

        self.eval_acc = e_accuracy
        self.embedding = self.model.get_embedding_output()  # [batch_size, seq_length, embedding_size]
        self.embedding_table = self.model.get_embedding_table()  # [vocab_size, embedding_size]
        self.logits = logits
        self.probs = probs
        self.sess = None

        self.mid_output = mid_first_token_tensor

        # self.pool_wgt = self.graph.get_tensor_by_name("bert/pooler/dense/kernel:0")  # [768, 768]
        # self.pool_bias = self.graph.get_tensor_by_name("bert/pooler/dense/bias:0")  # [768,]
        # self.output_wgt = self.graph.get_tensor_by_name("output_weights:0")  # [2, 768]
        # self.output_bias = self.graph.get_tensor_by_name("output_bias:0")  # [2,]

    def start_session(self):
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(tf.global_variables_initializer())

    def close_session(self):
        self.sess.close()

    def get_embedding_table(self):
        with self.graph.as_default():
            table = self.sess.run([self.embedding_table])
        return table

    def predict(self, batch_dict, mid=True, bs=1000):
        input_ids = batch_dict['input_ids']
        input_mask = batch_dict['input_mask']
        segment_ids = batch_dict['segment_ids']
        label_ids = batch_dict['label_ids']

        tot = input_ids.shape[0]
        lots = []
        mid_outputs = []
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
                logits_, probs_, mid_ = self.sess.run([self.logits, self.probs, self.mid_output],
                                                feed_dict={self.input_ids: mini_input_ids, self.input_mask: mini_input_mask,
                                                self.segment_ids: mini_segment_ids, self.label_ids: mini_label_ids})
                lots.append(logits_)
                mid_outputs.append(mid_)

        if mid:
            return np.concatenate(mid_outputs, axis=0)
        else:
            return np.concatenate(lots, axis=0)



