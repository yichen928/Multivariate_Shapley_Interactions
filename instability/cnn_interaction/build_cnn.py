import numpy as np
import tensorflow as tf
import os
import json
import random
import math


class TextModel(object):
    def __init__(self, checkpoint, config):
        checkpoint_file = tf.train.latest_checkpoint(checkpoint)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph, config=config)
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(self.sess, checkpoint_file)
            self.input_x = self.graph.get_tensor_by_name('input_x:0')
            self.keep_prob = self.graph.get_tensor_by_name('dropout_keep_prob:0')
            self.pred_logits = self.graph.get_tensor_by_name('output/scores:0')
            self.predictions = self.graph.get_tensor_by_name('output/predictions:0')


    def start_session(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

    def close_session(self):
        self.sess.close()

    def predict(self, batch_dict):
        with self.graph.as_default():
            logits_ = self.sess.run(self.pred_logits, {self.input_x: batch_dict['input_x'], self.keep_prob: 1.0})

        return logits_



