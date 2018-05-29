""""
This encoder model consists of instances of cnn and LSTM.
Here we combine output of each model to a separate fully connected layer to get the outputs of 1024.
In Hierachical co-attention model, this encoder model consists of code where co-attention is possible

"""
import tensorflow as tf
from vqa_cnn import *
from vqa_lstm import *

class vqa_encoder:
    def __init__(self,config):
        self.config = config
        self.cnn = vqa_cnn(self.config)
        ## LSTM code here
        self.lstm = vqa_lstm(self.config)

    def build(self,images,questions,question_masks,embedding_matrix):
        ## Build the CNN model
        self.cnn.build(images)
        ## Build the sentence level LSTM Model
        self.lstm.build(questions,question_masks,embedding_matrix)
        ## Combine the model
        self.build_encoder()


    def build_encoder(self):

        ## Get the features from rnn model

        ## Build a fully connected layer for CNN and LSTM to get 1024 features for each
        ## Get the features from CNN model
        print("CNN feature size {}".format(self.cnn.conv_feats.get_shape()))
        with tf.variable_scope('fc_cnn_model', reuse=tf.AUTO_REUSE) as scope:
            fc_cnn_model_w = tf.get_variable(
                initializer=tf.truncated_normal([self.cnn.dim_ctx, self.config.POINT_WISE_FEATURES],
                                                dtype=tf.float32,
                                                stddev=1e-1), name='W', trainable=True)
            fc_cnn_model_b = tf.get_variable(initializer=tf.constant(1.0, shape=[self.config.POINT_WISE_FEATURES], dtype=tf.float32),
                                  trainable=True, name='B')
            self.cnn_features = tf.nn.bias_add(tf.matmul(self.cnn.conv_feats, fc_cnn_model_w), fc_cnn_model_b)

        ## Get the features from LSTM model
        with tf.variable_scope('fc_lstm_model', reuse=tf.AUTO_REUSE) as scope:
            fc_lstm_model_w = tf.get_variable(
                initializer=tf.truncated_normal([self.lstm.dim, self.config.POINT_WISE_FEATURES],
                                                dtype=tf.float32,
                                                stddev=1e-1), name='W', trainable=True)
            fc_lstm_model_b = tf.get_variable(initializer=tf.constant(1.0, shape=[self.config.POINT_WISE_FEATURES], dtype=tf.float32),
                                  trainable=True, name='B')
            self.lstm_features = tf.nn.bias_add(tf.matmul(self.lstm.lstm_features, fc_lstm_model_w), fc_lstm_model_b)




