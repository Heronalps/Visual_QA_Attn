""""
This encoder model consists of instances
1. CNN 2. Word Level 3. Phrase Level 4. Sentence Level(Referred to as Question level in paper)
This layer output the hidden state vector for both image and question at all levels to the decoder.
Co-attention part is also taken care here

"""
import tensorflow as tf
from vqa_cnn import *
from vqa_word_level import *
from vqa_phrase_level import *
from vqa_lstm import *

class vqa_encoder:
    def __init__(self,config):
        self.config = config
        self.cnn = vqa_cnn(config)
        self.word_level = vqa_word_level(config)
        self.phrase_level = vqa_phrase_level(config)
        self.sentence_level = vqa_lstm(config)

    # def build(self, images, questions, question_masks, embedding_matrix):
    def build(self, image_features, questions, question_masks, embedding_matrix):
        # ## Build the CNN model
        if self.config.PHASE == "test":
            images = image_features
            self.cnn.build(images)

        self.image_features = image_features
        ## Build the word level
        self.word_level.build(questions, question_masks, embedding_matrix)
        ## Build the Phrase level

        self.phrase_level.build(self.word_level.word_embed)
        # ## Build the sentence level LSTM Model
        self.sentence_level.build(self.phrase_level.phrase_level_features)
        # ## Combine the model

        self.build_encoder()


    def build_encoder(self):

        config = self.config
        # config.IMAGE_FEATURES = self.cnn.num_ctx

        ## d = 512, N = 14, T = 25, k = 25
        ## Building Word Level features

        if self.config.PHASE == "test":
            print("CNN feature size {}".format(self.cnn.conv_feats.get_shape())) ## [BATCH_SIZE,14,512]
            self.V = tf.transpose(self.cnn.conv_feats, [0, 2, 1]) ##[BATCH_SIZE,512,14] (V) [?,d,N]
            print("V_word shape : {}".format(self.V.get_shape()))
        else:
            print("CNN feature size {}".format(self.image_features.get_shape()))  ## [BATCH_SIZE,14,512]
            self.V = tf.transpose(self.image_features, [0, 2, 1])  ##[BATCH_SIZE,512,14] (V) [?,d,N]
            print("V_word shape : {}".format(self.V.get_shape()))


        print("Word Level feature size {}".format(self.word_level.word_embed.get_shape())) ## [BATCH_SIZE,25,512]

        self.Q_word = tf.transpose(self.word_level.word_embed,[0,2,1]) ## [BATCH_SIZE,512,25] (Q) [?,d,T]

        ## Call the parallel co-attention model on word , phrase and sentence level
        self.v_attend_word, self.q_attend_word = self.parallel_co_attention(self.V, self.Q_word, "word")

        print("Phrase Level feature size {}".format(self.phrase_level.phrase_level_features.get_shape()))
        self.Q_phrase = tf.transpose(self.phrase_level.phrase_level_features,[0,2,1])

        self.v_attend_phrase, self.q_attend_phrase = self.parallel_co_attention(self.V, self.Q_phrase, "phrase")

        print("Sentence Level feature size {}".format(self.sentence_level.lstm_features.get_shape()))
        self.Q_sentence = tf.transpose(self.sentence_level.lstm_features,[0,2,1])

        self.v_attend_sentence, self.q_attend_sentence = self.parallel_co_attention(self.V,self.Q_sentence,"sentence")



    def parallel_co_attention(self,V,Q,name_scope="word"):
        config = self.config

        with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE) as scope:
            W_b = tf.get_variable(
                initializer=tf.truncated_normal([config.EMBEDDING_DIMENSION, config.EMBEDDING_DIMENSION],
                                                dtype=tf.float32, stddev=1e-1),
                name='W_b', trainable=True)  ##[d,d] W_b

            ## Multiplying weight matrix with word level question features
            C_intermediate = tf.scan(lambda a, x: tf.matmul(x, W_b),
                                                            tf.transpose(Q, [0, 2, 1]))  ## [?,T,d]

            print("Affinity_size_intermediate {0}: {1}".format(name_scope, C_intermediate.get_shape()))

            ## Multiplyint intermediate affinity with word level image features
            C = tf.tanh(tf.matmul(C_intermediate, V))  ## (C) [?,T,N]

            print("Affinity_size {0} : {1}".format(name_scope, C.get_shape()))

            W_v = tf.get_variable(
                initializer=tf.truncated_normal([config.INTERMEDIATE_DIMENSION, config.EMBEDDING_DIMENSION],
                                                dtype=tf.float32, stddev=1e-1),
                name='W_v', trainable=True)  ## W_v [k,d]

            W_q = tf.get_variable(
                initializer=tf.truncated_normal([config.INTERMEDIATE_DIMENSION, config.EMBEDDING_DIMENSION],
                                                dtype=tf.float32, stddev=1e-1),
                name='W_q', trainable=True)  ## W_q [k,d]

            ## Calcluate the H-values
            ## W_v * V [k,d] *[?,d,N] = [?,k,N]
            W_v_V = tf.scan(lambda b, x: tf.matmul(W_v, x), V,
                                      initializer=tf.get_variable(
                                          shape=[config.INTERMEDIATE_DIMENSION, config.IMAGE_FEATURES],
                                          name="W_v_V"))

            ## W_q * Q [k,d] * [?,d,T] = [?,k,T]
            W_q_Q = tf.scan(lambda b, x: tf.matmul(W_q, x), Q,
                                      initializer=tf.get_variable(
                                          shape=[config.INTERMEDIATE_DIMENSION, config.MAX_QUESTION_LENGTH],
                                          name="W_q_Q"))

            H_v = tf.tanh( tf.add(W_v_V, tf.matmul(W_q_Q, C)))  ## [?,k,N]
            H_q = tf.tanh(tf.add(W_q_Q, tf.matmul(W_v_V, tf.transpose(C,[0, 2, 1]))))  ## [?,k,T]

            print("H_v shape {0}: {1}".format(name_scope, H_v.get_shape()))
            print("H_q shape {0}: {1}".format(name_scope, H_q.get_shape()))

            w_h_v = tf.get_variable(initializer=tf.truncated_normal([config.INTERMEDIATE_DIMENSION, 1],
                                                                              dtype=tf.float32, stddev=1e-1),
                                              name='w_h_v', trainable=True)

            w_h_q = tf.get_variable(initializer=tf.truncated_normal([config.INTERMEDIATE_DIMENSION, 1],
                                                                              dtype=tf.float32, stddev=1e-1),
                                              name='w_h_q', trainable=True)

            ## Attention weights
            a_v = tf.nn.softmax(tf.scan(lambda a, x: tf.matmul(tf.transpose(w_h_v), x), H_v,
                                    initializer=tf.get_variable(shape=[1, config.IMAGE_FEATURES],
                                                                name="a_v")) ) ## [?,1,N]

            a_q = tf.nn.softmax(tf.scan(lambda a, x: tf.matmul(tf.transpose(w_h_q), x), H_q,
                                    initializer=tf.get_variable(shape=[1, config.MAX_QUESTION_LENGTH],
                                                                name="a_q")))  ## [?,1,T]

            print("a_v shape {0} : {1}".format(name_scope,a_v.get_shape()))
            print("a_q shape {0}: {1}".format(name_scope,a_q.get_shape()))

            ## Aggregate average with attention vectors

            V_attend = tf.matmul(a_v, tf.transpose(V, [0, 2, 1]))
            Q_attend = tf.matmul(a_q, tf.transpose(Q, [0, 2, 1]))

            ## Reshaped because it gives us a dimension [?,1,d]
            V_attend = tf.reshape(V_attend, [config.BATCH_SIZE, config.EMBEDDING_DIMENSION])
            Q_attend = tf.reshape(Q_attend, [config.BATCH_SIZE, config.EMBEDDING_DIMENSION])

            print("V_attend shape {0}: {1}".format(name_scope, V_attend.get_shape()))
            print("Q_attend shape {0}: {1}".format(name_scope, Q_attend.get_shape()))

            return V_attend, Q_attend




