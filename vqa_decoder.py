import tensorflow as tf
import numpy as np

class vqa_decoder:
    def __init__(self,config):
        self.config =  config
        print("decoder model created")


    def build(self,attend_image_word,attend_question_word,attend_image_phrase,
              attend_question_phrase,attend_image_sentence,attend_question_sentence):

        print("Building Decoder")
        config = self.config

        self.attend_image_word = attend_image_word
        self.attend_question_word = attend_question_word
        self.attend_image_phrase = attend_image_phrase
        self.attend_question_phrase = attend_question_phrase
        self.attend_image_sentence = attend_image_sentence
        self.attend_question_sentence = attend_question_sentence

        # Setup the placeholders
        if config.PHASE == 'train':
            # contexts = self.conv_feats
            self.answers = tf.placeholder(
                dtype=tf.int32,
                shape=[config.BATCH_SIZE, config.MAX_ANSWER_LENGTH])
            self.answer_masks = tf.placeholder(
                dtype=tf.int32,
                shape=[config.BATCH_SIZE, config.MAX_ANSWER_LENGTH])


        ## Create weights variable for each attention
        with tf.variable_scope("Attention_Weights_Decoder",reuse=tf.AUTO_REUSE) as scope :
            attend_weight_word = tf.get_variable(initializer=tf.truncated_normal([config.EMBEDDING_DIMENSION,config.EMBEDDING_DIMENSION],dtype=tf.float32,
                                                   stddev=1e-1), name='attend_weight_word',trainable=True)
            attend_weight_phrase = tf.get_variable(initializer=tf.truncated_normal([2*config.EMBEDDING_DIMENSION,config.EMBEDDING_DIMENSION],dtype=tf.float32,
                                                   stddev=1e-1),name='attend_weight_phrase', trainable=True)
            attend_weight_sentence = tf.get_variable(initializer=tf.truncated_normal([2*config.EMBEDDING_DIMENSION,config.EMBEDDING_DIMENSION],dtype=tf.float32,
                                                   stddev=1e-1),name='attend_weight_sentence', trainable=True)

            attend_bias_word = tf.get_variable(initializer=tf.truncated_normal([config.EMBEDDING_DIMENSION],dtype=tf.float32,
                                                stddev=1e-1), name='attend_bias_word', trainable=True)
            attend_bias_phrase = tf.get_variable(initializer=tf.truncated_normal([config.EMBEDDING_DIMENSION],dtype=tf.float32,
                                                stddev=1e-1), name='attend_bias_phrase', trainable=True)
            attend_bias_sentence = tf.get_variable(initializer=tf.truncated_normal([ config.EMBEDDING_DIMENSION],dtype=tf.float32,
                                                stddev=1e-1), name='attend_bias_sentence', trainable=True)



        attend_vector_word = tf.tanh(tf.matmul(self.attend_image_word+self.attend_question_word,attend_weight_word)+attend_bias_word)
        print("Attend Vector Word {}".format(attend_vector_word.get_shape()))

        temp_attend_phrase = self.attend_image_phrase+self.attend_question_phrase
        print("Temp Vector Phase {}".format(temp_attend_phrase.get_shape()))

        attend_vector_phrase = tf.tanh(tf.matmul(tf.concat([attend_vector_word,temp_attend_phrase],axis = 1),attend_weight_phrase) + attend_bias_phrase)
        print("Attend Vector Phrase {}".format(attend_vector_phrase.get_shape()))

        temp_attend_sentence = self.attend_image_sentence + self.attend_question_sentence
        attend_vector_sentence = tf.tanh(tf.matmul(tf.concat([attend_vector_phrase,temp_attend_sentence],axis = 1),attend_weight_sentence)+attend_bias_sentence)
        print("Attend Vector Sentence {}".format(attend_vector_sentence.get_shape()))

        ## Build a Fully Connected Layer
        with tf.variable_scope('fc_decoder', reuse=tf.AUTO_REUSE) as scope:
            fcw = tf.get_variable(initializer=tf.truncated_normal([self.config.EMBEDDING_DIMENSION, self.config.OUTPUT_SIZE],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='fc_W',trainable=True)
            fcb = tf.get_variable(initializer=tf.constant(1.0, shape=[self.config.OUTPUT_SIZE], dtype=tf.float32),
                               trainable=True, name='fc_b')
            fcl = tf.nn.bias_add(tf.matmul(attend_vector_sentence, fcw), fcb)
            self.logits = tf.nn.relu(fcl)

        if config.PHASE == 'train':
            # Compute the loss for this step, if necessary
            cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.answers[:,0],  ##[:,0] because answers is array of arrays
                logits=self.logits)

            self.optimizer = tf.train.AdamOptimizer(config.INITIAL_LEARNING_RATE).minimize(cross_entropy_loss)


        self.predictions = tf.argmax(self.logits, 1,output_type=tf.int32)
        self.softmax_logits = tf.nn.softmax(self.logits)
        if config.PHASE == 'train':
            ## Number of correct predictions in each run
            self.predictions_correct = tf.reduce_sum(tf.cast(tf.equal(self.predictions, self.answers[:, 0]),tf.float32))


        print(" Decoder model built")


