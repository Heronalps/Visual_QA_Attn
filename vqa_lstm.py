"""
VQA LSTM part
"""
import tensorflow as tf
import numpy as np

class vqa_lstm(object):
    def __init__(self, config):
        self.n_steps = config.LSTM_STEPS
        self.input_size = config.LSTM_INPUT_SIZE
        self.output_size = config.LSTM_OUTPUT_SIZE
        self.cell_size = config.LSTM_CELL_SIZE
        self.batch_size = config.LSTM_BATCH_SIZE
        self.drop_rate = config.LSTM_DROP_RATE
        self.lstm_layer = 1
        self.dim = self.lstm_layer * 512

    # def build(self, question_idxs, questions_mask, embedding_matrix):
    #
    #     print(" Shape of Question Tensor {}".format(question_idxs.get_shape()))
    #     word_embed = tf.nn.embedding_lookup(embedding_matrix, question_idxs)
    #     print(" Shape of Word Tensor {}".format(word_embed.get_shape()))
    #     lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(self.cell_size, state_is_tuple = True)
    #     lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(self.cell_size, state_is_tuple = True)
    #     multi_lstm_cell = tf.contrib.rnn.MultiRNNCell(cells = [lstm_cell_1, lstm_cell_2], state_is_tuple = True)
    #
    #     #with tf.name_scope('initial_state'):
    #     #    self.cell_init_state = lstm_cell_1.zero_state(self.batch_size, dtype=tf.float32)
    #
    #     self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
    #         multi_lstm_cell, word_embed, time_major=False, dtype = tf.float32)
    #
    #     #self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
    #     #    lstm_cell_1, word_embed, initial_state=self.cell_init_state, time_major=False)
    #     # print("LSTM final state shape :{}".format(self.cell_final_state.get_shape()))
    #     # print("LSTM final state size: 0 shape {0}, 1 shape {1}".format(self.cell_final_state[0].get_shape(), self.cell_final_state[1].get_shape()))
    #     self.lstm_features = tf.concat([self.cell_final_state[0][0], self.cell_final_state[0][1],self.cell_final_state[1][0], self.cell_final_state[1][1]], 1)
    #     print("LSTM Concat Feature size {}".format(self.lstm_features.get_shape()))

    ## Return self.lstm_features

    def build(self, sentences):
        """ Build the RNN. """
        print("Building the RNN...")

        # Setup the LSTM
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(
            self.cell_size,
            state_is_tuple = True)
        lstm_cell_1 = tf.contrib.rnn.DropoutWrapper(
            lstm_cell_1,
            input_keep_prob=1.0 - self.drop_rate,
            output_keep_prob=1.0 - self.drop_rate,
            state_keep_prob=1.0 - self.drop_rate)

        # lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(
        #     self.cell_size,
        #     state_is_tuple=True)


        # Initialize the LSTM using the mean context
        with tf.variable_scope("initialize"):
            initial_memory = tf.zeros([self.batch_size, lstm_cell_1.state_size[0]])
            initial_output = tf.zeros([self.batch_size, lstm_cell_1.state_size[1]])

        ## Initial memory and output are given zeros
        last_memory = initial_memory
        last_output = initial_output
        last_state = last_memory, last_output

        lstm_feature_arr = []

        # Generate the words one by one
        for idx in range(self.n_steps):

            # Apply the LSTM
            with tf.variable_scope("lstm",reuse = tf.AUTO_REUSE):
                current_input = sentences[:, idx]
                output, state = lstm_cell_1(current_input, last_state)
                memory, hidden_state = state

            last_state = state

            lstm_feature_arr.append(hidden_state)


        self.lstm_features = tf.stack(lstm_feature_arr, axis=1)


        print("RNN built.")
