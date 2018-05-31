import tensorflow as tf

class vqa_phrase_level:
    def __init__(self,config):
        self.config = config

    def build(self,word_embedding):
        config = self.config
        ## Unigram
        kernel_unigram = tf.get_variable(initializer=tf.truncated_normal([1, config.EMBEDDING_DIMENSION, config.EMBEDDING_DIMENSION],
                                                                         dtype=tf.float32,stddev=1e-1), name='kernel_unigram',
                                        trainable=True)
        unigram_conv = tf.nn.conv1d(word_embedding, kernel_unigram, stride=1, padding='SAME')

        unigram_biases = tf.get_variable(initializer=tf.constant(0.0, shape=[config.EMBEDDING_DIMENSION], dtype=tf.float32),
                                 trainable=True, name='bias_unigram')

        unigram_embedding = tf.nn.bias_add(unigram_conv, unigram_biases)
        print("Unigram shape {}".format(unigram_embedding.get_shape()))

        ## Bigram
        kernel_bigram = tf.get_variable(initializer=tf.truncated_normal([2, config.EMBEDDING_DIMENSION, config.EMBEDDING_DIMENSION],
                                                                        dtype=tf.float32, stddev=1e-1), name='kernel_bigram',
                                        trainable=True)
        bigram_conv = tf.nn.conv1d(word_embedding, kernel_bigram, stride=1, padding='SAME')

        bigram_biases = tf.get_variable(initializer=tf.constant(0.0, shape=[config.EMBEDDING_DIMENSION], dtype=tf.float32),
                                trainable=True, name='bias_bigram')

        bigram_embedding = tf.nn.bias_add(bigram_conv, bigram_biases)
        print("Bigram shape {}".format(bigram_embedding.get_shape()))

        ## Trigram

        kernel_trigram = tf.get_variable(initializer=tf.truncated_normal([3, config.EMBEDDING_DIMENSION, config.EMBEDDING_DIMENSION],
                                                                        dtype=tf.float32, stddev=1e-1), name='kernel_trigram',
                                        trainable=True)
        trigram_conv = tf.nn.conv1d(word_embedding, kernel_trigram, stride=1, padding='SAME')

        trigram_biases = tf.get_variable(initializer=tf.constant(0.0, shape=[config.EMBEDDING_DIMENSION], dtype=tf.float32),
                                    trainable=True, name='bias_trigram')

        trigram_embedding = tf.nn.bias_add(trigram_conv, trigram_biases)
        print("Trigram shape {}".format(trigram_embedding.get_shape()))

        ## Stacking the trigram
        stacked_grams = tf.stack([unigram_embedding,bigram_embedding,trigram_embedding],-1) ## [?,MAX_QUESTIO_LENGTH,EMBEDDING_DIMENSION,3]

        print("Concat shape {}".format(stacked_grams[0][1][:,0].get_shape()))

        # self.max_norm(stacked_grams[0][1])
        ## Getting the maximum of tri-gram

        self.phrase_level_features = tf.scan(lambda b, x: self.max_norm_complete(x), stacked_grams,
                        initializer=tf.get_variable(
                            shape=[config.MAX_QUESTION_LENGTH, config.EMBEDDING_DIMENSION],
                            name="phrase_level_features"))


    def max_norm_complete(self,sentence_all_grams):
        print("ALL grams shape {}".format(sentence_all_grams.get_shape()))
        return tf.scan(lambda c,y:self.max_norm(y),sentence_all_grams,
                       initializer=tf.get_variable(
                           shape=[self.config.EMBEDDING_DIMENSION],
                           name="phrase_all_grams")
                       )


    def max_norm(self,word_all_grams):
        print("Word grams shape {}".format(word_all_grams.get_shape()))
        unigram_vector = word_all_grams[:,0]
        bigram_vector  = word_all_grams[:,1]
        trigram_vector = word_all_grams[:,2]

        unigram_norm = tf.norm(unigram_vector)
        bigram_norm  = tf.norm(bigram_vector)
        trigram_norm = tf.norm(trigram_vector)

        if (unigram_norm == tf.maximum(unigram_norm,bigram_norm)):
            temp_max_norm = unigram_norm
            temp_max_vector = unigram_vector
        else:
            temp_max_norm = bigram_norm
            temp_max_vector = bigram_vector

        if(temp_max_norm == tf.maximum(temp_max_norm,trigram_norm)):
            return temp_max_vector
        else:
            return trigram_vector



