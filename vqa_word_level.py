import tensorflow as tf
class vqa_word_level:
    def __init__(self,config):
        self.config = config

    def build(self,question_idxs,question_masks,embedding_matrix):

        self.word_embed = tf.nn.embedding_lookup(embedding_matrix, question_idxs)
        #self.word_embed = tf.transpose(self.word_embed,[0,2,1])
