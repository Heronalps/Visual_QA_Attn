"""
This is the model file which has instance of encoder an decoder
functions for training,testing and evaluation of model

"""

import tensorflow as tf

from vqa_encoder import *
from vqa_decoder import *
from vqa_preprocessing import *
import copy

class vqa_model:
    def __init__(self,config):
        print("Crearing the Model")
        self.config = config
        self.encoder = vqa_encoder(self.config)
        self.decoder = vqa_decoder(self.config)
        self.image_loader = ImageLoader('./ilsvrc_2012_mean.npy',self.config)
        self.global_step = 0

    def build(self):
        ## Build the encoder and decoder models
        ## Place holder fo the images and questions and we pass them to the encoder
        print("Building the Model .....")
        self.images = tf.placeholder(
            dtype=tf.float32,
            shape=[self.config.BATCH_SIZE] + self.config.IMAGE_SHAPE)
        self.questions =tf.placeholder(
            dtype=tf.int32,
            shape=[self.config.BATCH_SIZE] + [self.config.MAX_QUESTION_LENGTH])
        self.question_masks = tf.placeholder(
            dtype=tf.int32,
            shape=[self.config.BATCH_SIZE] + [self.config.MAX_QUESTION_LENGTH])

        ## Initialise the embedding matrix

        self.embedding_matrix = tf.get_variable(
            name='weights',
            shape=[self.config.VOCAB_SIZE, self.config.EMBEDDING_DIMENSION],
            initializer=self.encoder.cnn.nn.fc_kernel_initializer,
            regularizer=self.encoder.cnn.nn.fc_kernel_regularizer,
            trainable=True)

        ## pass the images, questions and embedding matrix to the encoder
        self.encoder.build(self.images,self.questions,self.question_masks, self.embedding_matrix)
        # ## pass the outputs of encoder to decoder model
        self.decoder.build(self.encoder.v_attend_word,self.encoder.q_attend_word,
                           self.encoder.v_attend_phrase,self.encoder.q_attend_phrase,
                           self.encoder.v_attend_sentence,self.encoder.q_attend_sentence)
        #
        # self.build_model()

    def build_model(self):
        ## Assign variables that needs to be passed to variables from encoder and decoder
        pass

    def train(self,sess,train_data):
        print("Training the model")

        epoch_count = self.config.EPOCH_COUNT


        for _ in tqdm(list(range(self.config.NUM_EPOCHS)), desc='epoch'):
            total_predictions_correct = 0
            for _ in tqdm(list(range(train_data.num_batches)), desc='batch'):
            #for _ in tqdm(list(range(self.config.NUM_BATCHES)), desc='batch'):
                batch = train_data.next_batch()
                image_files, question_idxs, question_masks, answer_idxs, answer_masks = batch
                images = self.image_loader.load_images(image_files)

                feed_dict = {self.images:images,
                             self.questions:question_idxs,
                             self.question_masks:question_masks,
                             self.decoder.answers:answer_idxs,
                             self.decoder.answer_masks:answer_masks}

                _, predictions_correct = sess.run([self.decoder.optimizer,self.decoder.predictions_correct],feed_dict=feed_dict)

                # _ = sess.run(self.encoder.v_attend_phrase,
                #                               feed_dict=feed_dict)
                #
                # predictions_correct = 0

                ## Global step count in order to store the model between batches
                self.global_step += 1
                total_predictions_correct += predictions_correct


                if(self.global_step % int(self.config.SAVE_PERIOD) == 0):
                    self.save("step_"+ str(self.global_step))
                    print("Total Predictions correct : {0} at time step {1}".format(total_predictions_correct,self.global_step))
                    f = open("results.txt", "a")
                    f.write("Total Predictions correct : {0} at time step {1} \n".format(total_predictions_correct,self.global_step))
                    f.close()

            epoch_count += 1
            print("Total Predictions correct : {0} at epoch {1} \n".format(total_predictions_correct,epoch_count))
            ## Save after all epochs
            self.save("epoch_"+str(epoch_count))
            f = open("results.txt", "a")
            f.write("Total Predictions correct : {0} at epoch {1} \n".format(total_predictions_correct, epoch_count))
            f.close()
            train_data.reset()



    def save(self,file_name):
        """ Save the model. """
        config = self.config
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.SAVE_DIR, file_name)
        print((" Saving the model to %s..." % (save_path+".npy")))
        np.save(save_path, data)
        print("Model saved.")

    def load(self, sess, model_file=None):
        """ Load the model. """
        config = self.config
        print("Loading the model from %s..." %model_file)
        data_dict = np.load(model_file).item()
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("%d tensors loaded." %count)





