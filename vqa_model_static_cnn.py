"""
This is the model file which has instance of encoder an decoder
functions for training,testing and evaluation of model

"""

#import tensorflow as tf
import json

from vqa_encoder import *
from vqa_decoder import *
from vqa_preprocessing import *
import copy

class vqa_model_static_cnn:
    def __init__(self,config):
        print("Creating the CNN Static Model")
        self.config = config
        self.cnn = vqa_cnn(self.config)
        self.image_loader = ImageLoader('./ilsvrc_2012_mean.npy',self.config)
        self.global_step = 0

    def build(self):
        ## Build the encoder and decoder models
        ## Place holder fo the images and questions and we pass them to the encoder
        print("Building the Model .....")
        self.images = tf.placeholder(
            dtype=tf.float32,
            shape=[self.config.BATCH_SIZE] + self.config.IMAGE_SHAPE)

        self.cnn.build(self.images)

        # self.build_model()

    def build_model(self):
        ## Assign variables that needs to be passed to variables from encoder and decoder
        pass

    def train(self, sess, train_data, fc_file_name, conv_file_name):
        print("Training the CNN model")

        epoch_count = self.config.EPOCH_COUNT

        # Convolution feature map dictionary for VQA model with attention
        self.conv_dict = {}

        # Fully-connected feature map dictionary for VQA model without attention
        self.fc_dict = {}

        for _ in tqdm(list(range(train_data.num_batches)), desc='batch'):
            # for _ in tqdm(list(range(self.config.NUM_BATCHES)), desc='batch'):
            batch = train_data.next_batch()
            image_files, image_idxs = batch
            images = self.image_loader.load_images(image_files)

            feed_dict = {self.images: images}

            self.conv_feats, self.fc2 = sess.run([self.cnn.conv_feats, self.cnn.fc2], feed_dict=feed_dict)

            ## Save conv5_3 and fc2 into two dictionaries
            i = 0
            for idx in image_idxs:
                self.conv_dict[str(idx)] = self.conv_feats[i]

                self.fc_dict[str(idx)] = self.fc2[i]

                i = i + 1

        np.save(conv_file_name, self.conv_dict)
        np.save(fc_file_name, self.fc_dict)



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





