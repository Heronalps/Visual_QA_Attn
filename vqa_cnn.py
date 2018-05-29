import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from tqdm import tqdm
from resnet_v1 import resnet_v1_50
import resnet_utils
from scipy.misc import imread, imresize
from imagenet_classes import class_names

class NN(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.PHASE == 'train' else False
        self.train_cnn = self.is_train and config.TRAIN_CNN
        self.prepare()

    def prepare(self):
        """ Setup the weight initalizers and regularizers. """
        config = self.config

        self.conv_kernel_initializer = layers.xavier_initializer()

        if self.train_cnn and config.conv_kernel_regularizer_scale > 0:
            self.conv_kernel_regularizer = layers.l2_regularizer(
                scale = config.conv_kernel_regularizer_scale)
        else:
            self.conv_kernel_regularizer = None

        if self.train_cnn and config.conv_activity_regularizer_scale > 0:
            self.conv_activity_regularizer = layers.l1_regularizer(
                scale = config.conv_activity_regularizer_scale)
        else:
            self.conv_activity_regularizer = None

        self.fc_kernel_initializer = tf.random_uniform_initializer(
            minval = -config.fc_kernel_initializer_scale,
            maxval = config.fc_kernel_initializer_scale)

        if self.is_train and config.fc_kernel_regularizer_scale > 0:
            self.fc_kernel_regularizer = layers.l2_regularizer(
                scale = config.fc_kernel_regularizer_scale)
        else:
            self.fc_kernel_regularizer = None

        if self.is_train and config.fc_activity_regularizer_scale > 0:
            self.fc_activity_regularizer = layers.l1_regularizer(
                scale = config.fc_activity_regularizer_scale)
        else:
            self.fc_activity_regularizer = None

    def conv2d(self,
               inputs,
               filters,
               kernel_size = (3, 3),
               strides = (1, 1),
               activation = tf.nn.relu,
               use_bias = True,
               name = None):
        """ 2D Convolution layer. """
        if activation is not None:
            activity_regularizer = self.conv_activity_regularizer
        else:
            activity_regularizer = None
        return tf.layers.conv2d(
            inputs = inputs,
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding='same',
            activation = activation,
            use_bias = use_bias,
            trainable = self.train_cnn,
            kernel_initializer = self.conv_kernel_initializer,
            kernel_regularizer = self.conv_kernel_regularizer,
            activity_regularizer = activity_regularizer,
            name = name)

    def max_pool2d(self,
                   inputs,
                   pool_size = (2, 2),
                   strides = (2, 2),
                   name = None):
        """ 2D Max Pooling layer. """
        return tf.layers.max_pooling2d(
            inputs = inputs,
            pool_size = pool_size,
            strides = strides,
            padding='same',
            name = name)

    def dense(self,
              inputs,
              units,
              activation = tf.tanh,
              use_bias = True,
              name = None):
        """ Fully-connected layer. """
        if activation is not None:
            activity_regularizer = self.fc_activity_regularizer
        else:
            activity_regularizer = None
        return tf.layers.dense(
            inputs = inputs,
            units = units,
            activation = activation,
            use_bias = use_bias,
            trainable = self.is_train,
            kernel_initializer = self.fc_kernel_initializer,
            kernel_regularizer = self.fc_kernel_regularizer,
            activity_regularizer = activity_regularizer,
            name = name)

    def dropout(self,
                inputs,
                name = None):
        """ Dropout layer. """
        return tf.layers.dropout(
            inputs = inputs,
            rate = self.config.fc_drop_rate,
            training = self.is_train)

    def batch_norm(self,
                   inputs,
                   name = None):
        """ Batch normalization layer. """
        return tf.layers.batch_normalization(
            inputs = inputs,
            training = self.train_cnn,
            trainable = self.train_cnn,
            name = name
        )

class vqa_cnn():
    def __init__(self,config):
        self.config = config
        self.nn = NN(config)
        self.image_shape = self.config.IMAGE_SHAPE
        print("cnn_model_created")

    def build(self,images):
        """ Build the model. """
        self.build_cnn(images)

    def build_cnn(self,images):
        """ Build the CNN. """
        print("Building the CNN...")
        if self.config.CNN == 'vgg16':
            self.build_vgg16(images)
        else:
            self.build_resnet50(images)
        print("CNN built.")

    def build_vgg16(self,images):
        """ Build the VGG16 net. """
        config = self.config

        # conv1_1
        with tf.variable_scope('conv1_1', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(name='conv1_1_W', initializer=tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                                                       stddev=1e-1),
                                     trainable=config.TRAIN_CNN)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv1_1_b', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32),
                                     trainable=config.TRAIN_CNN)
            out1_1 = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out1_1)

        # # conv1_2
        with tf.variable_scope('conv1_2', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                                     stddev=1e-1), name='conv1_2_W',
                                     trainable=config.TRAIN_CNN)
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[64], dtype=tf.float32),
                                     trainable=config.TRAIN_CNN, name='conv1_2_b')
            out1_2 = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out1_2)
        #
        #
        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')
        #
        # # conv2_1
        with tf.variable_scope('conv2_1', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                                     stddev=1e-1), name='conv2_1_W',
                                     trainable=config.TRAIN_CNN)
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[128], dtype=tf.float32),
                                     trainable=config.TRAIN_CNN, name='conv2_1_b')
            out2_1 = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out2_1)
        #
        #
        # # conv2_2
        with tf.variable_scope('conv2_2', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                                     stddev=1e-1), name='conv2_2_W',
                                     trainable=config.TRAIN_CNN)
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[128], dtype=tf.float32),
                                     trainable=config.TRAIN_CNN, name='conv2_2_b')
            out2_2 = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out2_2)
        #
        #
        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')
        # # conv3_1
        with tf.variable_scope('conv3_1', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                                     stddev=1e-1), name='conv3_1_W',
                                     trainable=config.TRAIN_CNN)
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=config.TRAIN_CNN, name='conv3_1_b')
            out3_1 = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out3_1)
        #
        #
        # # conv3_2
        with tf.variable_scope('conv3_2', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                                     stddev=1e-1), name='conv3_2_W',
                                     trainable=config.TRAIN_CNN)
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=config.TRAIN_CNN, name='conv3_2_b')
            out3_2 = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out3_2)
        #
        #
        # # conv3_3
        with tf.variable_scope('conv3_3', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                                     stddev=1e-1), name='conv3_3_W',
                                     trainable=config.TRAIN_CNN)
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=config.TRAIN_CNN, name='conv3_3_b')
            out3_3 = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out3_3)

        # # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')
        #
        # # conv4_1
        with tf.variable_scope('conv4_1', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                                     stddev=1e-1), name='conv4_1_W',
                                     trainable=config.TRAIN_CNN)
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=config.TRAIN_CNN, name='conv4_1_b')
            out4_1 = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out4_1)
        #
        #
        # # conv4_2
        with tf.variable_scope('conv4_2', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                     stddev=1e-1), name='conv4_2_W',
                                     trainable=config.TRAIN_CNN)
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=config.TRAIN_CNN, name='conv4_2_b')
            out4_2 = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out4_2)
        #
        #
        # # conv4_3
        with tf.variable_scope('conv4_3', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                     stddev=1e-1), name='conv4_3_W',
                                     trainable=config.TRAIN_CNN)
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=config.TRAIN_CNN, name='conv4_3_b')
            out4_3 = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out4_3)
        #
        #
        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')
        #
        # # conv5_1
        with tf.variable_scope('conv5_1', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                     stddev=1e-1), name='conv5_1_W',
                                     trainable=config.TRAIN_CNN)
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=config.TRAIN_CNN, name='conv5_1_b')
            out5_1 = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out5_1)
        #
        #
        # # conv5_2
        with tf.variable_scope('conv5_2', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                     stddev=1e-1), name='conv5_2_W',
                                     trainable=config.TRAIN_CNN)
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=config.TRAIN_CNN, name='conv5_2_b')
            out5_2 = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out5_2)
        #
        #
        # # conv5_3
        #
        with tf.variable_scope('conv5_3', reuse=tf.AUTO_REUSE) as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                                     stddev=1e-1), name='conv5_3_W',
                                     trainable=config.TRAIN_CNN)
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=config.TRAIN_CNN, name='conv5_3_b')
            out5_3 = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out5_3)
        #
        #
        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')
        # # fc1
        with tf.variable_scope('fc6', reuse=tf.AUTO_REUSE) as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.get_variable(initializer=tf.truncated_normal([shape, 4096],
                                                                   dtype=tf.float32,
                                                                   stddev=1e-1), name='fc6_W',
                                   trainable=config.TRAIN_CNN)
            fc1b = tf.get_variable(initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                   name='fc6_b', trainable=config.TRAIN_CNN)
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
        #
        # # fc2
        with tf.variable_scope('fc7', reuse=tf.AUTO_REUSE) as scope:
            fc2w = tf.get_variable(initializer=tf.truncated_normal([4096, 4096],
                                                                   dtype=tf.float32,
                                                                   stddev=1e-1), name='fc7_W',
                                   trainable=config.TRAIN_CNN)
            fc2b = tf.get_variable(initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                   trainable=config.TRAIN_CNN, name='fc7_b')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
        #
        # fc3
        with tf.variable_scope('fc8', reuse=tf.AUTO_REUSE) as scope:
            fc3w = tf.get_variable(initializer=tf.truncated_normal([4096, 1000],
                                                                   dtype=tf.float32,
                                                                   stddev=1e-1), name='fc8_W',
                                   trainable=config.TRAIN_CNN)
            fc3b = tf.get_variable(initializer=tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                   trainable=True, name='fc8_b')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)

        # self.conv_feats = self.fc2
        ## Reshaping the 4096 to fit the lstm size
        reshaped_fc2_feats = tf.reshape(self.fc2,
                                        [config.BATCH_SIZE, 2, 2048])

        self.conv_feats = tf.reduce_mean(reshaped_fc2_feats, axis=1)
        self.num_ctx = 1
        self.dim_ctx = 2048
        self.images = images

    def build_resnet50(self,images):
        """ Build the ResNet50. """
        config = self.config

        conv1_feats = self.nn.conv2d(images,
                                  filters = 64,
                                  kernel_size = (7, 7),
                                  strides = (2, 2),
                                  activation = None,
                                  name = 'conv1')
        conv1_feats = self.nn.batch_norm(conv1_feats, 'bn_conv1')
        conv1_feats = tf.nn.relu(conv1_feats)
        pool1_feats = self.nn.max_pool2d(conv1_feats,
                                      pool_size = (3, 3),
                                      strides = (2, 2),
                                      name = 'pool1')

        res2a_feats = self.resnet_block(pool1_feats, 'res2a', 'bn2a', 64, 1)
        res2b_feats = self.resnet_block2(res2a_feats, 'res2b', 'bn2b', 64)
        res2c_feats = self.resnet_block2(res2b_feats, 'res2c', 'bn2c', 64)

        res3a_feats = self.resnet_block(res2c_feats, 'res3a', 'bn3a', 128)
        res3b_feats = self.resnet_block2(res3a_feats, 'res3b', 'bn3b', 128)
        res3c_feats = self.resnet_block2(res3b_feats, 'res3c', 'bn3c', 128)
        res3d_feats = self.resnet_block2(res3c_feats, 'res3d', 'bn3d', 128)

        res4a_feats = self.resnet_block(res3d_feats, 'res4a', 'bn4a', 256)
        res4b_feats = self.resnet_block2(res4a_feats, 'res4b', 'bn4b', 256)
        res4c_feats = self.resnet_block2(res4b_feats, 'res4c', 'bn4c', 256)
        res4d_feats = self.resnet_block2(res4c_feats, 'res4d', 'bn4d', 256)
        res4e_feats = self.resnet_block2(res4d_feats, 'res4e', 'bn4e', 256)
        res4f_feats = self.resnet_block2(res4e_feats, 'res4f', 'bn4f', 256)

        res5a_feats = self.resnet_block(res4f_feats, 'res5a', 'bn5a', 512)
        res5b_feats = self.resnet_block2(res5a_feats, 'res5b', 'bn5b', 512)
        res5c_feats = self.resnet_block2(res5b_feats, 'res5c', 'bn5c', 512)

        reshaped_res5c_feats = tf.reshape(res5c_feats,
                                         [config.BATCH_SIZE, 49, 2048])

        ## Reducing into 20148
        self.conv_feats = tf.reduce_mean(reshaped_res5c_feats, axis=1)
        self.num_ctx = 1
        self.dim_ctx = 2048
        self.images = images

    def resnet_block(self, inputs, name1, name2, c, s=2):
        """ A basic block of ResNet. """
        branch1_feats = self.nn.conv2d(inputs,
                                    filters = 4*c,
                                    kernel_size = (1, 1),
                                    strides = (s, s),
                                    activation = None,
                                    use_bias = False,
                                    name = name1+'_branch1')
        branch1_feats = self.nn.batch_norm(branch1_feats, name2+'_branch1')

        branch2a_feats = self.nn.conv2d(inputs,
                                     filters = c,
                                     kernel_size = (1, 1),
                                     strides = (s, s),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2a')
        branch2a_feats = self.nn.batch_norm(branch2a_feats, name2+'_branch2a')
        branch2a_feats = tf.nn.relu(branch2a_feats)

        branch2b_feats = self.nn.conv2d(branch2a_feats,
                                     filters = c,
                                     kernel_size = (3, 3),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2b')
        branch2b_feats = self.nn.batch_norm(branch2b_feats, name2+'_branch2b')
        branch2b_feats = tf.nn.relu(branch2b_feats)

        branch2c_feats = self.nn.conv2d(branch2b_feats,
                                     filters = 4*c,
                                     kernel_size = (1, 1),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2c')
        branch2c_feats = self.nn.batch_norm(branch2c_feats, name2+'_branch2c')

        outputs = branch1_feats + branch2c_feats
        outputs = tf.nn.relu(outputs)
        return outputs

    def resnet_block2(self, inputs, name1, name2, c):
        """ Another basic block of ResNet. """
        branch2a_feats = self.nn.conv2d(inputs,
                                     filters = c,
                                     kernel_size = (1, 1),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2a')
        branch2a_feats = self.nn.batch_norm(branch2a_feats, name2+'_branch2a')
        branch2a_feats = tf.nn.relu(branch2a_feats)

        branch2b_feats = self.nn.conv2d(branch2a_feats,
                                     filters = c,
                                     kernel_size = (3, 3),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2b')
        branch2b_feats = self.nn.batch_norm(branch2b_feats, name2+'_branch2b')
        branch2b_feats = tf.nn.relu(branch2b_feats)

        branch2c_feats = self.nn.conv2d(branch2b_feats,
                                     filters = 4*c,
                                     kernel_size = (1, 1),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2c')
        branch2c_feats = self.nn.batch_norm(branch2c_feats, name2+'_branch2c')

        outputs = inputs + branch2c_feats
        outputs = tf.nn.relu(outputs)
        return outputs

    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading the CNN from %s..." %data_path)
        ## Two separate conditions because one is vgg16 and other is resnet.
        if self.config.CNN == 'vgg16':
            data_dict = np.load(data_path,encoding='latin1')
            count = 0
            for param_name in tqdm(data_dict.keys()):
                op_name = param_name[:-2]
                with tf.variable_scope(op_name, reuse = True):
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data_dict[param_name]))
                        count += 1
                    except ValueError:
                        print("No such variable")
                        pass
        else:
            data_dict = np.load(data_path, encoding='latin1').item()
            count = 0
            for op_name in tqdm(data_dict):
                print(op_name )
                with tf.variable_scope(op_name, reuse = True):
                    for param_name, data in data_dict[op_name].items():
                        try:
                            var = tf.get_variable(param_name)
                            session.run(var.assign(data))
                            count += 1
                        except ValueError:
                            pass

        print("%d tensors loaded." %count)

    def tensorflow_resnet_model(self,images):
        image_shape = [224, 224, 3]
        inputs = images = tf.placeholder(
            dtype=tf.float32,
            shape=[self.config.BATCH_SIZE] + image_shape)
        with tf.contrib.slim.arg_scope(resnet_utils.resnet_arg_scope()):
            logits, endPoints = resnet_v1_50(inputs, num_classes=1000)
            probs = tf.nn.softmax(endPoints['predictions'])

        saver = tf.train.Saver()
        sess = tf.Session()

        saver.restore(sess, "./resnet_v1_50.ckpt")

        img1 = imread('./laska.png', mode='RGB')
        img1 = imresize(img1, (224, 224))

        prob = sess.run(probs, feed_dict={inputs: [img1]})[0]
        print(len(prob))
        preds = (np.argsort(prob)[::-1])
        for p in preds:
            print(class_names[p], prob[p])

    def test_cnn(self,sess):
        images = tf.placeholder(
            dtype=tf.float32,
            shape=[self.config.BATCH_SIZE] + self.image_shape)

        self.build_cnn(images)
        self.load_cnn(sess, self.config.CNN_PRETRAINED_FILE)

        probs = tf.nn.softmax(self.fc3l)
        img1 = imread('./laska.png', mode='RGB')
        img1 = imresize(img1, (224, 224))

        prob = sess.run(probs, feed_dict={images: [img1]})[0]
        preds = (np.argsort(prob)[::-1])[0:5]
        for p in preds:
            print(class_names[p], prob[p])



