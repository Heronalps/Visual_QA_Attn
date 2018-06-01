## This file contains the class for the data object we create

import numpy as np

class DataSet(object):
    def __init__(self,
                 image_id_list,
                 image_file_list,
                 question_id_list=None,
                 question_idxs_list=None,
                 question_masks_list=None,
                 question_type_list=None,
                 answer_id_list=None,
                 answer_idxs_list=None,
                 answer_masks_list=None,
                 answer_type_list=None,
                 batch_size=1,
                 phase="train",
                 shuffle=False):


        self.image_id_list = np.array(image_id_list)
        self.image_file_list = np.array(image_file_list)

        self.question_id_list = np.array(question_id_list)
        self.question_idxs_list = np.array(question_idxs_list)
        self.question_masks_list = np.array(question_masks_list)
        self.question_type_list = np.array(question_type_list)

        self.answer_ids_list = np.array(answer_id_list)
        self.answer_idxs_list = np.array(answer_idxs_list)
        self.answer_masks_list = np.array(answer_masks_list)
        self.answer_type_list = np.array(answer_type_list)

        self.batch_size = batch_size
        self.phase = phase
        self.shuffle = shuffle
        self.setup()

    def setup(self):
        """ Setup the dataset. """
        self.count = len(self.image_id_list)
        self.num_batches = int(np.ceil(self.count * 1.0 / self.batch_size))
        self.fake_count = self.num_batches * self.batch_size - self.count
        self.idxs = list(range(self.count))
        self.reset()

    def reset(self):
        """ Reset the dataset. """
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def next_batch(self):
        """ Fetch the next batch. """
        assert self.has_next_batch()

        if self.has_full_next_batch():
            start, end = self.current_idx, \
                         self.current_idx + self.batch_size
            current_idxs = self.idxs[start:end]
        else:
            start, end = self.current_idx, self.count
            current_idxs = self.idxs[start:end] + \
                           list(np.random.choice(self.count, self.fake_count))

        image_files = self.image_file_list[current_idxs]
        image_idxs = self.image_id_list[current_idxs]


        if self.phase == "train":
            question_idxs = self.question_idxs_list[current_idxs]
            question_masks = self.question_masks_list[current_idxs]
            answer_idxs = self.answer_idxs_list[current_idxs]
            answer_masks = self.answer_masks_list[current_idxs]
            self.current_idx += self.batch_size
            return image_files,image_idxs, question_idxs, question_masks, answer_idxs, answer_masks
        elif self.phase == "test":
            question_idxs = self.question_idxs_list[current_idxs]
            question_masks = self.question_masks_list[current_idxs]
            self.current_idx += self.batch_size
            return image_files,image_idxs,question_idxs,question_masks
        elif self.phase == "cnn_features":
            self.current_idx += self.batch_size
            return image_files, image_idxs



    def has_next_batch(self):
        """ Determine whether there is a batch left. """
        return self.current_idx < self.count

    def has_full_next_batch(self):
        """ Determine whether there is a full batch left. """
        return self.current_idx + self.batch_size <= self.count






