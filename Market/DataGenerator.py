import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from Market.Conf import DATASET_PATH
from tensorflow.keras.applications import inception_v3, resnet_v2


class Dataset:

    def __init__(self, mode="training", val_frac=0.1):
        '''
        :param mode:
        :param val_frac:
        Note: data format (200, 3, 288, 144, 3) -> (number of subjects, samples, image_dim) -> samples (sktech1, sketch2, color_img1, color_img2, color_img3, color_img4)
        '''
        if mode == "training" or mode == "train_val":
            with open(DATASET_PATH + "dataTrain.pkl", 'rb') as f:
                ds = pickle.load(f)
        elif mode == "test":
            with open(DATASET_PATH + "dataTest.pkl", 'rb') as f:
                ds = pickle.load(f)
            with open(DATASET_PATH + "dataQuery.pkl", 'rb') as f:
                q = pickle.load(f)
                self.query = q

        self.data = ds



        self.labels = list(self.data.keys())
        if mode == "train_val":
            random.seed(1)
            random.shuffle(self.labels)
            self.val_labels = self.labels[:int(len(self.labels) * val_frac)]  # take 10% classes as validation
            del self.labels[:int(len(self.labels) * val_frac)]

    def get_mini_offline_batches(self, n_class, shots=2, validation=False):
        anchor_positive = np.zeros(shape=(n_class * shots, 128, 64, 3))
        labels = self.labels
        if validation:
            labels = self.val_labels
        label_subsets = random.sample(labels, k=n_class)
        for i in range(len(label_subsets)):
            if shots> len(self.data[label_subsets[i]]):
                positive_to_split = random.choices(
                    self.data[label_subsets[i]], k=shots)
            else:
                positive_to_split = random.sample(
                    self.data[label_subsets[i]], k=shots)


            # set anchor and pair positives

            anchor_positive[i * shots:(i + 1) * shots] = positive_to_split

        return resnet_v2.preprocess_input(anchor_positive.astype(np.float32))

    # def get_train_batches(self):

    def get_batches(self):

        n_query = 3368 # number of query images
        n_references = 19732 # number of references
        temp_labels = np.zeros(shape=(n_query))
        temp_images = np.zeros(shape=(n_query, 128, 64, 3))
        ref_labels = np.zeros(shape=(n_references))
        ref_images = np.zeros(shape=(n_references, 128, 64, 3))

        # prepare query
        query_class = list(self.query.keys())
        idx = 0
        for i in range(len(query_class)):
            query = self.query[query_class[i]]
            len_q = len(query)
            temp_labels[idx:idx+len_q] = int(query_class[i])
            temp_images[idx:idx+len_q] = query
            idx += len_q
        # prepare references
        ref_class = list(self.data.keys())
        idx = 0
        for i in range(len(ref_class)):
            ref = self.data[ref_class[i]]
            len_r = len(ref)
            ref_labels[idx:idx + len_r] = int(ref_class[i])
            ref_images[idx:idx + len_r] = ref
            idx += len_r


        return resnet_v2.preprocess_input(temp_images.astype(np.float32)), temp_labels.astype(np.int32), resnet_v2.preprocess_input(ref_images.astype(
            np.float32)), ref_labels.astype(np.float32)


if __name__ == '__main__':
    test_dataset = Dataset(mode="test")
    query, labels, references, ref_labels = test_dataset.get_batches()
    # train_data, domain_labels = test_dataset.get_mini_sketch_batches(n_class=25, shots=4)
    print(query.shape)

    # for _, data in enumerate(test_data):
    #     print(data)

    # _, axarr = plt.subplots(nrows=5, ncols=3, figsize=(20, 20))
    #
    # sample_keys = list(test_dataset.data.keys())
    #
    # for a in range(5):
    #     for b in range(2):
    #         temp_image = test_dataset.sketch[sample_keys[a]][b]
    #         # temp_image = np.stack((temp_image[:, :, :],), axis=-1)
    #         temp_image *= 255
    #         temp_image = np.clip(temp_image, 0, 255).astype("uint8")
    #         if b == 2:
    #             axarr[a, b].set_title("Class : " +  sample_keys[a])
    #         axarr[a, b].imshow(temp_image)
    #         axarr[a, b].xaxis.set_visible(False)
    #         axarr[a, b].yaxis.set_visible(False)
    # plt.show()
