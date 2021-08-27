import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from Market.Conf import DATASET_PATH

import glob
from PIL import Image

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
            random.shuffle(self.labels)
            self.val_labels = self.labels[:int(len(self.labels) * val_frac)]  # take 10% classes as validation
            del self.labels[:int(len(self.labels) * val_frac)]

    def get_mini_offline_batches(self, n_class, shots=2, validation=False):
        anchor_positive = np.zeros(shape=(n_class * shots, 128, 64, 3))
        anchor_labels = np.zeros(shape=(n_class * shots, 1))
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
            anchor_labels[i * shots:(i + 1) * shots] = i

        return anchor_positive.astype(np.float32), anchor_labels

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


        return temp_images.astype(np.float32), temp_labels.astype(np.int32), ref_images.astype(
            np.float32), ref_labels.astype(np.int32)

class DataTest:


    def getTestData(self):
        n_query = 3368  # number of query images
        n_references = 19732  # number of references
        temp_labels = np.zeros(shape=(n_query))
        temp_images = np.zeros(shape=(n_query, 128, 64, 3))
        temp_cams = np.zeros(shape=(n_query))

        ref_labels = np.zeros(shape=(n_references))
        ref_images = np.zeros(shape=(n_references, 128, 64, 3))
        ref_cams = np.zeros(shape=(n_references))

        path = DATASET_PATH
        i = 0
        #prepare query
        for file in sorted(glob.glob(path + "query/*.jpg")):
            idx = file.split("/")[-1].split("_")[0]
            cam_idx = file.split("/")[-1].split("_")[1][1]

            im = np.array(Image.open(file))
            #assign files
            temp_images[i] = im
            temp_labels[i] = int(idx)
            temp_cams[i] = int(cam_idx)
            i+=1

        # prepare refereces
        i=0
        for file in sorted(glob.glob(path + "bounding_box_test/*.jpg")):
            idx = file.split("/")[-1].split("_")[0]
            cam_idx = file.split("/")[-1].split("_")[1][1]

            im = np.array(Image.open(file))
            # assign files
            ref_images[i] = im
            ref_labels[i] = int(idx)
            ref_cams[i] = int(cam_idx)
            i += 1

        return temp_images.astype(np.float32), temp_labels, temp_cams, ref_images.astype(np.float32), ref_labels, ref_cams




if __name__ == '__main__':
    test_dataset = Dataset("train_val")
    n_class = 5
    # test_dataset.getTestData()
    # query, labels, references, ref_labels = test_dataset.get_batches()
    train_data, _ = test_dataset.get_mini_offline_batches(n_class=n_class, shots=2)
    print(train_data.shape)
    # print(query.shape)
    # print(references.shape)
    #
    # for _, data in enumerate(test_data):
    #     print(data)
    #
    _, axarr = plt.subplots(nrows=n_class, ncols=2)

    j=0
    for a in range(n_class):
        for b in range(2):
            temp_image = train_data[j]
            # temp_image = np.stack((temp_image[:, :, :],), axis=-1)

            temp_image = np.clip(tf.image.resize(temp_image, (256, 128)).numpy(), 0, 255).astype("uint8")

            axarr[a, b].imshow(temp_image)
            axarr[a, b].xaxis.set_visible(False)
            axarr[a, b].yaxis.set_visible(False)
            j+=1
    plt.show()
