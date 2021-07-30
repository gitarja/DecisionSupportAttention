import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from PKU.Conf import DATASET_PATH
from tensorflow.keras.applications import inception_v3, vgg16


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

        self.data = {}
        self.sketch = {}

        sketch_mean = np.array([220.3052445, 219.7853072, 219.32116368])
        sketch_std = np.array([55.41796559, 56.29420328, 57.02388277])
        rgb_mean = np.array([112.69836251, 110.19273992, 104.90438247])
        rgb_std = np.array([61.46471997, 61.41596831, 62.20044001])
        def extraction(image, sketch=False):
            image = tf.cast(image, tf.float32)
            if sketch:
                image = (image - sketch_mean) / sketch_std
            else:
                image = (image - rgb_mean) / rgb_std

            return image

        for i in range(ds.shape[0]):

            for k in range(2):
                label = str(i)
                if label not in self.sketch:
                    self.sketch[label] = []
                self.sketch[label].append(extraction(ds[i, k, :, :], True))
            for l in range(2, ds[i].shape[0]):
                label = str(i)
                if label not in self.data:
                    self.data[label] = []
                self.data[label].append(extraction(ds[i, l, :, :]))

        self.labels = list(self.data.keys())
        if mode == "train_val":
            random.seed(1)
            random.shuffle(self.labels)
            self.val_labels = self.labels[:int(len(self.labels) * val_frac)]  # take 10% classes as validation
            del self.labels[:int(len(self.labels) * val_frac)]

    def get_mini_offline_batches(self, n_class, shots=2, validation=False):
        anchor_positive = np.zeros(shape=(n_class * (1 + shots), 288, 144, 3))
        domain_labels = np.zeros(1 + shots)
        domain_labels[0] = 1
        domain_labels = np.tile(domain_labels, n_class)
        if shots > 4:
            raise ValueError('Shots must be less or equal to 4')
        labels = self.labels
        if validation:
            labels = self.val_labels
        label_subsets = random.sample(labels, k=n_class)
        for i in range(len(label_subsets)):
            sketch = random.sample(
                self.sketch[label_subsets[i]], k=1)
            positive_to_split = random.sample(
                self.data[label_subsets[i]], k=shots)

            # set anchor and pair positives
            anchor_positive[i * (1 + shots), :] = sketch[0]
            anchor_positive[1 + (i * (1 + shots)): ((i + 1) * (1 + shots))] = positive_to_split
        domain_labels = np.expand_dims(domain_labels, -1)
        return anchor_positive.astype(np.float32), domain_labels.astype(np.float32)

    def get_mini_sketch_batches(self, n_class, shots=2, validation=False):
        sketch_positive = np.zeros(shape=(n_class , 288, 144, 3))
        anchor_positive = np.zeros(shape=(n_class * (shots), 288, 144, 3))
        if shots > 4:
            raise ValueError('Shots must be less or equal to 4')
        labels = self.labels
        if validation:
            labels = self.val_labels
        label_subsets = random.sample(labels, k=n_class)
        for i in range(len(label_subsets)):
            sketch = random.sample(
                self.sketch[label_subsets[i]], k=1)
            positive_to_split = random.sample(
                self.data[label_subsets[i]], k=shots)

            # set anchor and pair positives
            sketch_positive[i, :] = sketch[0]
            anchor_positive[i * shots: (i + 1) * shots] = positive_to_split
        return sketch_positive.astype(np.float32), anchor_positive.astype(np.float32)

    # def get_train_batches(self):

    def get_batches(self, shots, num_classes):

        temp_labels = np.zeros(shape=(num_classes))
        temp_images = np.zeros(shape=(num_classes, 288, 144, 3))
        ref_labels = np.zeros(shape=(num_classes * shots))
        ref_images = np.zeros(shape=(num_classes * shots, 288, 144, 3))

        labels = self.labels

        label_subsets = random.sample(labels, k=num_classes)
        for class_idx, class_obj in enumerate(label_subsets):
            temp_labels[class_idx] = class_idx
            ref_labels[class_idx * shots: (class_idx + 1) * shots] = class_idx

            # sample images
            # images_to_split = random.choices(
            #     self.data[label_subsets[class_idx]], k=shots+1)
            images_to_split = random.choices(
                self.data[label_subsets[class_idx]], k=shots)
            sketch = random.choices(
                self.sketch[label_subsets[class_idx]], k=1)
            temp_images[class_idx] = sketch[0]
            ref_images[class_idx * shots: (class_idx + 1) * shots] = images_to_split

        return temp_images.astype(np.float32), temp_labels.astype(np.int32), ref_images.astype(
            np.float32), ref_labels.astype(np.float32)


if __name__ == '__main__':
    test_dataset = Dataset(mode="test")
    query, labels, references, ref_labels = test_dataset.get_batches(shots=2, num_classes=5)
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
