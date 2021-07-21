import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import random
from Omniglot.Conf import DATASET_PATH
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow_addons as tfa
import math


class Dataset:

    def __init__(self, mode="training", val_frac=0.1):

        if mode == "training" or mode == "train_val":
            with open(DATASET_PATH + "dataTrain.pickle", 'rb') as f:
                ds = pickle.load(f)
        elif mode == "test":
            with open(DATASET_PATH + "dataTest.pickle", 'rb') as f:
                ds = pickle.load(f)

        self.data = {}

        def extraction(image, size=(28, 28), degree=0):
            image = tf.image.resize(tf.expand_dims(tf.image.convert_image_dtype(image, tf.float32), -1),size)
            if degree != 0:
                image = tf.image.rot90(image, k=degree)
            return image

        for i in range(ds.shape[0]):
            for l in range(ds[i].shape[0]):
                label = str(i)
                if label not in self.data:
                    self.data[label] = []
                if mode == "test":
                    self.data[label].append(extraction(ds[i, l, :, :]))
                else:
                    self.data[label].append(extraction(ds[i, l, :, :], size=(33, 33)))


        self.labels = list(self.data.keys())
        if mode == "train_val":
            random.shuffle(self.labels)
            self.val_labels = self.labels[:int(len(self.labels) * val_frac)]  # take 20% classes as validation
            del self.labels[:int(len(self.labels) * val_frac)]

        self.random_zoomout = tf.keras.layers.experimental.preprocessing.RandomZoom((-0.1, 0.1))

        self.random_zoomout_neg = tf.keras.layers.experimental.preprocessing.RandomZoom((-0.7, 0.7), (-0.7, 0.7))

    def data_aug_pos(self, x):
        x = tf.expand_dims(tf.expand_dims(x, -1), 0)
        deg = tf.random.uniform([len(x)], .1, 15.)
        optional = tf.random.uniform([len(x)], 0, 1, dtype=tf.int32)
        rad = ((tf.cast(optional, tf.float32) * 345.) + deg) * (math.pi / 180)
        x = tfa.image.rotate(x, rad, fill_mode="nearest")
        if optional == 1:
            x = self.random_zoomout(x)
        return x.numpy()[0,:,:,0]

    def data_aug_neg(self, x):
        deg = np.random.uniform(90., 270., 1) * (math.pi / 180)

        x = tfa.image.rotate(x, deg, fill_mode="nearest")
        x = tf.image.random_brightness(x, 0.2)
        # x = tf.image.random_crop(x, (len(x), 28, 28, 1))
        # x = self.random_zoomout(np.array(x))

        return x

    def get_mini_self_batches(self, batch_size, shots=2, validation=False):
        labels = self.labels
        if validation:
            labels = self.val_labels
        n_class = len(labels)
        anchors = np.zeros(shape=(n_class * shots, 33, 33, 1))
        for i in range(n_class):
            # set anchor and pair positives
            img_base = random.choices(
                self.data[labels[i]], k=shots)
            anchors[i*shots:(i+1) * shots] = img_base



        dataset = tf.data.Dataset.from_tensor_slices(
            (anchors.astype(np.float32)))

        if validation:
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.shuffle(n_class * shots).batch(batch_size)

        return dataset
    def get_mini_self_double_batches(self, batch_size, shots=2, validation=False):
        labels = self.labels
        if validation:
            labels = self.val_labels
        n_class = len(labels)
        anchors = np.zeros(shape=(n_class * shots, 28, 28, 1))
        positives = np.zeros(shape=(n_class * shots, 28, 28, 1))
        negatives = np.zeros(shape=(n_class * shots, 28, 28, 1))
        negatives2 = np.zeros(shape=(n_class * shots, 28, 28, 1))
        for i in range(n_class):
            # set anchor and pair positives
            img_base = random.choices(
                self.data[labels[i]], k=shots)
            anchors[i*shots:(i+1) * shots] = self.data_aug_pos(img_base)
            positives[i * shots:(i + 1) * shots] = self.data_aug_pos(img_base)
            negatives[i * shots:(i + 1) * shots] = self.data_aug_neg(img_base)
            negatives2[i * shots:(i + 1) * shots] = self.data_aug_pos( negatives[i * shots:(i + 1) * shots])


        dataset = tf.data.Dataset.from_tensor_slices(
            (anchors.astype(np.float32), positives.astype(np.float32), negatives.astype(np.float32), negatives2.astype(np.float32) ))

        if validation:
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.shuffle(n_class * shots).batch(batch_size)

        return dataset


    def get_mini_batches(self, n_class, batch_size, shots=2, validation=False):
        labels = self.labels
        if validation:
            labels = self.val_labels

        anchors = np.zeros(shape=(n_class * shots, 28, 28, 1))
        positives = np.zeros(shape=(n_class * shots, 28, 28, 1))
        negatives = np.zeros(shape=(n_class * shots, 28, 28, 1))
        for i in range(n_class):
            label_subset = random.choices(
                labels, k=2)
            # set anchor and pair positives
            img_base_pos = random.choices(
                self.data[label_subset[0]], k=shots)
            img_base_neg = random.choices(
                self.data[label_subset[1]], k=shots)
            anchors[i*shots:(i+1) * shots] = self.data_aug_pos(img_base_pos)
            positives[i * shots:(i + 1) * shots] = self.data_aug_pos(img_base_pos)
            negatives[i * shots:(i + 1) * shots] = tf.image.resize(img_base_neg, (28, 28))


        dataset = tf.data.Dataset.from_tensor_slices(
            (anchors.astype(np.float32), positives.astype(np.float32), negatives.astype(np.float32) ))

        if validation:
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.shuffle(n_class * shots * 2).batch(batch_size)

        return dataset

    def get_batches(self, shots, num_classes, outlier=False):

        temp_labels = np.zeros(shape=(num_classes))
        temp_images = np.zeros(shape=(num_classes, 28, 28, 1))
        ref_labels = np.zeros(shape=(num_classes * shots))
        ref_images = np.zeros(shape=(num_classes * shots, 28, 28, 1))

        if outlier == False:
            label_subsets = random.choices(self.labels, k=num_classes)
            for class_idx, class_obj in enumerate(label_subsets):
                temp_labels[class_idx] = class_idx
                ref_labels[class_idx * shots: (class_idx + 1) * shots] = class_idx

                # sample images

                images_to_split = random.choices(
                    self.data[label_subsets[class_idx]], k=shots + 1)
                temp_images[class_idx] = images_to_split[-1]
                ref_images[class_idx * shots: (class_idx + 1) * shots] = images_to_split[:-1]
        else:
            # generate support
            support_labels = random.choices(self.labels[:int(len(self.labels) / 2)], k=num_classes)
            for class_idx, class_obj in enumerate(support_labels):
                ref_labels[class_idx * shots: (class_idx + 1) * shots] = class_idx
                ref_images[class_idx * shots: (class_idx + 1) * shots] = random.choices(
                    self.data[support_labels[class_idx]], k=shots)

            # generate query
            query_labels = random.choices(self.labels[int(len(self.labels) / 2):], k=num_classes)
            for class_idx, class_obj in enumerate(query_labels):
                temp_labels[class_idx] = class_idx
                ref_images[class_idx] = random.choices(self.data[query_labels[class_idx]])

        return temp_images.astype(np.float32), temp_labels.astype(np.int32), ref_images.astype(
            np.float32), ref_labels.astype(np.float32)


if __name__ == '__main__':

    test_dataset = Dataset(mode="train_val")
    test_data = test_dataset.get_mini_self_batches(batch_size=5, shots=1,validation=True)

    # for _, data in enumerate(test_data):
    #     print(data)

    _, axarr = plt.subplots(nrows=5, ncols=3, figsize=(20, 20))

    sample_keys = list(test_dataset.data.keys())

    i = 0
    for a in test_data:
        for j in range(len(a)):


            temp_image = np.stack((a[j][:, :, 0], ) * 3, axis=2)
            temp_image *= 255
            temp_image = np.clip(temp_image, 0, 255).astype("uint8")
            axarr[i, 0].imshow(temp_image, cmap="gray")
            axarr[i, 0].xaxis.set_visible(False)
            axarr[i, 0].yaxis.set_visible(False)


            temp_image = np.stack((test_dataset.data_aug_pos(a[j][:, :, 0]), ) * 3, axis=2)
            temp_image *= 255
            temp_image = np.clip(temp_image, 0, 255).astype("uint8")
            axarr[i, 1].imshow(temp_image, cmap="gray")
            axarr[i, 1].xaxis.set_visible(False)
            axarr[i, 1].yaxis.set_visible(False)

            # temp_image = np.stack((test_dataset.data_aug_neg(a[j][:, :, 0]),) * 3, axis=2)
            # temp_image *= 255
            # temp_image = np.clip(temp_image, 0, 255).astype("uint8")
            # axarr[i, 2].imshow(temp_image, cmap="gray")
            # axarr[i, 2].xaxis.set_visible(False)
            # axarr[i, 2].yaxis.set_visible(False)

            i+=1
        plt.show()
