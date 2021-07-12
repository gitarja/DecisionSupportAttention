import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import random
from Omniglot.Conf import DATASET_PATH
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow_addons as tfa


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

        if mode == "test":
            for i in range(ds.shape[0]):
                for l in range(ds[i].shape[0]):
                    label = str(i)
                    if label not in self.data:
                        self.data[label] = []
                    self.data[label].append(extraction(ds[i, l, :, :]))
        else:
            label = 0
            for i in range(ds.shape[0]):
                for j in range(4):
                    for l in range(ds[i].shape[0]):
                        if label not in self.data:
                            self.data[label] = []
                        self.data[label].append(extraction(ds[i, l, :, :], size=(35, 35),degree=j))
                    label += 1

        self.labels = list(self.data.keys())
        if mode == "train_val":
            random.shuffle(self.labels)
            self.val_labels = self.labels[:int(len(self.labels) * val_frac)]  # take 20% classes as validation
            del self.labels[:int(len(self.labels) * val_frac)]

    def data_aug_pos(self, x):
        deg = np.random.uniform(0.05, .7, 1)
        x = tfa.image.rotate(x, deg, fill_mode="nearest")
        x = tf.image.random_crop(x, (len(x), 28, 28, 1))

        return x

    def data_aug_neg(self, x):
        deg = random.randint(1, 4)
        x = tf.image.random_crop(x, (len(x), 28, 28, 1))
        r = random.randint(0, 1) # whether to apply gaussian filter or not
        x = tf.image.rot90(x, deg)
        x = tf.image.random_brightness(x, 0.7)
        if r == 1:
            x = tfa.image.gaussian_filter2d(x)
        return x

    def get_mini_self_batches(self, n_buffer, batch_size, shots=2, validation=False):
        anchors = np.zeros(shape=(n_buffer * shots, 28, 28, 1))
        positives = np.zeros(shape=(n_buffer * shots, 28, 28, 1))
        negatives = np.zeros(shape=(n_buffer * shots, 28, 28, 1))
        labels = self.labels
        if validation:
            labels = self.val_labels

        for i in range(n_buffer):
            label_subsets = random.choices(labels, k=1)
            # set anchor and pair positives
            img_base = random.choices(
                self.data[label_subsets[0]], k=shots)
            anchors[i*shots:(i+1) * shots] = self.data_aug_pos(img_base)
            positives[i * shots:(i + 1) * shots] = self.data_aug_pos(img_base)
            negatives[i * shots:(i + 1) * shots] = self.data_aug_neg(img_base)


        dataset = tf.data.Dataset.from_tensor_slices(
            (anchors.astype(np.float32), positives.astype(np.float32), negatives.astype(np.float32) ))

        if validation:
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.shuffle(n_buffer * shots).batch(batch_size)

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
    test_data = test_dataset.get_mini_self_batches( n_buffer=5, batch_size=5, shots=1,validation=True)

    # for _, data in enumerate(test_data):
    #     print(data)

    _, axarr = plt.subplots(nrows=5, ncols=3, figsize=(20, 20))

    sample_keys = list(test_dataset.data.keys())

    i = 0
    for a, p, n in test_data:
        for j in range(len(a)):


            temp_image = np.stack((a[j][:, :, 0],) * 3, axis=2)
            temp_image *= 255
            temp_image = np.clip(temp_image, 0, 255).astype("uint8")
            axarr[i, 0].imshow(temp_image, cmap="gray")
            axarr[i, 0].xaxis.set_visible(False)
            axarr[i, 0].yaxis.set_visible(False)


            temp_image = np.stack((p[j][:, :, 0],) * 3, axis=2)
            temp_image *= 255
            temp_image = np.clip(temp_image, 0, 255).astype("uint8")
            axarr[i, 1].imshow(temp_image, cmap="gray")
            axarr[i, 1].xaxis.set_visible(False)
            axarr[i, 1].yaxis.set_visible(False)

            temp_image = np.stack((n[j][:, :, 0],) * 3, axis=2)
            temp_image *= 255
            temp_image = np.clip(temp_image, 0, 255).astype("uint8")
            axarr[i, 2].imshow(temp_image, cmap="gray")
            axarr[i, 2].xaxis.set_visible(False)
            axarr[i, 2].yaxis.set_visible(False)

            i+=1
        plt.show()
