import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import random
from Omniglot.Conf import DATASET_PATH
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.applications import resnet


class Dataset:

    def __init__(self, mode="training", val_frac=0.1):

        if mode=="train" or mode=="train_val":
            with open(DATASET_PATH + "dataTrain.pickle", 'rb') as f:
                ds = pickle.load(f)
        elif mode=="test":
            with open(DATASET_PATH + "dataTest.pickle", 'rb') as f:
                ds = pickle.load(f)

        self.data = {}

        def extraction(image, degree=0):
            image = tf.image.resize(tf.expand_dims(tf.image.convert_image_dtype(image, tf.float32), -1), [28, 28])
            if degree!=0:
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
                for j in range(1, 4, 1):
                    for l in range(ds[i].shape[0]):
                        if label not in self.data:
                            self.data[label] = []
                        self.data[label].append(extraction(ds[i, l, :, :], j))
                    label+=1

        self.labels = list(self.data.keys())
        if mode == "train_val":
            random.shuffle(self.labels)
            self.val_labels = self.labels[:int(len(self.labels) *val_frac)] #take 20% classes as validation
            del self.labels[:int(len(self.labels) *val_frac)]





    def get_mini_offline_batches(self, n_class, shots=2,  validation=False):
        anchor_positive = np.zeros(shape=(n_class * shots, 28, 28, 1))
        label_subsets = random.sample(n_class, k=2)
        for i in range(len(label_subsets)):
            positive_to_split = random.sample(
                self.data[label_subsets[i]], k=shots)

            #set anchor and pair positives

            anchor_positive[i*shots:(i+1) * shots] = positive_to_split

        return anchor_positive.astype(np.float32)


    def get_mini_pairoffline_batches(self, n_buffer, batch_size, shots=2,  validation=False):
        anchor_positive = np.zeros(shape=(n_buffer * shots, 28, 28, 1))
        anchor_negative = np.zeros(shape=(n_buffer * shots, 28, 28, 1))

        # pairs
        pair_positive = np.zeros(shape=(n_buffer * shots, 28, 28, 1))
        pair_negative = np.zeros(shape=(n_buffer * shots, 28, 28, 1))
        labels = self.labels
        if validation:
            labels = self.val_labels


        for i in range(n_buffer):
            label_subsets = random.sample(labels, k=2)
            if shots <= len(self.data[label_subsets[0]]):
                positive_sample = random.sample(
                    self.data[label_subsets[0]], k=shots)
                negative_sample = random.sample(
                    self.data[label_subsets[1]], k=shots)
                anchor_positive[i * shots:(i + 1) * shots] = positive_sample
                anchor_negative[i * shots:(i + 1) * shots] = negative_sample
                pair_positive[i * shots:(i + 1) * shots] = positive_sample[::-1]
                pair_negative[i * shots:(i + 1) * shots] = negative_sample[::-1]
            else:
                anchor_positive[i*shots:(i+1) * shots] = random.choices(
                    self.data[label_subsets[0]], k=shots)
                anchor_negative[i*shots:(i+1) * shots] = random.choices(
                    self.data[label_subsets[1]], k=shots)
                pair_positive[i * shots:(i + 1) * shots] = random.choices(
                    self.data[label_subsets[0]], k=shots)
                pair_negative[i * shots:(i + 1) * shots] = random.choices(
                    self.data[label_subsets[1]], k=shots)





        dataset = tf.data.Dataset.from_tensor_slices(
            (anchor_positive.astype(np.float32), pair_positive.astype(np.float32), anchor_negative.astype(np.float32),
             pair_negative.astype(np.float32))
        )
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
            label_subsets = random.sample(self.labels, k=num_classes)
            for class_idx, class_obj in enumerate(label_subsets):
                temp_labels[class_idx] = class_idx
                ref_labels[class_idx * shots: (class_idx + 1) * shots] = class_idx

                # sample images

                images_to_split = random.sample(
                    self.data[label_subsets[class_idx]], k=shots + 1)
                temp_images[class_idx] = images_to_split[-1]
                ref_images[class_idx * shots: (class_idx + 1) * shots] = images_to_split[:-1]
        else:
            # generate support
            support_labels = random.sample(self.labels[:int(len(self.labels)/2)], k=num_classes)
            for class_idx, class_obj in enumerate(support_labels):
                ref_labels[class_idx * shots: (class_idx + 1) * shots] = class_idx
                ref_images[class_idx * shots: (class_idx + 1) * shots] = random.choices(
                    self.data[support_labels[class_idx]], k=shots)

            # generate query
            query_labels = random.sample(self.labels[int(len(self.labels) / 2):], k=num_classes)
            for class_idx, class_obj in enumerate(query_labels):
                temp_labels[class_idx ] = class_idx
                ref_images[class_idx] = random.choices(self.data[query_labels[class_idx]])

        return temp_images.astype(np.float32), temp_labels.astype(np.int32), ref_images.astype(
            np.float32), ref_labels.astype(np.float32)



if __name__ == '__main__':

    test_dataset = Dataset(mode="test")
    test_data = test_dataset.get_batches(shots=5, num_classes=5)

    # for _, data in enumerate(test_data):
    #     print(data)

    _, axarr = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

    sample_keys = list(test_dataset.data.keys())

    for a in range(5):
        for b in range(5):
            temp_image = test_dataset.data[sample_keys[a]][b]
            temp_image = np.stack((temp_image[:, :, 0],) * 3, axis=2)
            temp_image *= 255
            temp_image = np.clip(temp_image, 0, 255).astype("uint8")
            if b == 2:
                axarr[a, b].set_title("Class : " + sample_keys[a])
            axarr[a, b].imshow(temp_image, cmap="gray")
            axarr[a, b].xaxis.set_visible(False)
            axarr[a, b].yaxis.set_visible(False)
    plt.show()
