import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from PKU.Conf import DATASET_PATH
from tensorflow.keras.applications import resnet50

class Dataset:

    def __init__(self, mode="training", val_frac=0.1):
        '''
        :param mode:
        :param val_frac:
        Note: data format (200, 3, 288, 144, 3) -> (number of subjects, samples, image_dim) -> samples (sktech1, sketch2, color_img1, color_img2, color_img3, color_img4)
        '''
        if mode=="training" or mode=="train_val":
            with open(DATASET_PATH + "dataTrain.pkl", 'rb') as f:
                ds = pickle.load(f)
        elif mode=="test":
            with open(DATASET_PATH + "dataTest.pkl", 'rb') as f:
                ds = pickle.load(f)

        self.data = {}
        self.sketch = {}

        def extraction(image, idx=1):
            image = tf.image.convert_image_dtype(image, tf.float32)
            # image = resnet50.preprocess_input(image)
            return image

        for i in range(ds.shape[0]):
            for k in range(2):
                label = str(i)
                if label not in self.sketch:
                    self.sketch[label] = []
                self.sketch[label].append(extraction(ds[i, k, :, :]))
            for l in range(2, ds[i].shape[0]):
                label = str(i)
                if label not in self.data:
                    self.data[label] = []
                self.data[label].append(extraction(ds[i, l, :, :]))

        self.labels = list(self.data.keys())
        if mode == "train_val":
            random.seed(1)
            random.shuffle(self.labels)
            self.val_labels = self.labels[:int(len(self.labels) *val_frac)] #take 10% classes as validation
            del self.labels[:int(len(self.labels) *val_frac)]


    def get_mini_offline_batches(self, n_class, shots=2,  validation=False):
        anchor_positive = np.zeros(shape=(n_class * (1 + shots), 288, 144, 3))
        domain_labels = np.zeros(1+shots)
        domain_labels[0] = 1
        domain_labels = np.tile(domain_labels, n_class)
        if shots>4:
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

            #set anchor and pair positives
            anchor_positive[i*shots, :] =sketch[0]
            anchor_positive[1 + (i*shots): 1 + ((i+1) * shots)] = positive_to_split
        domain_labels = np.expand_dims(domain_labels, -1)
        return anchor_positive.astype(np.float32), domain_labels.astype(np.float32)

    # def get_train_batches(self):

    def get_batches(self, shots, num_classes):
        random.seed(0)
        temp_labels = np.zeros(shape=(num_classes))
        temp_images = np.zeros(shape=(num_classes, 288, 144, 3))
        ref_labels = np.zeros(shape=(num_classes * shots))
        ref_images = np.zeros(shape=(num_classes * shots, 288, 144, 3))

        labels = self.labels


        for idx in range(0, len(labels), num_classes):
            label_subsets = labels[idx:idx+num_classes]

            for class_idx, class_obj in enumerate(label_subsets):
                temp_labels[class_idx] = class_idx
                ref_labels[class_idx*shots : (class_idx+1) * shots] = class_idx

                # sample images
                # images_to_split = random.choices(
                #     self.data[label_subsets[class_idx]], k=shots+1)
                images_to_split = random.choices(
                    self.data[label_subsets[class_idx]], k=shots+1)
                sketch = random.choices(
                    self.sketch[label_subsets[class_idx]], k=1)
                temp_images[class_idx] = sketch
                ref_images[class_idx * shots: (class_idx + 1) * shots] = images_to_split[1:shots+1]

            yield temp_images.astype(np.float32), temp_labels.astype(np.int32), ref_images.astype(np.float32), ref_labels.astype(np.float32)


if __name__ == '__main__':

    test_dataset = Dataset(mode="test")
    test_data = test_dataset.get_batches(shots=3, num_classes=5)
    train_data, _  = test_dataset.get_mini_offline_batches(n_class=25, shots=4)
    print(train_data.shape)

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

