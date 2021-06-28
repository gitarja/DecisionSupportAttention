import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from PKU.Conf import DATASET_PATH
from tensorflow.keras.applications import resnet

class Dataset:

    def __init__(self, mode="training", val_frac=0.1):

        if mode=="training" or mode=="train_val":
            with open(DATASET_PATH + "dataTrain.pkl", 'rb') as f:
                ds = pickle.load(f)
        elif mode=="test":
            with open(DATASET_PATH + "dataTest.pkl", 'rb') as f:
                ds = pickle.load(f)

        self.data = {}

        def extraction(image, idx=1):
            image = tf.image.convert_image_dtype(image, tf.float32)
            # if idx==1:
            #     image = image / 255.
            # else:
            #     image = resnet.preprocess_input(image)
            return image

        for i in range(ds.shape[0]):
            for l in range(ds[i].shape[0]):
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


    def get_mini_batches(self, n_buffer, batch_size, shots, num_classes, validation=False):

        temp_labels = np.zeros(shape=(num_classes * shots))
        temp_images = np.zeros(shape=(num_classes * shots, 288, 144, 3))
        label_subsets = random.choices(self.labels, k=num_classes)
        if validation:
            label_subsets = self.val_labels
            temp_labels = np.zeros(shape=(len(label_subsets) * shots))
            temp_images = np.zeros(shape=(len(label_subsets) * shots, 288, 144, 3))
        for class_idx, class_obj in enumerate(label_subsets):
            temp_labels[class_idx * shots: (class_idx + 1) * shots] = class_idx

            # sample images
            temp_images[class_idx * shots: (class_idx + 1) * shots] = random.choices(
                    self.data[label_subsets[class_idx]], k=shots)

        dataset = tf.data.Dataset.from_tensor_slices(
            (temp_images.astype(np.float32), temp_labels.astype(np.int32))
        )
        if validation:
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.shuffle(n_buffer).batch(batch_size)

        return dataset

    # def get_train_batches(self):

    def get_batches(self, shots, num_classes):
        random.seed(0)
        temp_labels = np.zeros(shape=(num_classes))
        temp_images = np.zeros(shape=(num_classes, 288, 144, 3))
        ref_labels = np.zeros(shape=(num_classes * shots))
        ref_images = np.zeros(shape=(num_classes * shots, 288, 144, 3))

        labels = self.labels
        random.shuffle(labels)

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
                temp_images[class_idx] = images_to_split[0]
                ref_images[class_idx * shots: (class_idx + 1) * shots] = images_to_split[1:shots+1]

            yield temp_images.astype(np.float32), temp_labels.astype(np.int32), ref_images.astype(np.float32), ref_labels.astype(np.float32)


if __name__ == '__main__':

    test_dataset = Dataset(mode="test")
    test_data = test_dataset.get_batches(shots=3, num_classes=5)

    # for _, data in enumerate(test_data):
    #     print(data)

    _, axarr = plt.subplots(nrows=5, ncols=3, figsize=(20, 20))

    sample_keys = list(test_dataset.data.keys())

    for a in range(5):
        for b in range(3):
            temp_image = test_dataset.data[sample_keys[a]][b]
            # temp_image = np.stack((temp_image[:, :, :],), axis=-1)
            temp_image *= 255
            temp_image = np.clip(temp_image, 0, 255).astype("uint8")
            if b == 2:
                axarr[a, b].set_title("Class : " +  sample_keys[a])
            axarr[a, b].imshow(temp_image)
            axarr[a, b].xaxis.set_visible(False)
            axarr[a, b].yaxis.set_visible(False)
    plt.show()

