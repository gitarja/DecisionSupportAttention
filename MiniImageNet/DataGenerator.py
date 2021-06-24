from MiniImageNet.Conf import DATASET_PATH
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
class Dataset:

    def __init__(self, mode="training"):

        if mode == "training":
            with open(DATASET_PATH + "mini-imagenet-cache-train.pkl", 'rb') as f:
                ds = pickle.load(f)
        elif mode == "validation":
            with open(DATASET_PATH + "mini-imagenet-cache-val.pkl", 'rb') as f:
                ds = pickle.load(f)
        else:
            with open(DATASET_PATH + "mini-imagenet-cache-test.pkl", 'rb') as f:
                ds = pickle.load(f)
        self.data = {}
        def extraction(image):
            image = tf.image.convert_image_dtype(image, tf.float32)
            return image
        images = ds["image_data"]
        images = images.reshape([-1, 600, 84, 84, 3])
        labels = list(ds["class_dict"].keys())
        for i in range(images.shape[0]):
            for l in range(images[i].shape[0]):
                label = labels[i]
                if label not in self.data:
                    self.data[label] = []
                self.data[label].append(extraction(images[i, l, :, :, :]))

        self.labels = list(self.data.keys())

    def get_mini_batches(self, batch_size, repetitions, shots, num_classes, split=False):

        temp_labels = np.zeros(shape=(num_classes * shots))
        temp_images = np.zeros(shape=(num_classes * shots, 28, 28, 1))

        if split:
            test_labels = np.zeros(shape=(num_classes))
            test_images = np.zeros(shape=(num_classes, 28, 28, 1))

        label_subsets = random.choices(self.labels, k=num_classes)

        for class_idx, class_obj in enumerate(label_subsets):

            temp_labels[class_idx * shots: (class_idx + 1) * shots] = class_idx

            if split:
                test_labels[class_idx] = class_idx

                images_to_split = random.choices(self.data[label_subsets[class_idx]], k=shots + 1)
                test_images[class_idx] = images_to_split[-1]
                temp_images[class_idx * shots: (class_idx + 1) * shots] = images_to_split[:-1]

            else:
                temp_images[class_idx * shots: (class_idx + 1) * shots] = random.choices(
                    self.data[label_subsets[class_idx]], k=shots)

            dataset = tf.data.Dataset.from_tensor_slices(
                (temp_images.astype(np.float32), temp_labels.astype(np.int32))
            )
            dataset = dataset.shuffle(100).batch(batch_size).repeat(repetitions)

            if split:
                return dataset, test_images, test_labels

            return dataset

if __name__ == '__main__':

    test_dataset = Dataset(mode="test")

    _, axarr = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

    sample_keys = list(test_dataset.data.keys())

    for a in range(5):
        for b in range(5):
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
