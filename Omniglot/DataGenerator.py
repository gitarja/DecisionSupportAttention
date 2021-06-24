import tensorflow as tf
import numpy as np
import random
from Omniglot.Conf import DATASET_PATH
import numpy as np
import matplotlib.pyplot as plt
import pickle


class Dataset:

    def __init__(self, training):

        if training:
            with open(DATASET_PATH + "dataTrain.pickle", 'rb') as f:
                ds = pickle.load(f)
        else:
            with open(DATASET_PATH + "dataTest.pickle", 'rb') as f:
                ds = pickle.load(f)

        self.data = {}

        def extraction(image):
            image = tf.expand_dims(tf.image.convert_image_dtype(image, tf.float32), -1)
            image = tf.image.resize(image, [105, 105])
            return image

        for i in range(ds.shape[0]):
            for l in range(ds[i].shape[0]):
                label = str(i)
                if label not in self.data:
                    self.data[label] = []
                self.data[label].append(extraction(ds[i, l, :, :]))

        self.labels = list(self.data.keys())

    def get_mini_batches(self, n_buffer, batch_size, repetitions, shots, num_classes, split=False, test_shots=1):

        temp_labels = np.zeros(shape=(num_classes * shots))
        temp_images = np.zeros(shape=(num_classes * shots, 105, 105, 1))

        if split:
            test_labels = np.zeros(shape=(num_classes * test_shots))
            test_images = np.zeros(shape=(num_classes * test_shots, 105, 105, 1))

        label_subsets = random.choices(self.labels, k=num_classes)

        for class_idx, class_obj in enumerate(label_subsets):

            temp_labels[class_idx * shots: (class_idx + 1) * shots] = class_idx

            if split:
                test_labels[class_idx] = class_idx
                images_to_split = random.choices(self.data[label_subsets[class_idx]], k=shots + test_shots)
                test_images[class_idx * test_shots: (class_idx + 1) * test_shots] = images_to_split[-test_shots]
                temp_images[class_idx * shots: (class_idx + 1) * shots] = images_to_split[:-test_shots]

            else:
                # sample images
                temp_images[class_idx * shots: (class_idx + 1) * shots] = random.choices(
                    self.data[label_subsets[class_idx]], k=shots)

        dataset = tf.data.Dataset.from_tensor_slices(
            (temp_images.astype(np.float32), temp_labels.astype(np.int32))
        )
        dataset = dataset.shuffle(n_buffer).batch(batch_size).repeat(repetitions)

        if split:
            return dataset, test_images, test_labels.astype(np.int32)

        return dataset

    def get_batches(self, shots, num_classes):

        temp_labels = np.zeros(shape=(num_classes))
        temp_images = np.zeros(shape=(num_classes, 105, 105, 1))
        ref_images = np.zeros(shape=(num_classes * shots, 105, 105, 1))

        labels = self.labels
        random.shuffle(labels)

        for idx in range(0, len(labels), num_classes):
            label_subsets = labels[idx:idx+num_classes]

            for class_idx, class_obj in enumerate(label_subsets):
                temp_labels[class_idx] = class_idx

                # sample images
                # images_to_split = random.choices(
                #     self.data[label_subsets[class_idx]], k=shots+1)
                images_to_split = self.data[label_subsets[class_idx]]
                temp_images[class_idx] = images_to_split[0]
                ref_images[class_idx * shots: (class_idx + 1) * shots] = images_to_split[1:shots+1]

                yield temp_images.astype(np.float32), temp_labels.astype(np.int32), ref_images.astype(np.float32),


if __name__ == '__main__':

    test_dataset = Dataset(training=False)
    test_data = test_dataset.get_batches(shots=5, num_classes=5)

    for _, data in enumerate(test_data):
        print(data)

    # _, axarr = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
    #
    # sample_keys = list(test_dataset.data.keys())
    #
    # for a in range(5):
    #     for b in range(5):
    #         temp_image = test_dataset.data[sample_keys[a]][b]
    #         temp_image = np.stack((temp_image[:, :, 0],) * 3, axis=2)
    #         temp_image *= 255
    #         temp_image = np.clip(temp_image, 0, 255).astype("uint8")
    #         if b == 2:
    #             axarr[a, b].set_title("Class : " + sample_keys[a])
    #         axarr[a, b].imshow(temp_image, cmap="gray")
    #         axarr[a, b].xaxis.set_visible(False)
    #         axarr[a, b].yaxis.set_visible(False)
    # plt.show()
