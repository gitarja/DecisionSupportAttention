import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import random
class Dataset:

    def __init__(self, training):
        split = "train" if training else "test"
        ds = tfds.load("omniglot", split=split, as_supervised=True, shuffle_files=False)

        self.data = {}

        def extraction(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [28, 28])
            return image, label

        for image, label in ds.map(extraction):
            image = image.numpy()
            label = str(label.numpy())

            if label not in self.data:
                self.data[label] = []
            self.data[label] = image
            self.labels = list(self.data.keys())


    def get_mini_batches(self, batch_size, repetitions, shots, num_classes, split=False):

        temp_labels = np.zeros(shape=(num_classes * shots))
        temp_images = np.zeros(shape=(num_classes * shots, 28, 28, 1))

        if split:
            test_labels = np.zeros(shape=(num_classes))
            test_images = np.zeros(shape=(num_classes, 28, 28, 1))


        label_subsets = random.choices(self.labels, k=num_classes)

        for class_idx, class_obj in enumerate(label_subsets):

            temp_labels[class_idx *  shots: (class_idx+1)*shots] = class_idx

            if split:
                test_labels[class_idx] = class_idx

                images_to_split = random.choices(self.data[label_subsets[class_idx]], k=shots+1)
                test_images[class_idx] = images_to_split[-1]
                temp_images[class_idx * shots : (class_idx+1) *  shots] = images_to_split[:-1]

            else:
                temp_images[class_idx * shots: (class_idx + 1) * shots] = random.choices(self.data[label_subsets[class_idx]], k=shots)

            dataset = tf.data.Dataset.from_tensor_slices(
                (temp_images.astype(np.float32), temp_labels.astype(np.int32))
            )
            dataset = dataset.shuffle(100).batch(batch_size).repeat(repetitions)

            if split:
                return dataset, test_images, test_labels

            return dataset



