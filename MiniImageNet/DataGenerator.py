from MiniImageNet.Conf import DATASET_PATH
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
class Dataset:

    def __init__(self, mode="training", val_frac=0.1):

        if mode == "training":
            with open(DATASET_PATH + "mini-imagenet-cache-train.pkl", 'rb') as f:
                ds = pickle.load(f)
        elif mode == "validation":
            with open(DATASET_PATH + "mini-imagenet-cache-val.pkl", 'rb') as f:
                ds = pickle.load(f)
        elif mode == "train_val":
            with open(DATASET_PATH + "mini-imagenet-cache-train.pkl", 'rb') as f:
                ds_train = pickle.load(f)
            with open(DATASET_PATH + "mini-imagenet-cache-val.pkl", 'rb') as f:
                ds_val = pickle.load(f)
            ds = ds_train
            ds.update(ds_val)
        else:
            with open(DATASET_PATH + "mini-imagenet-cache-test.pkl", 'rb') as f:
                ds = pickle.load(f)

        self.data = {}
        mean = np.array([0.485, 0.456, 0.406])
        std =  np.array([0.229, 0.224, 0.225])
        def extraction(image):
            image = ((image / 255.) - mean) / std
            return image

        images = ds["image_data"]
        images = images.reshape([-1, 600, 84, 84, 3])
        labels = list(ds["class_dict"].keys())
        for i in range(images.shape[0]):
            for l in range(images[i].shape[0]):
                label = str(i)
                if label not in self.data:
                    self.data[label] = []
                self.data[label].append(extraction(images[i, l, :, :, :]))

        self.labels = list(self.data.keys())


    def get_mini_offline_batches(self, n_class, shots=2):
        anchor_positive = np.zeros(shape=(n_class * shots, 84, 84, 3))
        anchor_labels = np.zeros(shape=(n_class * shots, 84, 84, 3))

        label_subsets = random.sample(self.labels, k=n_class)

        for i in range(len(label_subsets)):
            positive_to_split = random.sample(
                self.data[label_subsets[i]], k=shots)

            #set anchor and pair positives

            anchor_positive[i*shots:(i+1) * shots] = positive_to_split
            anchor_labels[i * shots:(i + 1) * shots] = i

        return anchor_positive.astype(np.float32), anchor_labels

    def get_batches(self, shots, num_classes):

        temp_labels = np.zeros(shape=(num_classes))
        temp_images = np.zeros(shape=(num_classes, 84, 84, 3))
        ref_images = np.zeros(shape=(num_classes * shots, 84, 84, 3))
        ref_labels = np.zeros(shape=(num_classes * shots))

        labels = self.labels
        label_subsets = random.sample(self.labels, k=num_classes)
        for idx in range(0, len(labels), num_classes):

            for i in range(len(label_subsets)):
                temp_labels[i] = i
                ref_labels[i * shots: (i + 1) * shots] = i
                # sample images
                images_to_split = random.sample(
                    self.data[label_subsets[i]], k=shots + 1)

                temp_images[i] = images_to_split[-1]
                ref_images[i * shots: (i + 1) * shots] = images_to_split[:-1]

            return temp_images.astype(np.float32), temp_labels.astype(np.int32), ref_images.astype(np.float32),  ref_labels.astype(np.int32)

if __name__ == '__main__':

    test_dataset = Dataset(mode="training")
    test_data, _ = test_dataset.get_mini_offline_batches(shots=5, n_class=5)
    _, _, evaluation_data, _ = test_dataset.get_batches(5, 5)

    # for _, data in enumerate(test_data):
    #     print(data)

    _, axarr = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

    sample_keys = list(test_dataset.data.keys())
    j = 0
    for a in range(5):
        for b in range(5):
            temp_image = evaluation_data[j]

            temp_image = np.clip(temp_image, 0, 255).astype("uint8")
            if b == 2:
                axarr[a, b].set_title("Class : " + sample_keys[a])
            axarr[a, b].imshow(temp_image)
            axarr[a, b].xaxis.set_visible(False)
            axarr[a, b].yaxis.set_visible(False)
            j+=1
    plt.show()

