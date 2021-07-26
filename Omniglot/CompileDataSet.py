from Omniglot.Conf import DATASET_PATH
import os
import glob
import cv2
import tensorflow as tf
import numpy as np
import pickle
train_split = DATASET_PATH + "/vinyals/trainval.txt"
train_pickle = DATASET_PATH + "/data_train.pickle"
all_images = []
with open(train_split, "r") as a_file:

  for line in a_file:
    stripped_path = line.strip()
    img_path = DATASET_PATH + "/images/" + "/".join(stripped_path.split("/")[:-1])
    rot = int(int(stripped_path.split("/")[-1].replace("rot", ""))/90)
    print(rot)
    imgs = []
    for img_file in glob.glob(img_path+"/*.png"):
      img = tf.image.resize(tf.image.rot90(cv2.imread(img_file), rot), [28,28]).numpy()
      imgs.append(np.mean(img, -1))
    all_images.append(imgs)

all_images = np.array(all_images)
print(all_images.shape)
with open(train_pickle, 'wb') as f:
  pickle.dump(all_images, f)
