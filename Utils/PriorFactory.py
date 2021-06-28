#refer: https://github.com/hwalsuklee/tensorflow-mnist-AAE/blob/master/prior_factory.py
import numpy as np
from math import sin,cos,sqrt
import tensorflow as tf

def Gaussian(batch_size, n_dim, mean=0, var=1, n_labels=10, use_label_info=False):
    tf.random.set_seed(0)
    if use_label_info:
        # if n_dim != 2 or n_labels != 10:
        #     raise Exception("n_dim must be 2 and n_labels must be 10.")

        def sample(n_labels):
            x, y = np.random.normal(mean, var, (2,))
            angle = np.angle((x-mean) + 1j*(y-mean), deg=True)
            dist = np.sqrt((x-mean)**2+(y-mean)**2)

            # label 0
            if dist <1.0:
                label = 0
            else:
                label = ((int)((n_labels-1)*angle))//360

                if label<0:
                    label+=n_labels-1

                label += 1

            return np.array([x, y]).reshape((2,)), label

        z = np.empty((batch_size, n_dim), dtype=np.float32)
        z_id = np.empty((batch_size), dtype=np.int32)
        for batch in range(batch_size):
            for zi in range((int)(n_dim/2)):
                    a_sample, a_label = sample(n_labels)
                    z[batch, zi*2:zi*2+2] = a_sample
                    z_id[batch] = a_label
        return z
    else:
        z = tf.cast(tf.random.normal((batch_size, n_dim), mean, var), tf.float32)
        return z

def GaussianMultivariate(batch_size, n_dim, mean=0., var=1.):
    cov_mat = np.diag([var for i in range(n_dim)])
    mean_vec = [mean for i in range(n_dim)]

    z = np.random.multivariate_normal(mean_vec, cov_mat, (batch_size, ))
    return tf.cast(z, tf.float32)

def GaussianMixture(batch_size, n_dim=2, n_labels=10, x_var=0.5, y_var=0.1, label_indices=None):
    # if n_dim != 2:
    #     raise Exception("n_dim must be 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, (int)(n_dim/2)))
    y = np.random.normal(0, y_var, (batch_size, (int)(n_dim/2)))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range((int)(n_dim/2)):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)

    return z