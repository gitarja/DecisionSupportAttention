import unittest
from NNModels.RGBModel import RGBModel, DomainDiscriminator
from Utils.CustomLoss import CentroidTriplet
import tensorflow as tf
class MyTestCase(unittest.TestCase):
    def test_sketch_model(self):
        x = tf.random.normal((5*4, 288, 144, 3))
        model = RGBModel(z_dim=64, model="face")
        model_dom = DomainDiscriminator()
        compute_loss = CentroidTriplet(n_shots=4)
        z = model(x)
        y = model_dom(z)
        loss = compute_loss(z)

        self.assertEqual(z.shape, (5, 64))
        self.assertEqual(y.shape, (5, 1))


if __name__ == '__main__':
    unittest.main()
