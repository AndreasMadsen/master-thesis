
import tensorflow as tf

from code.model.abstract import Model


class DummyModel(Model):
    def __init__(self, dataset, translated,
                 save_dir='asset/dummy-model',
                 **kwargs):
        self._translated = translated
        super().__init__(dataset, save_dir=save_dir, **kwargs)

    def inference_model(self, x, reuse=False):
        observations = int(self.dataset.target.get_shape()[0])

        translated = tf.convert_to_tensor(self._translated)
        translated = tf.train.slice_input_producer(
            [translated], shuffle=False
        )[0]
        translated = tf.train.batch(
            [translated], observations,
            name='inference',
            num_threads=1,
            capacity=observations,
            allow_smaller_final_batch=False
        )

        return translated
