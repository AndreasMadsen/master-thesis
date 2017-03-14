
import tensorflow as tf


def shuffle_tensor_index(input_queue, dequeue_many=32, **kwargs):
    dequeue_op = input_queue.dequeue_many(dequeue_many)
    dtypes = [dequeue_op.dtype]
    shapes = [dequeue_op.get_shape()[1:]]

    shuffle_queue = tf.RandomShuffleQueue(
        dtypes=dtypes, shapes=shapes,
        **kwargs)
    shuffle_enqueue = shuffle_queue.enqueue_many([dequeue_op])
    tf.train.add_queue_runner(
        tf.train.QueueRunner(shuffle_queue, [shuffle_enqueue])
    )
    return shuffle_queue.dequeue()
