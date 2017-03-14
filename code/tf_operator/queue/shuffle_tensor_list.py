
import tensorflow as tf


def shuffle_tensor_list(input_tensors, **kwargs):
    dtypes = [tensor.dtype for tensor in input_tensors]

    shuffle_queue = tf.RandomShuffleQueue(dtypes=dtypes, **kwargs)
    shuffle_enqueue = shuffle_queue.enqueue(input_tensors)
    tf.train.add_queue_runner(
        tf.train.QueueRunner(shuffle_queue, [shuffle_enqueue])
    )

    output_tensors = shuffle_queue.dequeue()
    for output_tensor, input_tensor in zip(output_tensors, input_tensors):
        output_tensor.set_shape(input_tensor.get_shape())

    return tuple(output_tensors)
