
import tensorflow as tf


def make_sequence_example(length, source, target):
    # The object we return
    example = tf.train.SequenceExample()
    # A non-sequential feature of our example
    example.context.feature["length"].int64_list.value.append(length)
    # Feature lists for the two sequential features of our example
    fl_source = example.feature_lists.feature_list["source"]
    fl_target = example.feature_lists.feature_list["target"]
    for source_char, target_char in zip(source, target):
        fl_source.feature.add().int64_list.value.append(source_char)
        fl_target.feature.add().int64_list.value.append(target_char)
    return example


def parse_sequence_example(serialized_example):
    context, sequence = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features={
             "length": tf.FixedLenFeature([], dtype=tf.int64)
        },
        sequence_features={
             "source": tf.FixedLenSequenceFeature([], dtype=tf.int64),
             "target": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }
    )

    return (context['length'], sequence['source'], sequence['target'])
