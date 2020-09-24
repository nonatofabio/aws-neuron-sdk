#!/usr/bin/env python 

import shutil
import tensorflow as tf
import tensorflow.neuron as tfn


def no_fuse_condition(op):
    return any(op.name.startswith(pat) for pat in ['reshape', 'lambda_1/Cast', 'lambda_2/Cast', 'lambda_3/Cast'])

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['serve'], './yolo_v4_coco_saved_model')
    no_fuse_ops = [op.name for op in sess.graph.get_operations() if no_fuse_condition(op)]

shutil.rmtree('./yolo_v4_coco_saved_model_neuron', ignore_errors=True)

result = tfn.saved_model.compile(
                './yolo_v4_coco_saved_model', './yolo_v4_coco_saved_model_neuron',
                # we partition the graph before casting from float16 to float32, to help reduce the output tensor size by 1/2
                no_fuse_ops=no_fuse_ops,
                # to enforce trivial compilable subgraphs to run on CPU
                minimum_segment_size=100,
                batch_size=1,
                dynamic_batch_size=True,
)

print(result)
