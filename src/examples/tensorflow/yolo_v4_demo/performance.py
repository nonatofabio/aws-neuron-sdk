#performance.py
import os
import argparse
import time
from concurrent import futures
import numpy as np
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help='SaveModel')
parser.add_argument('--num_session', type=int, default=1, help='Number of (parallel) tensorflow sessions')
parser.add_argument('--num_thread', type=int, default=8, help='Number of threads that work on each tensorflow session')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--sequence_length', type=int, default=64)
parser.add_argument('--latency_window_size', type=int, default=1000)
parser.add_argument('--throughput_time', type=int, default=300)
parser.add_argument('--throughput_interval', type=int, default=10)
args = parser.parse_args()

pred_list = [tf.contrib.predictor.from_saved_model(args.model_dir) for _ in range(args.num_session)]
pred_mp_list = []
for pred in pred_list:
    pred_mp_list.extend(pred for _ in range(args.num_thread))

image_path = './val2017/000000581781.jpg'
with open(image_path, 'rb') as f:
    feed = f.read()
    f.close()

feed_batch = []
for _ in range(args.batch_size):
    feed_batch.append(feed)

model_feed_dict = {}
for name, tensor in pred_list[0].feed_tensors.items():
    shape = tensor.shape.as_list()
    model_feed_dict[name] = feed_batch


live = True
num_infer = 0
latency_list = []
def one_thread(pred, model_feed_dict):
    global latency_list
    global num_infer
    global live
    while True:
        start = time.time()
        result = pred(model_feed_dict)
        latency = time.time() - start
        latency_list.append(latency)
        num_infer += args.batch_size
        if not live:
            break
def current_performance():
    last_num_infer = num_infer
    for _ in range(args.throughput_time // args.throughput_interval):
        current_num_infer = num_infer
        throughput = (current_num_infer - last_num_infer) / args.throughput_interval
        p50 = 0.0
        p90 = 0.0
        if latency_list:
            p50 = np.percentile(latency_list[-args.latency_window_size:], 50)
            p90 = np.percentile(latency_list[-args.latency_window_size:], 90)
        print('pid {}: current infers {} throughput {}, latency p50={:.3f} p90={:.3f}'.format(os.getpid(), current_num_infer,throughput, p50, p90))
        last_num_infer = current_num_infer
        time.sleep(args.throughput_interval)
    global live
    live = False
executor = futures.ThreadPoolExecutor(max_workers=len(pred_mp_list)+1)
executor.submit(current_performance)
for pred in pred_mp_list:
    executor.submit(one_thread, pred, model_feed_dict)