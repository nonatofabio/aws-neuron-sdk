#!/usr/bin/env python

import json
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from perf_utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path', help='Path to saved model')
args = parser.parse_args()

yolo_pred = tf.contrib.predictor.from_saved_model(args.path)

val_coco_root = './val2017'
val_annotate = './annotations/instances_val2017.json'
clsid2catid = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16,
               15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31,
               27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43,
               39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56,
               51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72,
               63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85,
               75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

eval_batch_size = 8

with open(val_annotate, 'r', encoding='utf-8') as f2:
    for line in f2:
        line = line.strip()
        dataset = json.loads(line)
        images = dataset['images']

box_ap = evaluate(yolo_pred, images, val_coco_root, val_annotate, eval_batch_size, clsid2catid)
