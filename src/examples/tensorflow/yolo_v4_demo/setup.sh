#!/bin/bash

echo "Set up data & model"
curl -LO http://images.cocodataset.org/zips/val2017.zip
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q val2017.zip
unzip annotations_trainval2017.zip

python3 yolo_v4_coco_saved_model.py ./yolo_v4_coco_saved_model
