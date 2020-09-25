#!/usr/bin/env python

import os
import json
import time
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from concurrent import futures

def cocoapi_eval(jsonfile,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000)):
    """
    Args:
        jsonfile: Evaluation json file, eg: bbox.json, mask.json.
        style: COCOeval style, can be `bbox` , `segm` and `proposal`.
        coco_gt: Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file: COCO annotations file.
        max_dets: COCO evaluation maxDets.
    """
    assert coco_gt is not None or anno_file is not None

    if coco_gt is None:
        coco_gt = COCO(anno_file)
    print("Start evaluate...")
    coco_dt = coco_gt.loadRes(jsonfile)
    if style == 'proposal':
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.useCats = 0
        coco_eval.params.maxDets = list(max_dets)
    else:
        coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


def bbox_eval(anno_file, bbox_list):
    coco_gt = COCO(anno_file)

    outfile = 'bbox_detections.json'
    print('Generating json file...')
    with open(outfile, 'w') as f:
        json.dump(bbox_list, f)

    map_stats = cocoapi_eval(outfile, 'bbox', coco_gt=coco_gt)
    return map_stats


def get_image_as_bytes(images, eval_pre_path, eval_batch_size=1):
    batch_im_id_list = []
    batch_im_name_list = []
    batch_img_bytes_list = []
    n = len(images)
    batch_im_id = []
    batch_im_name = []
    batch_img_bytes = []
    for i, im in enumerate(images):
        im_id = im['id']
        file_name = im['file_name']
        if i % eval_batch_size == 0 and i != 0:
            batch_im_id_list.append(batch_im_id)
            batch_im_name_list.append(batch_im_name)
            batch_img_bytes_list.append(batch_img_bytes)
            batch_im_id = []
            batch_im_name = []
            batch_img_bytes = []
        batch_im_id.append(im_id)
        batch_im_name.append(file_name)

        with open(os.path.join(eval_pre_path, file_name), 'rb') as f:
            batch_img_bytes.append(f.read())
    return batch_im_id_list, batch_im_name_list, batch_img_bytes_list


def analyze_bbox(results, batch_im_id, _clsid2catid):
    bbox_list = []
    k = 0
    for boxes, scores, classes in zip(results['boxes'], results['scores'], results['classes']):
        if boxes is not None:
            im_id = batch_im_id[k]
            n = len(boxes)
            for p in range(n):
                clsid = classes[p]
                score = scores[p]
                xmin, ymin, xmax, ymax = boxes[p]
                catid = (_clsid2catid[int(clsid)])
                w = xmax - xmin + 1
                h = ymax - ymin + 1

                bbox = [xmin, ymin, w, h]
                # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
                bbox = [round(float(x) * 10) / 10 for x in bbox]
                bbox_res = {
                    'image_id': im_id,
                    'category_id': catid,
                    'bbox': bbox,
                    'score': float(score),
                }
                bbox_list.append(bbox_res)
        k += 1
    return bbox_list

def evaluate(yolo_predictor, images, eval_pre_path, anno_file, eval_batch_size, _clsid2catid):
    batch_im_id_list, batch_im_name_list, batch_img_bytes_list = get_image_as_bytes(images, eval_pre_path,eval_batch_size)

    # warm up
    yolo_predictor({'image': np.array(batch_img_bytes_list[0], dtype=object)})

    with futures.ThreadPoolExecutor(4) as exe:
        fut_im_list = []
        fut_list = []
        start_time = time.time()
        for batch_im_id, batch_im_name, batch_img_bytes in zip(batch_im_id_list, batch_im_name_list, batch_img_bytes_list):
            if len(batch_img_bytes) != eval_batch_size:
                continue
            fut = exe.submit(yolo_predictor, {'image': np.array(batch_img_bytes, dtype=object)})
            fut_im_list.append((batch_im_id, batch_im_name))
            fut_list.append(fut)
        bbox_list = []
        count = 0
        for (batch_im_id, batch_im_name), fut in zip(fut_im_list, fut_list):
            results = fut.result()
            bbox_list.extend(analyze_bbox(results, batch_im_id, _clsid2catid))
            for _ in batch_im_id:
                count += 1
                if count % 100 == 0:
                    print('Test iter {}'.format(count))

        print('==================== Performance Measurement ====================')
        print('Finished inference on {} images in {} seconds'.format(len(images), time.time() - start_time))
        print('=================================================================')

    # start evaluation
    box_ap_stats = bbox_eval(anno_file, bbox_list)
    return box_ap_stats

def timed_evaluate(yolo_predictor, images, eval_pre_path, anno_file, eval_batch_size, _clsid2catid, bench_time=30):
    batch_im_id_list, batch_im_name_list, batch_img_bytes_list = get_image_as_bytes(images, eval_pre_path,eval_batch_size)

    # warm up
    yolo_predictor({'image': np.array(batch_img_bytes_list[0], dtype=object)})

    with futures.ThreadPoolExecutor(4) as exe:
        count = 0
        start_time = time.time()
        batch_time = start_time
        
        while time.time() - start_time < bench_time:
            fut_im_list = []
            fut_list = []
            fut = []
            for batch_im_id, batch_im_name, batch_img_bytes in zip(batch_im_id_list, batch_im_name_list, batch_img_bytes_list):
                if len(batch_img_bytes) != eval_batch_size:
                    continue
                fut = exe.submit(yolo_predictor, {'image': np.array(batch_img_bytes, dtype=object)})
                fut_im_list.append((batch_im_id, batch_im_name))
                fut_list.append(fut)
            
            bbox_list = []
            for (batch_im_id, batch_im_name), fut in zip(fut_im_list, fut_list):
                results = fut.result()
                bbox_list.extend(analyze_bbox(results, batch_im_id, _clsid2catid))
                for _ in batch_im_id:
                    count += 1
                    if count % 100 == 0:
                        print('Test iter {} time {:.3} seconds'.format(count,time.time()-batch_time))
                        batch_time = time.time()
                if time.time() - start_time > bench_time:
                    break

        print('==================== Performance Measurement ====================')
        print('Finished inference on {} images in {:.3} seconds'.format(count, time.time() - start_time))
        print('=================================================================')

    # start evaluation
    box_ap_stats = bbox_eval(anno_file, bbox_list)
    return box_ap_stats
