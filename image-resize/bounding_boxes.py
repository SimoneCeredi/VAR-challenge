import math

import numpy as np
from matplotlib import pyplot as plt



def from_rel_xywh_to_xywh(dataset_rel_xywh_bboxes, images_path):

  dataset_xywh_bboxes = []
  count = 0
  for image_bboxes in dataset_rel_xywh_bboxes:
    image_path = images_path[count]
    image_height, image_width = plt.imread(image_path).shape[:2]
    image_xywh_bboxes = []
    for image_bbox in image_bboxes:
      xmin = image_bbox[0] * image_width
      ymin = image_bbox[1] * image_height
      w = image_bbox[2] * image_width
      h = image_bbox[3] * image_height
      image_xywh_bboxes.append([xmin, ymin, w, h])
    dataset_xywh_bboxes.append(image_xywh_bboxes)
    count += 1

  return dataset_xywh_bboxes
