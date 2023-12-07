import math

import numpy as np
from matplotlib import pyplot as plt

def from_tlwh_to_tlbr(bboxes, images_path):
  tlbr_bboxes = []
  count = 0
  for bbox_list in bboxes:
    image = plt.imread(images_path[count])
    image_height, image_width = image.shape[:2]
    image_tlbr_bboxes = []
    for bbox in bbox_list:
      top_left_x = bbox[0] * image_width
      top_left_y = bbox[1] * image_height
      bottom_right_x = top_left_x + bbox[2] * image_width
      bottom_right_y = top_left_y + bbox[3] * image_height
      image_tlbr_bboxes.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])
    tlbr_bboxes.append(image_tlbr_bboxes)
    count += 1
  return tlbr_bboxes

