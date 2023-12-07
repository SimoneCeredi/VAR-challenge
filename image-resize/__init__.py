import os

import matplotlib.pyplot as plt
import numpy as np
import cv2

from yolo_transformations import load_dataset_info, load_tlbr_dataset_info
from bounding_boxes import from_tlwh_to_tlbr
from plots import plot_images_with_tlbr


def load_images_paths_and_labels_paths_from_folder(folder):
  images_path = os.path.join(folder, 'images')
  labels_path = os.path.join(folder, 'labels')
  images = sorted(
    [os.path.join(images_path, filename) for filename in os.listdir(images_path)]
  )
  labels = sorted(
    [os.path.join(labels_path, filename) for filename in os.listdir(labels_path)]
  )
  return images, labels


def resize_img(img_path, input_size):
  my_image = plt.imread(img_path)

  if max(my_image.shape[0], my_image.shape[1]) > input_size:
    if my_image.shape[0] >= my_image.shape[1]:
      height = input_size
      width = int(my_image.shape[1] * input_size / my_image.shape[0])
    else:
      width = input_size
      height = int(my_image.shape[0] * input_size / my_image.shape[1])
    my_image = cv2.resize(my_image, (width, height), interpolation=cv2.INTER_LINEAR)

  plt.imsave(img_path, my_image)


def pad_img(img_path, input_size):
  my_image = plt.imread(img_path)

  # Punto 5
  pad_width = input_size - my_image.shape[0]
  pad_height = input_size - my_image.shape[1]
  my_padded_image = np.pad(my_image,
                           ((0, pad_width), (0, pad_height), (0, 0)),
                           'constant')
  plt.imsave(img_path, my_padded_image)


def save_bboxes(bboxes, label_path):
  with open(label_path, 'w') as file:
    for bbox in bboxes:
      file.write('1 ' + ' '.join(map(str, bbox)) + '\n')


class_labels = ['none', 'logo']
input_size = 512

images, labels = load_images_paths_and_labels_paths_from_folder('data')

for img_path in images:
  resize_img(img_path, input_size)

img_paths, yolov8_boxes, id_list = load_dataset_info(labels, images)

tlbr_boxes = from_tlwh_to_tlbr(yolov8_boxes, img_paths)

for i in range(len(labels)):
  save_bboxes(tlbr_boxes[i], labels[i])

for img_path in images:
  pad_img(img_path, input_size)



# plot image 0
plt.figure(figsize=(10, 10))
plt.imshow(plt.imread(img_paths[0]))
plt.show()
# plot_images_with_xywh_bounding_boxes([plt.imread(img_path) for img_path in img_paths], yolov8_boxes, id_list,
#                                      class_labels, image_per_row=4, show_labels=True)
plot_images_with_tlbr([plt.imread(img_path) for img_path in img_paths], tlbr_boxes, id_list,
                      class_labels, image_per_row=4, show_labels=True)

images, labels = load_images_paths_and_labels_paths_from_folder('data')
img_paths, yolov8_boxes, id_list = load_tlbr_dataset_info(labels, images)

plt.figure(figsize=(10, 10))
plt.imshow(plt.imread(img_paths[0]))
plt.show()
# plot_images_with_xywh_bounding_boxes([plt.imread(img_path) for img_path in img_paths], yolov8_boxes, id_list,
#                                      class_labels, image_per_row=4, show_labels=True)
plot_images_with_tlbr([plt.imread(img_path) for img_path in img_paths], tlbr_boxes, id_list,
                      class_labels, image_per_row=4, show_labels=True)