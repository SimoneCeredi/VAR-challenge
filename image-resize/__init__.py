import os

import matplotlib.pyplot as plt

from yolo_transformations import load_dataset_info
from bounding_boxes import from_rel_xywh_to_xywh
from plots import plot_images_with_xywh_bounding_boxes


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


class_labels = ['logo', 'none']

images, labels = load_images_paths_and_labels_paths_from_folder('data')
print(images[0], labels[0])

img_paths, yolov8_boxes, id_list = load_dataset_info(labels, images)

print(img_paths[0], yolov8_boxes[0], id_list[0])

# train_xywh_box_list=from_rel_xywh_to_xywh(train_yolov8_box_list,original_image_size)
yolov8_boxes = from_rel_xywh_to_xywh(yolov8_boxes, images)

print(img_paths[0], yolov8_boxes[0], id_list[0])
# plot image 0
plt.figure(figsize=(10, 10))
plt.imshow(plt.imread(img_paths[0]))
plt.show()
plot_images_with_xywh_bounding_boxes([plt.imread(img_path) for img_path in img_paths], yolov8_boxes, id_list,
                                     class_labels, image_per_row=4, show_labels=True)
