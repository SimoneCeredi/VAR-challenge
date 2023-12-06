import matplotlib.pyplot as plt
import numpy as np
import math

def plot_images_with_tlbr(images, boxes, class_ids, class_labels, image_per_row=4, show_labels=True,
                          confidences=None):
  class_colors = plt.cm.hsv(np.linspace(0, 1, len(class_labels) + 1)).tolist()
  image_count = len(images)
  row_count = math.ceil(image_count / image_per_row)
  col_count = image_per_row

  _, axs = plt.subplots(nrows=row_count, ncols=col_count, figsize=(18, 4 * row_count), squeeze=False)
  for r in range(row_count):
    for c in range(col_count):
      axs[r, c].axis('off')

  for i in range(image_count):
    r = i // image_per_row
    c = i % image_per_row

    axs[r, c].imshow(images[i])
    for box_idx in range(len(boxes[i])):
      box = boxes[i][box_idx]
      class_idx = class_ids[i][box_idx]
      color = class_colors[class_idx]
      xmin = box[0]
      ymin = box[1]
      xmax = box[2]
      ymax = box[3]
      w = xmax - xmin
      h = ymax - ymin
      axs[r, c].add_patch(plt.Rectangle((xmin, ymin), w, h, color=color, fill=False, linewidth=2))
      if show_labels:
        label = '{}'.format(class_labels[class_idx])
        if confidences is not None:
          label += ' {:.2f}'.format(confidences[i][box_idx])
        axs[r, c].text(xmin, ymin, label, size='large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
  plt.show()
