def parse_yolov8_annotation(txt_file):
  yolov8_boxes = []
  class_ids = []
  with open(txt_file) as file:
    for line in file:
      splitted_line = line.split()
      class_ids.append(int(splitted_line[0]))
      rcx = float(splitted_line[1])
      rcy = float(splitted_line[2])
      rw = float(splitted_line[3])
      rh = float(splitted_line[4])
      rxmin = rcx - rw / 2
      rymin = rcy - rh / 2
      yolov8_boxes.append([rxmin, rymin, rw, rh])  # rel_xywh

  return yolov8_boxes, class_ids


def load_dataset_info(txt_file_list, images):
  image_yolov8_box_list = []
  image_class_id_list = []
  for txt_file in txt_file_list:
    yolov8_boxes, class_ids = parse_yolov8_annotation(txt_file)
    image_yolov8_box_list.append(yolov8_boxes)
    image_class_id_list.append(class_ids)

  return images, image_yolov8_box_list, image_class_id_list
