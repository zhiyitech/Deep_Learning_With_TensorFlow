"""
Usage:
  python video_generator.py

"""

import sys
import os
import numpy as np
import tensorflow as tf

sys.path.append("..")

from object_detection.utils import label_map_util, visualization_utils as vis_util  # todo: dependency to custom_model_object_detection official model

# frozen inference输出的文件目录
MODEL_NAME = '/data/home/goosegu/video/models/research/models_nba/export'  # todo: use the folder with the frozen inference graph
# frozen inference输出的文件之一
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# nba_label_map.pbtxt配置的目录，用来给出目标名称
PATH_TO_LABELS = os.path.join('/data/home/goosegu/video/models/research/data/nba', 'nba_label_map.pbtxt') # todo: use the label map file
# 类别数量，与模型训练config配置里保持一致，这里只有一个类-----leonard
NUM_CLASSES = 1 # todo change to number of classes
IMAGE_SIZE = (12, 8) # todo change to image size selected

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# tf提供的目标检测接口
def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # 实际的检测运行步骤
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # 使检测结果可视化，对视频帧画检测框
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4)

    return image_np

# 读取图片
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# 读取模型
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# 使用moviepy编辑视频片段
from moviepy.editor import VideoFileClip
def process_image(image):
    #return image
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            #针对视频帧，进行目标检测
            image_process = detect_objects(image, sess, detection_graph)
            #返回检测结果帧图像
            return image_process
          
# 输出视频名称，这个视频会包含leonard检测的结果
white_output = 'OUTPUT.mp4' 
# 待检测的视频，这里使用了生成训练数据的视频。
clip1 = VideoFileClip("../data/leonard.mp4") 
# 按照时间起止点（t_start，t_end）截取希望处理视频的片段，并调用process_image按帧进行目标检测
white_clip = clip1.fl_image(process_image).subclip(t_start=(00,00.00),t_end=(00,00.30)) 
# 写入检测结果
white_clip.write_videofile(white_output, audio=False)
