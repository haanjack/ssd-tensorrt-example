import os
import numpy as np

import tensorrt as trt

from utils.dataset import coco
import utils.annotation as annotation

class PathInfo(object):
    base_model_path = "models"
    uff_model_path = "uff"
    trt_engine_path = "engine"
    
    @staticmethod
    def get_frozen_model_path(model_name):
        return os.path.join(PathInfo.base_model_path, model_name)
        
    @staticmethod
    def get_pb_model_path(model_name):
        return os.path.join(PathInfo.get_frozen_model_path(model_name), 'frozen_inference_graph.pb')

    @staticmethod
    def get_uff_model_path(model_name):
        return os.path.join(
            PathInfo.get_frozen_model_path(model_name),
            'frozen_inference_graph.uff'
        )

    @staticmethod
    def get_engine_path(model_name, inference_type=trt.DataType.FLOAT, max_batch_size=1):
        inference_type_to_str = {
            trt.DataType.FLOAT: 'FLOAT',
            trt.DataType.HALF: 'HALF',
            trt.DataType.INT32: 'INT32',
            trt.DataType.INT8: 'INT8'
        }

        # ssd_xx/engine/float
        return os.path.join(
            PathInfo.get_frozen_model_path(model_name),
            'engine',
            '{}_b{}.plan'.format(inference_type_to_str[inference_type], max_batch_size))

    @staticmethod
    def get_model_url(model_name):
        return 'http://download.tensorflow.org/models/object_detection/{}.tar.gz'.format(model_name)

class DetectionModel(object):
    input_name = "Input"
    input_shape = (3, 300, 300) # CHW
    
    output_name = "NMS"
    output_layout = {
        "image_id": 0,
        "label": 1,
        "confidence": 2,
        "xmin": 3,
        "ymin": 4,
        "xmax": 5,
        "ymax": 6
    }

    visualization_threadshold = 0.5

    @staticmethod
    def get_input_channels():
        return DetectionModel.input_shape[0]
    
    @staticmethod
    def get_input_height():
        return DetectionModel.input_shape[1]
    
    @staticmethod
    def get_input_width():
        return DetectionModel.input_shape[2]

    @staticmethod
    def _fetch_prediction_field(field_name, detection_out, pred_start_idx):
        return detection_out[pred_start_idx + DetectionModel.output_layout[field_name]]
    
    @staticmethod
    def analyze_prediction(detection_out, pred_start_idx, image_pil):
        image_id = int(DetectionModel._fetch_prediction_field("image_id", detection_out, pred_start_idx))
        label    = int(DetectionModel._fetch_prediction_field("label", detection_out, pred_start_idx))
        confidence = DetectionModel._fetch_prediction_field("confidence", detection_out, pred_start_idx)
        xmin = DetectionModel._fetch_prediction_field("xmin", detection_out, pred_start_idx)
        ymin = DetectionModel._fetch_prediction_field("ymin", detection_out, pred_start_idx)
        xmax = DetectionModel._fetch_prediction_field("xmax", detection_out, pred_start_idx)
        ymax = DetectionModel._fetch_prediction_field("ymax", detection_out, pred_start_idx)

        if confidence > DetectionModel.visualization_threadshold:
            class_name = coco.ls_coco_class[label]
            confidence_percentage = "{0:.0%}".format(confidence)
            print("Detected {} with confidence {}".format(
                class_name, confidence_percentage))
            annotation.draw_bounding_boxes_on_image(
                image_pil, np.array([[ymin, xmin, ymax, xmax]]),
                display_str_list=["{}: {}".format(
                    class_name, confidence_percentage)],
                color=coco.ls_coco_color[label]
            )
            

