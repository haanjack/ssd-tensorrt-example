#!/usr/bin/python3

import os
import sys
import time
import argparse

import cv2
import numpy as np
from PIL import Image
import tensorrt as trt

from utils import model
from utils.config import PathInfo, DetectionModel
from utils.inference import TRTInference
from utils.dataset import coco

## Remote Debugging
import ptvsd

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run object detection inference on input image.')
    parser.add_argument('--input_image_dir', default='/coco/test2017', 
        help='directory of image files to run inference on')
    parser.add_argument('--input_image_path', default='',
        help='an image file to run inference on')
    parser.add_argument('-p', '--precision', choices=['FLOAT', 'HALF', 'INT8'], default='FLOAT',
        help='desired precision to build an engine with')
    parser.add_argument('-b', '--max_batch_size', type=int, default=1,
        help='max TensorRT engine batch size')
    # parser.add_argument('-w', '--workspace_dir',
    #     help='sample workspace directory')
    # parser.add_argument('-fc', '--flatten_concat',
    #     help='path of built FlattenConcat plugin')
    parser.add_argument('-d', '--calib_dataset', default='/VOCdevkit/VOC2007/JPEGImages',
        help='path to the calibration dataset')
    parser.add_argument('-c', '--camera', default=False,
        help='if True, will run webcam application')
    parser.add_argument('--remote_debug', action='store_true', default=False,
        help='pend for the remote debugger attatch.')
    parser.add_argument('--remote_debug_port', default=5678,
        help='set remote debug port (default: 5678)')
    parser.add_argument('--debug', action='store_true', default=False,
        help='print debug messages')
    parser.add_argument('--num_iter', default=10,
        help='number of testing iteration')
    args = parser.parse_args()

    trt_precision_to_datatype = {
        'FLOAT': trt.DataType.FLOAT,
        'HALF': trt.DataType.HALF,
        'INT8': trt.DataType.INT8
    }
    args.trt_engine_datatype = trt_precision_to_datatype[args.precision]
    args.num_iter = int(args.num_iter)

    args.model_name = 'ssd_inception_v2_coco_2017_11_17'
    args.model_base_path = PathInfo.get_frozen_model_path(args.model_name)
    args.model_uff_path = PathInfo.get_uff_model_path(args.model_name)
    args.trt_engine_path = PathInfo.get_engine_path(args.model_name, args.trt_engine_datatype, args.max_batch_size)

    return args

def main():

    args = parse_arguments()

    if args.remote_debug is True:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=('0.0.0.0', int(args.remote_debug_port)), redirect_output=True)
        ptvsd.wait_for_attach()

    # Fetch .uff model path, convert from .pb
    # if needed, using prepare_ssd_model
    if not os.path.exists(args.model_uff_path):
        model.prepare_ssd_model(args.model_name)

    ## import frozen model then convert to uff
    if not os.path.exists(args.model_uff_path):
        model.model_to_uff(args.model_base_path, args.model_uff_path)

    ## build tensorrt engine and save as file
    runtime = TRTInference(args.trt_engine_path,
        args.model_uff_path,
        trt_engine_datatype=args.trt_engine_datatype,
        calib_dataset=args.calib_dataset,
        batch_size=args.max_batch_size)

    ## initialize test dataset loader
    input_shape = (DetectionModel.get_input_height(), DetectionModel.get_input_width(), DetectionModel.get_input_channels())
    test_loader = coco(directory=args.input_image_dir, batch_size=args.max_batch_size, target_shape=input_shape, shuffle=False)

    if args.camera is False and args.input_image_path is not '':
        print("Running input image:", args.input_image_path)
        if os.path.exists(args.input_image_path) is False:
            raise Exception("Input file is not exist!!")

        inference_start_time = time.time()

        for i in range(args.num_iter):
            image_np = test_loader.load_image(args.input_image_path)
            runtime.infer(image_np)

        # output inference time
        print("TensorRT inference time: {} ms".format(
            int(round((time.time() - inference_start_time) * 1000 / args.num_iter))))

    elif args.camera is False and args.input_image_path is '':
        print("Running inference in ", args.input_image_dir)
        if os.path.exists(args.input_image_dir) is False:
            raise Exception("Input directory is not exist!!")

        inference_start_time = time.time()

        for i in range(args.num_iter):
            ls_image_np = test_loader.next_batch()
            runtime.infer_batch(ls_image_np, len(ls_image_np))

        print("TensorRT inference time: {} ms".format(
            int(round((time.time() - inference_start_time) * 1000 / args.num_iter))))

    else:
        cam = cv2.VideoCapture(0)

        # loop for running inference on frames from teh webcam
        while True:
            ret, image_np = cam.read()

            detection_out, keep_count_out = runtime.infer_webcam(image_np)

            image_pil = Image.fromarray(image_np)
            prediction_fields = len(DetectionModel.output_layout)
            for det in range(int(keep_count_out[0])):
                DetectionModel.analyze_prediction(detection_out, det * prediction_fields, image_pil)
            final_image = np.asarray(image_pil)

            # display output
            cv2.imshow('object detection', final_image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    main()
    