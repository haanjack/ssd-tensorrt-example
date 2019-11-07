
import os
import sys
from PIL import Image
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from utils.engine import TRTEngine
from utils.config import DetectionModel

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInference(object):
    """manages TensorRT objects for model inference."""
    def __init__(self, trt_engine_path, uff_model_path, trt_engine_datatype=trt.DataType.FLOAT, calib_dataset=None, batch_size=1):
        """ build TensorRT engine """

        # load all custom plugins shipepd with TensorRT
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        # initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # TensorRT engine placeholder
        trt_engine = TRTEngine()

        # display requested engine settings to stdout
        print("TensorRT inference engine settings:")
        print("  * Inference precision - {}".format(trt_engine_datatype))
        print("  * Max batch size - {}\n".format(batch_size))

        if not os.path.exists(os.path.dirname(trt_engine_path)):
            os.mkdir(os.path.dirname(trt_engine_path))

        # if engine is not cached, we need to build it
        if not os.path.exists(trt_engine_path):
            # this function uses supplied .uff file alongside with UffParser to build TensorRT engine.
            # For more detials, check implementation
            trt_engine.build(uff_model_path=uff_model_path, trt_logger=TRT_LOGGER, trt_engine_datatype=trt_engine_datatype, calib_dataset=calib_dataset, batch_size=batch_size)

            # save the engine to file
            trt_engine.save(trt_engine_path)
        else:
            print("loading cashed TensorRT engine from {}".format(trt_engine_path))
            trt_engine.load(self.trt_runtime, trt_engine_path)

        if trt_engine.engine is None:
            raise Exception('Error TensorRT engine is not created!!')

        # this allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = \
            trt_engine.allocate_buffers()

        # execution context is needed for inference
        self.context = trt_engine.engine.create_execution_context()

        # allocate memory for multiple usage [e.g. multiple batch inference]
        # input_volume = trt.volume(DetectionModel.input_shape)
        # self.ls_image = np.zeros((trt_engine.max_batch_size, input_volume))
        # print(":::: input_volume:", input_volume)
        # print(":::: input_shape:", DetectionModel.input_shape)

        self.trt_engine = trt_engine

    def infer(self, input_image_np):
        """Infers model on given image

        Args:
            image_path (str): image to run object detection model on
        """

        # copy it into appropriate place into memory
        np.copyto(self.inputs[0].host, input_image_np.ravel())

        # fetch output from the model (batch_size=1)
        [detection_out, keepCount_out] = self.trt_engine.do_inference(
            self.context)

        return detection_out, keepCount_out

    def infer_batch(self, ls_input_image_np, batch_size=1):
        """Infers model on batch of same size images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch 
                and fed into model
        """

        # Verify if the supplied batch size is not too big
        max_batch_size = self.trt_engine.max_batch_size
        if batch_size > max_batch_size:
            raise Exception("Requested batch size ({}) is larger than engine optimized ({})!".format(batch_size, max_batch_size))

        # load images into engine input
        np.copyto(self.inputs[0].host, ls_input_image_np.ravel())
        [detection_out, keepCount_out] = self.trt_engine.do_inference(
            self.context, batch_size=max_batch_size)

        return detection_out, keepCount_out

    def infer_webcam(self, image_np):
        """Infers model on given image

        Args:
            image_np (numpy array): image to run object detection model on
        """

        # load image into engine input
        image = self._load_image_webcam(image_np)
        np.copyto(self.inputs[0].host, image.ravel())

        inference_start_time = time.time()

        [detection_out, keepCount_out] = self.trt_engine.do_inference(self.context)

        print("TensorRT inference time: {} ms".format(int(round((time.time() - inference_start_time) * 1000))))

        return detection_out, keepCount_out

    def _load_image_webcam(self, arr):
        image = Image.fromarray(np.uint8(arr))
        model_input_width = DetectionModel.get_input_width()
        model_input_height= DetectionModel.get_input_height()
        image_resized = image.resized(
            size=(model_input_width, model_input_height),
            resample=Image.BILINEAR
        )
        img_np = self._load_image_into_numpy_array(image_resized)
        # HWC -> CHW
        img_np = img_np.transpose((2, 0, 1))
        # Normalize to [-1.0, 1.0] interval (expected by model)
        img_np = (2.0 / 255.0) * img_np - 1.0
        img_np = img_np.ravel()
        return img_np

    def _load_image_into_numpy_array(self, image):
        """ load image with HWC type """
        (im_width, im_height) = image.size
        return np.array(image).reshape(
            (im_height, im_width, DetectionModel.get_input_height())
        ).astype(np.uint8)

    

