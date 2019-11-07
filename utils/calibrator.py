
import os
import sys

import numpy as np
from PIL import Image

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from utils.dataset import voc

from utils.config import DetectionModel

class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_dir, cache_file, batch_size=8):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.num_calib_images = 128  # recommended under 200
        self.batch_size = batch_size # calibration batch size
        self.batch_shape = (self.batch_size, DetectionModel.get_input_channels(), DetectionModel.get_input_height(), DetectionModel.get_input_width())
        self.cache_file = cache_file

        # list-up the calibration dataset files
        calib_images = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.calib_images = np.random.choice(calib_images, self.num_calib_images)
        self.counter = 0

        # allocate device memory for calibration data
        self.device_input = cuda.mem_alloc(trt.volume(self.batch_shape) * trt.float32.itemsize)

        # create a generator that will give us batches. We can use next() to iterate over teh result.
        def load_batches():
            while self.counter < self.num_calib_images:
                data, num_read = self.read_image_batch()
                self.counter += num_read
                yield data
        self.batches = load_batches()

    def get_batch_size(self):
        return self.batch_size

    def read_image_batch(self):
        num_read = 0
        host_buffer = np.zeros(shape=[self.batch_size] + list(DetectionModel.input_shape))
        for idx in range(min(self.batch_size, self.num_calib_images - self.counter)):
            image = Image.open(self.calib_images[idx + self.counter])

            image_resized = image.resize(
                size=(DetectionModel.get_input_height(), DetectionModel.get_input_width()),
                resample=Image.BILINEAR
            )
            image_np = self._load_image_into_numpy_array(image_resized)
            # HWC -> CHW
            image_np = image_np.transpose((2, 0, 1))
            # Normalize to [-1.0, 1.0] interval (expected by model)
            image_np = (2.0 / 255.0) * image_np - 1.0

            host_buffer[idx,:,:,:] = image_np
            num_read += 1

        return host_buffer, num_read

    def get_batch(self, names):
        try:
            data = np.ascontiguousarray(next(self.batches), np.float32)

            print("Calibration batch {}/{}".format(self.counter, self.num_calib_images))

            # copy to device memory
            # data = ls_image_np.astype(np.float32).ravel().tobytes()
            cuda.memcpy_htod(self.device_input, data)

            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        print("writing calibration file")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def _load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image).reshape(
            (im_height, im_width, 3)
        ).astype(np.uint8)