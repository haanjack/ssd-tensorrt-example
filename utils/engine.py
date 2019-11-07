
import os
import sys

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from utils.config import DetectionModel
from utils.calibrator import Calibrator
from utils.common import HostDeviceMem

class TRTEngine(object):
    def __init__(self):
        pass

    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    max_batch_size = 1
    engine = None

    # Current NMS implementation in TRT only supports DataType.FLOAT but
    # it may change in the future, which could brake this sample here
    # when using lower precision [e.g. NMS output would not be np.float32
    # anymore, even though this is assumed in binding_to_type]
    binding_to_type = {"Input": np.float32, "NMS": np.float32, "NMS_1": np.int32}

    def build(self, uff_model_path, trt_logger, trt_engine_datatype=trt.DataType.FLOAT, calib_dataset=None, batch_size=1):
        print('Building TensorRT engine. This may take few minutes.')
        print('uff_model_path:', uff_model_path)
        print('trt_engine_datatype:', trt_engine_datatype)
        with trt.Builder(trt_logger) as builder, builder.create_network() as network, trt.UffParser() as parser:
            builder.max_workspace_size = 2 << 30    # 1GiB
            self.max_batch_size = batch_size
            builder.max_batch_size = batch_size
            if trt_engine_datatype == trt.DataType.HALF:
                builder.fp16_mode = True
            elif trt_engine_datatype == trt.DataType.INT8:
                builder.fp16_mode = True
                builder.int8_mode = True
                builder.int8_calibrator = Calibrator(data_dir=calib_dataset, cache_file='INT8CacheFile')

            parser.register_input(DetectionModel.input_name, DetectionModel.input_shape)
            parser.register_output(DetectionModel.output_name) # "MarkOutput_0" ???
            parser.parse(uff_model_path, network)

            self.engine = builder.build_cuda_engine(network)

    def load(self, trt_runtime, engine_path):
        """loads TensorRT Engine (.plan)
        """
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        self.engine = trt_runtime.deserialize_cuda_engine(engine_data)

        print("loaded TensorRT engine: {}".format(engine_path))
        print("max_batch_size: {}".format(self.engine.max_batch_size))
        self.max_batch_size = self.engine.max_batch_size
        return self.engine

    def save(self, engine_dest_path):
        print('Engine save:', engine_dest_path)
        buf = self.engine.serialize()
        with open(engine_dest_path, 'wb') as f:
            f.write(buf)

    def allocate_buffers(self):
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = self.binding_to_type[str(binding)]
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
        return self.inputs, self.outputs, self.bindings, self.stream

    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference(self, context, batch_size=1):
        if batch_size > self.max_batch_size:
            raise Exception('Got too large batch size request than TensorRT engine configured')

        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in self.outputs]

    
