# SSD TensorRT Example

This repository is inspired by NVIDIA's [Object Detection TensorRT example](https://github.com/NVIDIA/object-detection-tensorrt-example) implementation. Which introduces how you can use a TensorRT uff_ssd example to apply webcam source.

In my implementation, I recovered the removed functions and re-organized the samples to have solid dataset loader and model definition code. Currently SSD is only supported, so the source can be changed along with that.

1. Build container
```
bash scripts/docker/build.sh
```

2. Launch container
*You might want to modify the predefined dataset mounting paths or GPU index*
```
bash scripts/docker/launch.sh
```

1. Prepare your dataset
```
## Download coco dataset using TensorFlow's coco dataset download and preprocess tool 
##    - https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/download_and_preprocess_mscoco.sh

# Download VOC2007 for calibration dataset
bash ./setup.sh
```

4. Run the engine build and inference
```
(in the container)
python object_detection.py
```

## Usage
```
usage: object_detection.py [-h] [--input_image_dir INPUT_IMAGE_DIR]
                           [--input_image_path INPUT_IMAGE_PATH]
                           [-p {FLOAT,HALF,INT8}] [-b MAX_BATCH_SIZE]
                           [-d CALIB_DATASET] [-c CAMERA] [--remote_debug]
                           [--remote_debug_port REMOTE_DEBUG_PORT] [--debug]
                           [--num_iter NUM_ITER]
```

* If you use input_image_path, the ```object_detection.py``` will inference for the target file with 1 batch.
* Remote debug works with ptvsd which works with vscode python remote debugging.

### Example
#### float32 / float16 infernce
```
python object_detection -p FLOAT/HALF -b 8 --num_iter 10
```
#### int8 inference
```
python object_detection -p INT8 -b 8 --num_iter 10
```

5. Future work
   [ ] validation task integration
   [ ] another object detection model
   [ ] profile result
   [ ] tensorboard integration