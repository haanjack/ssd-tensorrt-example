import os
import random

import numpy as np
from PIL import Image

class dataset(object):
    def __init__(self, ls_class, directory='', batch_size=1, target_shape=(300, 300, 3), shuffle=False):
        self.ls_class = ls_class
        self.set_classes = set(ls_class)
        self.batch_size = batch_size
        self.image_shape = target_shape

        # list-up the files
        if os.path.exists(directory) is False:
            raise Exception('Target image directory is not exist!!')
        self.image_dir = directory

        self.ls_image_path = []
        for f in os.listdir(self.image_dir):
            self.ls_image_path.append(os.path.join(self.image_dir, f))
        if len(self.ls_image_path) is 0:
            raise Exception('Target directory is empty')

        if shuffle is True:
            random.shuffle(self.ls_image_path)

        # shared buffer initialization
        self.ls_image_np = np.zeros((batch_size, target_shape[0]*target_shape[1]*target_shape[2]), dtype=np.float32)
        self.index = 0

        # initialize label info
        self.dict_label_id = {
            cls_name: idx for idx, cls_name in enumerate(self.ls_class)
        }
        self.ls_class_color = np.random.uniform(0, 255, size=(len(self.ls_class), 3)).astype(np.uint8)

    def next_batch(self):
        for idx, image_path in enumerate(self.ls_image_path[self.index*self.batch_size:(self.index+1)*self.batch_size]):
            self.ls_image_np[idx] = self.load_image(image_path, self.image_shape)

        return self.ls_image_np

    @classmethod
    def load_image(cls, image_path, image_shape=(300, 300, 3)):
        # can be switched with dali ?
        image = Image.open(image_path)
        if (image.mode != 'RGB'):
            image = image.convert('RGB')
        image_resized = image.resize(
            size=(image_shape[0], image_shape[1]),
            resample=(Image.BILINEAR)
        )
        image_np = cls._load_image_into_numpy_array(image_resized, image_shape[2])
        # HWC -> CHW (To comfort with TF weights)
        image_np = image_np.transpose((2, 0, 1))
        # Normalize to [-1.0, 1.0] interval (expected by model)
        image_np = (2.0 / 255.0) * image_np - 1.0
        image_np = image_np.ravel()

        return image_np

    @classmethod
    def _load_image_into_numpy_array(cls, image, channel_size):
        """ load image with HWC type """
        (im_width, im_height) = image.size
        return np.array(image).reshape(
            (im_height, im_width, channel_size)
        ).astype(np.uint8)

    def get_label_color(self, label):
        """Returns color corresponding to give COCO label, or Name.

        Args:
            label (str): object label
        Returns:
            np.array: RGB color described in 3-element np.array
        """

        if not self.is_labeled(label):
            return None
        else:
            return self.ls_class_color[self.dict_label_id[label]]

    def is_labeled(self, label):
        return label in self.set_classes


class coco(dataset):
    ls_coco_class = [
            'unlabeled',
            'person',
            'bicycle',
            'car',
            'motorcycle',
            'airplane',
            'bus',
            'train',
            'truck',
            'boat',
            'traffic light',
            'fire hydrant',
            'street sign',
            'stop sign',
            'parking meter',
            'bench',
            'bird',
            'cat',
            'dog',
            'horse',
            'sheep',
            'cow',
            'elephant',
            'bear',
            'zebra',
            'giraffe',
            'hat',
            'backpack',
            'umbrella',
            'shoe',
            'eye glasses',
            'handbag',
            'tie',
            'suitcase',
            'frisbee',
            'skis',
            'snowboard',
            'sports ball',
            'kite',
            'baseball bat',
            'baseball glove',
            'skateboard',
            'surfboard',
            'tennis racket',
            'bottle',
            'plate',
            'wine glass',
            'cup',
            'fork',
            'knife',
            'spoon',
            'bowl',
            'banana',
            'apple',
            'sandwich',
            'orange',
            'broccoli',
            'carrot',
            'hot dog',
            'pizza',
            'donut',
            'cake',
            'chair',
            'couch',
            'potted plant',
            'bed',
            'mirror',
            'dining table',
            'window',
            'desk',
            'toilet',
            'door',
            'tv',
            'laptop',
            'mouse',
            'remote',
            'keyboard',
            'cell phone',
            'microwave',
            'oven',
            'toaster',
            'sink',
            'refrigerator',
            'blender',
            'book',
            'clock',
            'vase',
            'scissors',
            'teddy bear',
            'hair drier',
            'toothbrush',
        ]
    
    def __init__(self, directory='', batch_size=1, target_shape=(300, 300, 3), shuffle=False):
        super(coco, self).__init__(self.ls_coco_class, directory=directory, batch_size=batch_size, target_shape=target_shape, shuffle=shuffle)

class voc(dataset):
    ls_voc_class = [
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
    ]

    def __init__(self, directory='', batch_size=1, target_shape=(300, 300, 3), shuffle=False):
        super(voc, self).__init__(self.ls_voc_class, directory=directory, batch_size=batch_size, target_shap=target_shape, shuffle=shuffle)

    @staticmethod
    def convert_coco_to_voc(label):
        """Converts COCO class name to VOC class name, if possible.

        COCO classes are a superset of VOC classes, but
        some classes have different names (e.g. airplane
        in COCO is aeroplane in VOC). This function gets
        COCO label and converts it to VOC label,
        if conversion is needed.

        Args:
            label (str): COCO label
        Returns:
            str: VOC label corresponding to given label if such exists,
                otherwise returns original label
        """
        COCO_VOC_DICT = {
            'airplane': 'aeroplane',
            'motorcycle': 'motorbike',
            'dining table': 'diningtable',
            'potted plant': 'pottedplant',
            'couch': 'sofa',
            'tv': 'tvmonitor'
        }
        if label in COCO_VOC_DICT:
            return COCO_VOC_DICT[label]
        else:
            return label

    @staticmethod
    def coco_label_to_voc_label(label):
        """Returns VOC label corresponding to given COCO label.

        COCO classes are superset of VOC classes, this function
        returns label corresponding to given COCO class label
        or None if such label doesn't exist.

        Args:
            label (str): COCO class label
        Returns:
            str: VOC label corresponding to given label or None
        """
        label = voc.convert_coco_to_voc(label)
        if label in voc.set_classes:
            return label
        else:
            return None
    
