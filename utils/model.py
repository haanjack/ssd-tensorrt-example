import os
import sys
import requests
import tarfile

import graphsurgeon as gs
import uff

from utils.config import PathInfo, DetectionModel
import utils.parser as ModelParser

def model_to_uff(model_path, uff_model_path):
    print("model_path:", model_path)
    dynamic_graph = gs.DynamicGraph(model_path)
    dynamic_graph = ModelParser.convert_unsupported_nodes_to_plugins(dynamic_graph)

    if os.path.exists(uff_model_path) is False:
        uff.from_tensorflow(
            dynamic_graph.as_graph_def(),
            [DetectionModel.output_name],
            output_filename=uff_model_path,
            text=True
        )

# Model download functionality

def download_file(model_url, model_local_path):
    """Downloads file from supplied URL and puts it into supplied directory.

    Args:
        file_url (str): URL with file to download
        file_dest_path (str): path to save downloaded file in
        silent (bool): if False, writes progress messages to stdout
    """
    model_file_path = os.path.join(model_local_path, 'downloaded_model.tar.gz')
    with open(model_file_path, "wb") as model_file:
        print("Downloading {}".format(model_url))
        response = requests.get(model_url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:
            model_file.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                model_file.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()

    return model_file_path

def download_model(model_name="ssd_inception_v2_coco_2017_11_17"):
    """Downloads model_name from Tensorflow model zoo.

    Args:
        model_name (str): chosen object detection model
        silent (bool): if False, writes progress messages to stdout
    """
    model_url = PathInfo.get_model_url(model_name)
    model_local_path = "models"

    if os.path.exists(model_local_path) is False:
        os.makedirs(model_local_path)
    model_file_path = download_file(model_url, model_local_path)

    # untar model file
    model_dir = model_local_path # os.path.join(model_local_path, model_name)
    with tarfile.open(model_file_path, "r:gz") as tar:
        tar.extractall(path=model_dir)
    os.remove(model_file_path)


def prepare_ssd_model(model_name="ssd_inception_v2_coco_2017_11_17"):
    """Downloads pretrained object detection model and converts it to UFF.

    The model is downloaded from Tensorflow object detection model zoo.
    Currently only ssd_inception_v2_coco_2017_11_17 model is supported
    due to model_to_uff() using logic specific to that network when converting.

    Args:
        model_name (str): chosen object detection model
        silent (bool): if False, writes progress messages to stdout
    """
    if model_name != "ssd_inception_v2_coco_2017_11_17":
        raise NotImplementedError(
            "model {} is not supported yet.".format(model_name))
    
    if not os.path.exists(PathInfo.get_frozen_model_path(model_name)):
        download_model(model_name)
    frozen_model_path = PathInfo.get_pb_model_path(model_name)
    uff_model_path = PathInfo.get_uff_model_path(model_name)
    model_to_uff(frozen_model_path, uff_model_path)