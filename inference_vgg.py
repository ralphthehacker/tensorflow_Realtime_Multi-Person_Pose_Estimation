import tensorflow as tf
import cv2
import numpy as np
import os

from estimation.config import get_default_configuration
from estimation.coordinates import get_coordinates
from estimation.connections import get_connections
from estimation.estimators import estimate
from estimation.renderers import draw

from models import create_openpose_2branches_vgg as make_vgg
from tf_netbuilder_ext.extensions import register_tf_netbuilder_extensions
register_tf_netbuilder_extensions()

HEATMAP_INDEX = 3
PAF_INDEX = 2



class InferenceWrapper(object):
  def __init__(self):
    self.model = make_vgg(pretrained=True, training=False)

  def run_inference_single_image(self,image):
    """ Returns a tuple with (skeleton_image:CV2_img
                              , number_of_skeletons: int)"""
    img = cv2.imread(image)  # B,G,R order
    # resize to proper size
    img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
    input_img = img[np.newaxis, :, :, [2, 1, 0]]
    inputs = tf.convert_to_tensor(input_img)

    outputs = self.model.predict(inputs)
    pafs = outputs[PAF_INDEX][0, ...]
    heatmaps = outputs[HEATMAP_INDEX][0, ...]

    cfg = get_default_configuration()

    coordinates = get_coordinates(cfg, heatmaps)

    connections = get_connections(cfg, coordinates, pafs)

    skeletons = estimate(cfg, connections)

    output = draw(cfg, img, coordinates, skeletons, resize_fac=8)

    image_name = image.split("/")[-1] # rofl.jpg
    image_name = image_name.split(".")[0] + "_skeleton." + image_name.split(".")[-1]
    output_folder = os.path.join(output_folder, image_name)
    return(output, skeletons)


