import logging
import cv2
from ikomia.utils.tests import run_for_test


logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::infer Hugging Face instance segmentation=====")
    params = t.get_parameters()
    params["model_name"] = "facebook/maskformer-swin-tiny-ade"
    img = cv2.imread(data_dict["images"]["detection"]["coco"])[::-1]
    input_img_0 = t.get_input(0)
    input_img_0.set_image(img)
    t.set_parameters(params)
    yield run_for_test(t)