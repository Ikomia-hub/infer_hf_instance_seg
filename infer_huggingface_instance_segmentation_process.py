# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
import copy
# Your imports below
import transformers
from transformers import AutoFeatureExtractor, AutoModelForInstanceSegmentation
from transformers.utils import logging
from ikomia.utils import strtobool
import numpy as np
import torch
import numpy as np
import random
import os

transformers.utils.logging.set_verbosity_error()

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferHuggingfaceInstanceSegmentationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.cuda = torch.cuda.is_available()
        self.model_name = "facebook/maskformer-swin-base-coco"
        self.checkpoint_path = ""
        self.checkpoint = False
        self.conf_thres = 0.500
        self.conf_mask_thres = 0.5
        self.conf_overlap_mask_area_thres = 0.8
        self.update = False

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = strtobool(param_map["cuda"])
        self.model_name = str(param_map["model_name"])
        self.pretrained = strtobool(param_map["checkpoint"])
        self.checkpoint_path = param_map["checkpoint_path"]
        self.conf_thres = float(param_map["conf_thres"])
        self.conf_mask_thres = float(param_map["conf_mask_thres"])
        self.conf_overlap_mask_area_thres = float(param_map["conf_overlap_mask_area_thres"])
        self.update = strtobool(param_map["update"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["cuda"] = str(self.cuda)
        param_map["model_name"] = str(self.model_name)
        param_map["checkpoint"] = str(self.checkpoint)
        param_map["checkpoint_path"] = self.checkpoint_path
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["conf_mask_thres"] = str(self.conf_mask_thres)
        param_map["conf_overlap_mask_area_thres"] = str(self.conf_overlap_mask_area_thres)
        param_map["update"] = str(self.update)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferHuggingfaceInstanceSegmentation(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        self.addOutput(dataprocess.CInstanceSegIO())

        # Create parameters class
        if param is None:
            self.setParam(InferHuggingfaceInstanceSegmentationParam())
        else:
            self.setParam(copy.deepcopy(param))

        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.instance_output = None
        self.model = None
        self.model_id = None
        self.feature_extractor = None
        self.colors = None
        self.classes = None
        self.update = False

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def infer(self, image):
        param = self.getParam()

        # Image pre-pocessing (image transformation and conversion to PyTorch tensor)
        encoding = self.feature_extractor(image, return_tensors="pt")
        if param.cuda is True:
            encoding = encoding.to(self.device)
        h, w, _ = np.shape(image)
        # Prediction
        with torch.no_grad():
            outputs = self.model(**encoding)
        results = self.feature_extractor.post_process_panoptic_segmentation(
                                outputs,
                                threshold = param.conf_thres,
                                mask_threshold= param.conf_mask_thres,
                                overlap_mask_area_threshold = param.conf_overlap_mask_area_thres,
                                target_sizes=[[h, w]],
                                label_ids_to_fuse= None,
                                )[0]

        segments_info = results["segments_info"]

        self.instance_output = self.getOutput(1)
        self.instance_output.init("PanopticSegmentation", 0, w, h)

        # dstImage
        dst_image = results["segmentation"].cpu().detach().numpy().astype(dtype=np.uint8)
        # Generating binary masks for each object present in the groundtruth mask
        unique_colors = np.unique(dst_image).tolist()
        unique_colors = [x for x in unique_colors if x != 0]

        masks = np.zeros(dst_image.shape)
        mask_list = []
        for color in unique_colors:
            object_mask = np.where(dst_image == color, 1, 0)
            mask_list.append(object_mask)
            masks = np.dstack([object_mask, masks])

        # Get bounding boxes from masks
        boxes = []
        for i in range(masks.shape[-1]):
            m = masks[:, :, i]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
            boxes.append([x1, y1, x2, y2])
        boxes = boxes[:-1]
        boxes.reverse()

        # Add segmented instance to the output
        for i, b, ml in zip(segments_info, boxes, mask_list):
            x_obj = float(b[0])
            y_obj = float(b[1])
            h_obj = (float(b[3]) - y_obj)
            w_obj = (float(b[2]) - x_obj)

            ml = ml.astype(dtype='uint8')  
            self.instance_output.addInstance(
                                    i["id"]-1,
                                    0,
                                    i["label_id"],
                                    self.classes[i["label_id"]],
                                    float(i["score"]),
                                    x_obj,
                                    y_obj,
                                    w_obj,
                                    h_obj,
                                    ml,
                                    self.colors[i["label_id"]]
                                    )

        self.forwardInputImage(0, 0)

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        image_in = self.getInput(0)

        # Get image from input/output (numpy array):
        image = image_in.getImage()

        param = self.getParam()

        if param.update or self.model is None:
        # Feature extractor selection
            model_id = None
            # Feature extractor selection
            if param.checkpoint is False:
                model_id = param.model_name
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            else:
                feature_extractor_path = os.path.join(
                                                    param.checkpoint_path,
                                                    "preprocessor_config.json"
                                                    )
                model_id = param.checkpoint_path
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)

            # Loading model weight
            self.model = AutoModelForInstanceSegmentation.from_pretrained(model_id)
            self.device = torch.device("cuda") if param.cuda else torch.device("cpu")
            self.model.to(self.device)
            print("Will run on {}".format(self.device.type))

            # Get label name
            self.classes = list(self.model.config.id2label.values())

            # Color palette
            n = len(self.classes)
            random.seed(14)
            self.colors = []
            for i in range(n):
                self.colors.append(random.choices(range(256), k=3))
            param.update = False

        # Inference
        self.infer(image)
        
        self.setOutputColorMap(0, 1, [[0, 0, 0]] + self.colors)
        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferHuggingfaceInstanceSegmentationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_huggingface_instance_segmentation"
        self.info.shortDescription = "Instance segmentation using models from Hugging Face."
        self.info.description = "This plugin proposes inference for instance segmentation"\
                                "using transformers models from Hugging Face. It regroups"\
                                "models covered by the Hugging Face class:"\
                                "<AutoModelForInstanceSegmentation>. Models can be loaded either"\
                                "from your fine-tuned model (local) or from the Hugging Face Hub."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.0.0"
        self.info.iconPath = "icons/icon.png"
        self.info.authors = "Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond,"\
                            "Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault,"\
                            "Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer,"\
                            "Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu,"\
                            "Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame,"\
                            "Quentin Lhoest, Alexander M. Rush"
        self.info.article = "Huggingface's Transformers: State-of-the-art Natural Language Processing"
        self.info.journal = "EMNLP"
        self.info.license = "Apache License Version 2.0"
        # URL of documentation
        self.info.documentationLink = "https://www.aclweb.org/anthology/2020.emnlp-demos.6"
        # Code source repository
        self.info.repository = "https://github.com/huggingface/transformers"
        # Keywords used for search
        self.info.keywords = "semantic, segmentation, inference, transformer,"\
                            "Hugging Face, Pytorch, Maskformer"

    def create(self, param=None):
        # Create process object
        return InferHuggingfaceInstanceSegmentation(self.info.name, param)
