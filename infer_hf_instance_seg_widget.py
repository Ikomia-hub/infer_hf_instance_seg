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
from ikomia.utils import pyqtutils, qtconversion
from infer_hf_instance_seg.infer_hf_instance_seg_process import InferHfInstanceSegParam
from torch.cuda import is_available
# PyQt GUI framework
from PyQt5.QtWidgets import *
import os
from infer_hf_instance_seg.utils import Autocomplete

# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferHfInstanceSegWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferHfInstanceSegParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        # Loading model from list
        model_list_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        "model_list.txt")
        model_list_file = open(model_list_path, "r")

        model_list = model_list_file.read()
        model_list = model_list.split("\n")
        self.combo_model = Autocomplete(model_list, parent=None, i=True, allow_duplicates=False)
        self.label_model = QLabel("Model name")
        self.gridLayout.addWidget(self.combo_model, 0, 1)
        self.gridLayout.addWidget(self.label_model, 0, 0)
        self.combo_model.setCurrentText(self.parameters.model_name)
        model_list_file.close()

        # Cuda
        self.check_cuda = pyqtutils.append_check(
                        self.gridLayout, "Cuda",
                        self.parameters.cuda and is_available())

        # Threshold to keep predicted instance masks.
        self.double_spin_thres = pyqtutils.append_double_spin(
                                self.gridLayout,
                                "Confidence threshold",
                                self.parameters.conf_thres,
                                min = 0., max = 1.,
                                step = 0.01, decimals = 2)

        # mask_threshold  to use when turning the predicted masks into binary values.
        self.double_spin_mask_thres = pyqtutils.append_double_spin(
                                self.gridLayout,
                                "Confidence mask threshold",
                                self.parameters.conf_mask_thres,
                                min = 0., max = 1.,
                                step = 0.01, decimals = 2)

        # overlap_mask_area_threshold overlap mask area threshold to merge
        # or discard small disconnected parts within each binary instance mask.
        self.ds_overlap_mask_area_thres = pyqtutils.append_double_spin(
                                self.gridLayout,
                                "Confidence IOU",
                                self.parameters.conf_overlap_mask_area_thres,
                                min = 0., max = 1.,
                                step = 0.01, decimals = 2)

        # Link of available models from Hugging face hub
        urlLink = "<a href=\"https://huggingface.co/models?sort=downloads&search=maskformer\">"\
                 "List of Masformer models [Hugging Face Hub] </a>"
        self.qlabelModelLink = QLabel(urlLink)
        self.qlabelModelLink.setOpenExternalLinks(True)
        self.gridLayout.addWidget(self.qlabelModelLink, 7, 0)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot
        self.parameters.update = True
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.conf_thres = self.double_spin_thres.value()
        self.parameters.conf_mask_thres = self.double_spin_mask_thres.value()
        self.parameters.conf_overlap_mask_area_thres = self.ds_overlap_mask_area_thres.value()
        self.parameters.cuda = self.check_cuda.isChecked()
        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferHfInstanceSegWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_hf_instance_seg"

    def create(self, param):
        # Create widget object
        return InferHfInstanceSegWidget(param, None)
