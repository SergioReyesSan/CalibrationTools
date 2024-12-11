#!/usr/bin/env python3

# Copyright 2024 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import logging
import os

from PySide2.QtCore import Signal
from PySide2.QtWidgets import QComboBox
from PySide2.QtWidgets import QFileDialog
from PySide2.QtWidgets import QGroupBox
from PySide2.QtWidgets import QPushButton
from PySide2.QtWidgets import QRadioButton
from PySide2.QtWidgets import QVBoxLayout
from PySide2.QtWidgets import QWidget
from ament_index_python.packages import get_package_share_directory
from intrinsic_camera_calibrator.board_parameters.board_parameters_factory import (
    make_board_parameters,
)
from intrinsic_camera_calibrator.boards import BoardEnum
from intrinsic_camera_calibrator.camera_models.camera_model import CameraModel
from intrinsic_camera_calibrator.data_sources.data_source import DataSource
from intrinsic_camera_calibrator.data_sources.data_source import DataSourceEnum
from intrinsic_camera_calibrator.data_sources.data_source_factory import make_data_source
from intrinsic_camera_calibrator.types import OperationMode
from intrinsic_camera_calibrator.utils import load_intrinsics
from intrinsic_camera_calibrator.views.image_files_view import ImageFilesView
from intrinsic_camera_calibrator.views.parameter_view import ParameterView
from intrinsic_camera_calibrator.views.ros_bag_view import RosBagView
from intrinsic_camera_calibrator.views.ros_topic_view import RosTopicView
import yaml


class InitializationView(QWidget):
    """Initial widget to let the user configure the calibrator."""

    parameter_changed = Signal()
    closed = Signal()

    def __init__(self, calibrator: "CameraIntrinsicsCalibratorUI", cfg):  # noqa F821
        super().__init__()

        self.setWindowTitle("Initial configuration")
        self.setMinimumWidth(300)

        self.calibrator = calibrator
        self.cfg = cfg
        self.data_source_view: QWidget = None
        self.data_source: DataSource = None
        self.board_parameters_dict = {
            board_type: make_board_parameters(board_type, cfg=self.cfg["board_parameters"])
            for board_type in BoardEnum
        }

        # Get the package share directory
        package_share_dir = get_package_share_directory("intrinsic_camera_calibrator")
        # Get the path to the config directory
        config_dir = os.path.join(package_share_dir, "config")

        self.layout = QVBoxLayout(self)

        # Source
        self.source_group = QGroupBox("Source options")
        self.source_group.setFlat(True)

        self.data_source_combobox = QComboBox()

        for data_source in DataSourceEnum:
            self.data_source_combobox.addItem(str(data_source), data_source)

        source_layout = QVBoxLayout()
        source_layout.addWidget(self.data_source_combobox)
        self.source_group.setLayout(source_layout)

        # Calibration parameters
        self.params_combobox = QComboBox()

        # Add all YAML files from the config directory
        for file in os.listdir(config_dir):
            if file.endswith(".yaml"):  # and file != 'intrinsics_calibrator.yaml':
                file_path = os.path.join(config_dir, file)
                self.params_combobox.addItem(
                    file.split(".")[0].replace("_", " ").title(), file_path
                )
        # Add "Load File" option at the end
        self.params_combobox.addItem("Load File", None)

        def on_params_combo_box_changed(index):
            selected_params_file = self.params_combobox.itemData(index)
            if selected_params_file is None:
                file_name, _ = QFileDialog.getOpenFileName(
                    self, "Open File", "", "All Files (*.*);;Text Files (*.yaml)"
                )
                if file_name:
                    logging.info(f"Selected file: {file_name}")
                    config_file_path = file_name
            else:
                config_file_path = selected_params_file

            logging.info(f"Selected config file={config_file_path}")

            if config_file_path:
                cfg = {}
                try:
                    with open(config_file_path, "r") as stream:
                        cfg = yaml.safe_load(stream)
                        self.cfg = defaultdict(dict, cfg)
                        self.update_board_type()
                        logging.info("Successfully opened parameters file")
                except Exception as e:
                    logging.error(f"Could not load the parameters from the YAML file ({e})")

        self.params_combobox.currentIndexChanged.connect(on_params_combo_box_changed)

        self.params_group = QGroupBox("Parameters Profile")
        self.params_group.setFlat(True)
        params_layout = QVBoxLayout()
        params_layout.addWidget(self.params_combobox)
        self.params_group.setLayout(params_layout)

        # Board
        self.board_group = QGroupBox("Board options")
        self.board_group.setFlat(True)

        self.board_type_combobox = QComboBox()
        self.board_parameters_button = QPushButton("Board parameters")

        for board_type in BoardEnum:
            self.board_type_combobox.addItem(board_type.value["display"], board_type)

        self.update_board_type()

        def board_parameters_on_closed():
            self.setEnabled(True)

        def board_parameters_button_callback():
            board_parameters_view = ParameterView(
                self.board_parameters_dict[self.board_type_combobox.currentData()]
            )
            board_parameters_view.closed.connect(board_parameters_on_closed)
            self.setEnabled(False)

        self.board_parameters_button.clicked.connect(board_parameters_button_callback)

        board_layout = QVBoxLayout()
        board_layout.addWidget(self.board_type_combobox)
        board_layout.addWidget(self.board_parameters_button)
        self.board_group.setLayout(board_layout)

        # Mode
        self.mode_group = QGroupBox("Mode options")
        self.mode_group.setFlat(True)

        self.training_radio_button = QRadioButton("Calibration mode")
        self.evaluation_radio_button = QRadioButton("Evaluation mode")
        self.evaluation_radio_button.setEnabled(False)
        self.training_radio_button.setChecked(True)

        mode_layout = QVBoxLayout()
        mode_layout.addWidget(self.training_radio_button)
        mode_layout.addWidget(self.evaluation_radio_button)
        self.mode_group.setLayout(mode_layout)

        # Initial intrinsics
        self.initial_intrinsics_group = QGroupBox("Initial intrinsics")
        self.initial_intrinsics_group.setFlat(True)

        self.load_intrinsics_button = QPushButton("Load intrinsics")
        self.initial_intrinsics: CameraModel = None

        def load_intrinsics_button_callback():
            intrinsics_path, _ = QFileDialog.getOpenFileName(
                self, "Open initial intrinsics", "", "Image files (*.yaml)"
            )
            self.initial_intrinsics = load_intrinsics(intrinsics_path)
            self.evaluation_radio_button.setEnabled(True)
            self.training_radio_button.setEnabled(False)
            self.evaluation_radio_button.setChecked(True)

        self.load_intrinsics_button.clicked.connect(load_intrinsics_button_callback)

        initial_intrinsics_layout = QVBoxLayout()
        initial_intrinsics_layout.addWidget(self.load_intrinsics_button)
        self.initial_intrinsics_group.setLayout(initial_intrinsics_layout)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.on_start)

        self.layout.addWidget(self.source_group)
        self.layout.addWidget(self.board_group)
        self.layout.addWidget(self.mode_group)
        self.layout.addWidget(self.params_group)
        self.layout.addWidget(self.initial_intrinsics_group)
        self.layout.addWidget(self.start_button)

        self.show()

    def update_board_type(self):
        if self.cfg["board_type"] != "":
            self.board_type_combobox.setCurrentIndex(
                BoardEnum.from_name(self.cfg["board_type"]).get_id()
            )
        else:
            self.board_type_combobox.setCurrentIndex(0)

        self.board_parameters_dict = {
            board_type: make_board_parameters(board_type, cfg=self.cfg["board_parameters"])
            for board_type in BoardEnum
        }

    def on_start(self):
        """Start the calibration process after receiving the user settings."""

        def on_success():
            """Handle the successful initialization of the data source."""
            mode = (
                OperationMode.CALIBRATION
                if self.training_radio_button.isChecked()
                else OperationMode.EVALUATION
            )
            board_type = self.board_type_combobox.currentData()
            self.calibrator.start(
                mode,
                self.data_source,
                board_type,
                self.board_parameters_dict[board_type],
                self.initial_intrinsics,
                self.cfg,
            )
            self.close()

        def on_failed():
            """Handle the unsuccessful initialization of the data source."""
            self.setEnabled(True)
            self.data_source = None
            self.data_source_view = None

        source_type = self.data_source_combobox.currentData()

        if source_type == DataSourceEnum.TOPIC:
            self.data_source = make_data_source(self.data_source_combobox.currentData())
            self.data_source.set_data_callback(self.calibrator.data_source_external_callback)

            self.data_source_view = RosTopicView(self.data_source)
            self.data_source_view.failed.connect(on_failed)
            self.data_source_view.success.connect(on_success)
            self.setEnabled(False)

        elif source_type == DataSourceEnum.BAG2:
            self.data_source = make_data_source(self.data_source_combobox.currentData())
            self.data_source.set_data_callback(self.calibrator.data_source_external_callback)

            self.data_source_view = RosBagView(self.data_source)
            self.data_source_view.failed.connect(on_failed)
            self.data_source_view.success.connect(on_success)
            self.setEnabled(False)
        elif source_type == DataSourceEnum.FILES:
            self.data_source = make_data_source(self.data_source_combobox.currentData())
            self.data_source.set_data_callback(self.calibrator.data_source_external_callback)

            self.data_source_view = ImageFilesView(self.data_source)
            self.data_source_view.failed.connect(on_failed)
            self.data_source_view.success.connect(on_success)
            self.setEnabled(False)
        else:
            raise NotImplementedError

    def closeEvent(self, event):
        """When the window is closed, the widget is marked for deletion."""
        self.closed.emit()
        event.accept()
        self.deleteLater()
