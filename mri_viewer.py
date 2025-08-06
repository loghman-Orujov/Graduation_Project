#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRI Viewer
----------
Tool for viewing and comparing MRI files in NIfTI (.nii/.nii.gz) format.
"""

import os
import sys
import glob
import numpy as np
import nibabel as nib
from scipy import ndimage
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QSlider, QMessageBox,
                            QPushButton, QFileDialog, QGroupBox, QGridLayout, 
                            QFrame, QSplitter, QStatusBar, QProgressBar, QSpacerItem,
                            QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

class MRICanvas(FigureCanvasQTAgg):
    """Matplotlib-based canvas for MRI visualization"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True, facecolor='#2b2b2b')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#2b2b2b')
        self.axes.axis('off')
        super().__init__(self.fig)
        self.setStyleSheet("background-color: #2b2b2b; border: 2px solid #444444; border-radius: 8px;")
        self.mri_data = None
        self.current_slice = 0
        self.orientation = 'axial'  # default: axial (x-y plane)
        self.im = None
        self.title = ""
        
    def set_data(self, mri_data, title=""):
        """Set MRI data and display the first slice"""
        if mri_data is None:
            self.clear()
            return
            
        # If new data arrives or dimensions change, update
        if self.mri_data is None or mri_data.shape != self.mri_data.shape:
            self.mri_data = mri_data
            self.update_slice(0)  # Show first slice
            
        self.title = title
        self.update_display()
        
    def update_orientation(self, orientation):
        """Change image orientation"""
        self.orientation = orientation
        if self.mri_data is not None:
            max_slice = self.get_max_slice()
            self.current_slice = min(self.current_slice, max_slice-1)  # Don't exceed boundaries
            self.update_display()
            
    def get_max_slice(self):
        """Return maximum number of slices according to selected orientation"""
        if self.mri_data is None:
            return 0
            
        if self.orientation == 'sagittal':  # x axis (right-left)
            return self.mri_data.shape[0]
        elif self.orientation == 'coronal':  # y axis (front-back)
            return self.mri_data.shape[1]
        else:  # axial (z axis, bottom-top)
            return self.mri_data.shape[2]
    
    def update_slice(self, slice_num):
        """Update displayed slice"""
        self.current_slice = int(slice_num)
        if self.mri_data is not None:
            self.update_display()
    
    def update_display(self):
        """Update the image"""
        if self.mri_data is None:
            return
            
        self.axes.clear()
          # Get slice according to selected orientation - Adjusted for straight view
        if self.orientation == 'sagittal':  # x axis (right-left)
            slice_data = self.mri_data[self.current_slice, :, :]
        elif self.orientation == 'coronal':  # y axis (front-back)
            slice_data = self.mri_data[:, self.current_slice, :]
            # Rotate for straight view
            slice_data = np.rot90(slice_data)
        else:  # axial (z axis, bottom-top)
            slice_data = self.mri_data[:, :, self.current_slice]
          # Display image (straight, without transpose)
        self.im = self.axes.imshow(slice_data, cmap='gray')
        self.axes.set_title(f"{self.title}\nSlice: {self.current_slice+1}", fontsize=10, color='white', pad=10)
        self.axes.axis('off')
        self.draw()


class DifferenceCanvas(FigureCanvasQTAgg):
    """Visualization of differences between MRI images"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True, facecolor='#2b2b2b')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#2b2b2b')
        self.axes.axis('off')
        super().__init__(self.fig)
        self.setStyleSheet("background-color: #2b2b2b; border: 2px solid #444444; border-radius: 8px;")
        self.mri_data1 = None
        self.mri_data2 = None
        self.diff_data = None
        self.current_slice = 0
        self.orientation = 'axial'  # default: axial
        self.im = None
        self.title = ""
        
    def set_data(self, mri_data1, mri_data2, title=""):
        """Set MRI data and display the first slice"""
        if mri_data1 is None or mri_data2 is None:
            self.clear()
            return
            
        # If new data arrives or dimensions change, update
        update_needed = (self.mri_data1 is None or self.mri_data2 is None or 
                        mri_data1.shape != self.mri_data1.shape or
                        mri_data2.shape != self.mri_data2.shape)
                        
        if update_needed:
            # Check dimensions of both images
            if mri_data1.shape != mri_data2.shape:
                print(f"Warning: MRI images have different dimensions. Applying resizing.")
                # Resize second image according to first image dimensions
                scale_factors = [s2/s1 for s1, s2 in zip(mri_data2.shape, mri_data1.shape)]
                mri_data2 = ndimage.zoom(mri_data2, scale_factors, order=1)
            
            self.mri_data1 = mri_data1
            self.mri_data2 = mri_data2
            
            # Calculate difference
            self.diff_data = mri_data1 - mri_data2
            
            # Filter small differences (reduce noise)
            # Take values above average + standard deviation of absolute difference
            threshold = np.mean(np.abs(self.diff_data)) + 0.5 * np.std(np.abs(self.diff_data))
            mask = np.abs(self.diff_data) < threshold
            filtered_diff = self.diff_data.copy()
            filtered_diff[mask] = 0
            self.diff_data = filtered_diff
            
            self.update_slice(0)  # Show first slice
            
        self.title = title
        self.update_display()
        
    def update_orientation(self, orientation):
        """Change image orientation"""
        self.orientation = orientation
        if self.diff_data is not None:
            max_slice = self.get_max_slice()
            self.current_slice = min(self.current_slice, max_slice-1)
            self.update_display()
            
    def get_max_slice(self):
        """Return maximum number of slices according to selected orientation"""
        if self.diff_data is None:
            return 0
            
        if self.orientation == 'sagittal':  # x axis
            return self.diff_data.shape[0]
        elif self.orientation == 'coronal':  # y axis
            return self.diff_data.shape[1]
        else:  # axial (z axis)
            return self.diff_data.shape[2]
    
    def update_slice(self, slice_num):
        """Update displayed slice"""
        self.current_slice = int(slice_num)
        if self.diff_data is not None:
            self.update_display()
    
    def update_display(self):
        """Update the image"""
        if self.diff_data is None:
            return
            
        self.axes.clear()
          # Get slice according to selected orientation - Adjusted for straight view
        if self.orientation == 'sagittal':  # x axis
            slice_data = self.diff_data[self.current_slice, :, :]
        elif self.orientation == 'coronal':  # y axis
            slice_data = self.diff_data[:, self.current_slice, :]
            # Rotate for straight view
            slice_data = np.rot90(slice_data)
        else:  # axial (z axis)
            slice_data = self.diff_data[:, :, self.current_slice]
          # Display difference image (straight, without transpose)
        vmax = np.max(np.abs(slice_data)) if np.max(np.abs(slice_data)) > 0 else 1
        self.im = self.axes.imshow(slice_data, cmap='bwr', 
                                  vmin=-vmax, vmax=vmax)
        self.axes.set_title(f"{self.title}\nSlice: {self.current_slice+1}", fontsize=10, color='white', pad=10)
        self.axes.axis('off')
        self.draw()


class MRIViewer(QMainWindow):
    """MRI Viewer Main Window"""
    
    def __init__(self):
        super().__init__()
        
        # Set application style
        self.setStyleSheet(self.get_modern_style())
        
        # Main window settings
        self.setWindowTitle("🧠 Advanced MRI Viewer - Alzheimer's Disease Analysis")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        # MRI file list
        self.mri_files = []
        self.mri_paths = []
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready - Select MRI folder to begin analysis", 5000)
        
        # Progress bar for loading
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar.addPermanentWidget(self.progress_bar)
        
        # Setup UI
        self.setup_ui()
        
        # Automatically find initial MRI folder
        self.find_mri_folder()
    
    def get_modern_style(self):
        """Return modern dark theme stylesheet"""
        return """
        QMainWindow {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        QGroupBox {
            font-size: 14px;
            font-weight: bold;
            color: #ffffff;
            background-color: #2d2d2d;
            border: 2px solid #555555;
            border-radius: 10px;
            margin-top: 10px;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 10px 0 10px;
            color: #4CAF50;
            font-weight: bold;
            font-size: 16px;
        }
        
        QPushButton {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 12px 24px;
            font-size: 14px;
            font-weight: bold;
            border-radius: 8px;
            min-width: 120px;
        }
        
        QPushButton:hover {
            background-color: #45a049;
            transform: translateY(-2px);
        }
        
        QPushButton:pressed {
            background-color: #3d8b40;
        }
        
        QComboBox {
            background-color: #3d3d3d;
            border: 2px solid #555555;
            border-radius: 6px;
            padding: 8px;
            min-width: 150px;
            color: #ffffff;
            font-size: 13px;
        }
        
        QComboBox:hover {
            border-color: #4CAF50;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 30px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border: 5px solid transparent;
            border-top: 8px solid #ffffff;
            margin-right: 10px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #3d3d3d;
            color: #ffffff;
            selection-background-color: #4CAF50;
            border: 1px solid #555555;
            border-radius: 6px;
        }
        
        QLabel {
            color: #ffffff;
            font-size: 14px;
            font-weight: 500;
        }
        
        QSlider::groove:horizontal {
            border: 1px solid #555555;
            height: 8px;
            background: #3d3d3d;
            border-radius: 4px;
        }
        
        QSlider::handle:horizontal {
            background: #4CAF50;
            border: 2px solid #4CAF50;
            width: 20px;
            margin: -6px 0;
            border-radius: 10px;
        }
        
        QSlider::handle:horizontal:hover {
            background: #45a049;
            border-color: #45a049;
        }
        
        QSlider::sub-page:horizontal {
            background: #4CAF50;
            border-radius: 4px;
        }
        
        QStatusBar {
            background-color: #2d2d2d;
            color: #ffffff;
            border-top: 1px solid #555555;
            font-size: 12px;
        }
        
        QProgressBar {
            border: 2px solid #555555;
            border-radius: 5px;
            text-align: center;
            background-color: #3d3d3d;
            color: #ffffff;
        }
        
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 3px;
        }
        """
    
    def setup_ui(self):
        """Setup the user interface with modern styling"""
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header section
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.StyledPanel)
        header_frame.setStyleSheet("QFrame { background-color: #2d2d2d; border-radius: 10px; padding: 10px; }")
        header_layout = QHBoxLayout(header_frame)
        
        # Title and description
        title_widget = QWidget()
        title_layout = QVBoxLayout(title_widget)
        
        title_label = QLabel("🧠 Advanced MRI Viewer")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #4CAF50; margin: 0;")
        
        subtitle_label = QLabel("Neuroimaging Analysis Tool for Alzheimer's Disease Research")
        subtitle_label.setStyleSheet("font-size: 14px; color: #cccccc; margin: 0;")
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        title_layout.setContentsMargins(10, 5, 0, 5)
        
        header_layout.addWidget(title_widget)
        header_layout.addStretch()
        
        # MRI folder selection button
        self.folder_button = QPushButton("📁 Select MRI Folder")
        self.folder_button.clicked.connect(self.select_folder)
        self.folder_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                font-size: 16px;
                padding: 15px 25px;
                min-width: 180px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        
        header_layout.addWidget(self.folder_button)
        main_layout.addWidget(header_frame)
        
        # Control panel with better organization
        control_splitter = QSplitter(Qt.Horizontal)
        control_splitter.setStyleSheet("QSplitter::handle { background-color: #555555; width: 3px; }")
        
        # MRI file selection panel
        mri_select_group = QGroupBox("🔬 MRI File Selection")
        mri_select_layout = QGridLayout(mri_select_group)
        mri_select_layout.setSpacing(10)
        
        self.mri1_label = QLabel("Primary MRI:")
        self.mri1_label.setStyleSheet("color: #81C784; font-weight: bold;")
        self.mri1_combo = QComboBox()
        self.mri1_combo.currentIndexChanged.connect(self.update_mri_views)
        
        self.mri2_label = QLabel("Comparison MRI:")
        self.mri2_label.setStyleSheet("color: #64B5F6; font-weight: bold;")
        self.mri2_combo = QComboBox()
        self.mri2_combo.currentIndexChanged.connect(self.update_mri_views)
        
        mri_select_layout.addWidget(self.mri1_label, 0, 0)
        mri_select_layout.addWidget(self.mri1_combo, 0, 1)
        mri_select_layout.addWidget(self.mri2_label, 1, 0)
        mri_select_layout.addWidget(self.mri2_combo, 1, 1)
        
        control_splitter.addWidget(mri_select_group)
        
        # Orientation selection panel
        orientation_group = QGroupBox("📐 Viewing Plane")
        orientation_layout = QVBoxLayout(orientation_group)
        
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems([
            '🔄 Axial (Top-Bottom)', 
            '↕️ Coronal (Front-Back)', 
            '↔️ Sagittal (Right-Left)'
        ])
        self.orientation_combo.currentIndexChanged.connect(self.change_orientation)
        
        orientation_layout.addWidget(self.orientation_combo)
        control_splitter.addWidget(orientation_group)
        
        # Set splitter proportions
        control_splitter.setSizes([400, 300])
        main_layout.addWidget(control_splitter)
        
        # Slice control panel
        slice_frame = QFrame()
        slice_frame.setFrameStyle(QFrame.StyledPanel)
        slice_frame.setStyleSheet("QFrame { background-color: #2d2d2d; border-radius: 10px; padding: 15px; }")
        slice_layout = QHBoxLayout(slice_frame)
        
        self.slice_label = QLabel("📊 Slice Navigation:")
        self.slice_label.setStyleSheet("font-weight: bold; color: #FFA726;")
        
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self.change_slice)
        
        self.slice_value_label = QLabel("0/0")
        self.slice_value_label.setStyleSheet("font-weight: bold; color: #FFD54F; min-width: 60px;")
        
        slice_layout.addWidget(self.slice_label)
        slice_layout.addWidget(self.slice_slider, 1)
        slice_layout.addWidget(self.slice_value_label)
        
        main_layout.addWidget(slice_frame)
        
        # MRI display panel with splitters
        display_splitter = QSplitter(Qt.Horizontal)
        display_splitter.setStyleSheet("QSplitter::handle { background-color: #555555; width: 5px; }")
        
        # MRI 1 Canvas
        mri1_group = QGroupBox("🧠 Primary MRI")
        mri1_group.setStyleSheet("QGroupBox::title { color: #81C784; }")
        mri1_layout = QVBoxLayout(mri1_group)
        mri1_layout.setContentsMargins(10, 25, 10, 10)
        self.mri1_canvas = MRICanvas(width=5, height=5)
        mri1_layout.addWidget(self.mri1_canvas)
        display_splitter.addWidget(mri1_group)
        
        # MRI 2 Canvas
        mri2_group = QGroupBox("🔬 Comparison MRI")
        mri2_group.setStyleSheet("QGroupBox::title { color: #64B5F6; }")
        mri2_layout = QVBoxLayout(mri2_group)
        mri2_layout.setContentsMargins(10, 25, 10, 10)
        self.mri2_canvas = MRICanvas(width=5, height=5)
        mri2_layout.addWidget(self.mri2_canvas)
        display_splitter.addWidget(mri2_group)
        
        # Difference Canvas
        diff_group = QGroupBox("🎯 Difference Analysis")
        diff_group.setStyleSheet("QGroupBox::title { color: #FF8A65; }")
        diff_layout = QVBoxLayout(diff_group)
        diff_layout.setContentsMargins(10, 25, 10, 10)
        self.diff_canvas = DifferenceCanvas(width=5, height=5)
        diff_layout.addWidget(self.diff_canvas)
        display_splitter.addWidget(diff_group)
        
        # Set equal sizes for all three panels
        display_splitter.setSizes([400, 400, 400])
        main_layout.addWidget(display_splitter)
        
        # Add tooltips and keyboard shortcuts
        self.setup_tooltips_and_shortcuts()
    
    def setup_tooltips_and_shortcuts(self):
        """Setup tooltips and keyboard shortcuts for better user experience"""
        # Tooltips
        self.folder_button.setToolTip("Select a folder containing NIfTI (.nii or .nii.gz) MRI files\nShortcut: Ctrl+O")
        self.mri1_combo.setToolTip("Select the primary MRI scan for analysis")
        self.mri2_combo.setToolTip("Select the comparison MRI scan")
        self.orientation_combo.setToolTip("Choose the viewing plane:\n• Axial: Top-down view\n• Coronal: Front-back view\n• Sagittal: Left-right view\nShortcut: Ctrl+1/2/3")
        self.slice_slider.setToolTip("Navigate through MRI slices\nUse arrow keys or mouse wheel")
        
        # Keyboard shortcuts
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        
        # File operations
        open_shortcut = QShortcut(QKeySequence.Open, self)
        open_shortcut.activated.connect(self.select_folder)
        
        # Orientation shortcuts
        axial_shortcut = QShortcut(QKeySequence("Ctrl+1"), self)
        axial_shortcut.activated.connect(lambda: self.orientation_combo.setCurrentIndex(0))
        
        coronal_shortcut = QShortcut(QKeySequence("Ctrl+2"), self)
        coronal_shortcut.activated.connect(lambda: self.orientation_combo.setCurrentIndex(1))
        
        sagittal_shortcut = QShortcut(QKeySequence("Ctrl+3"), self)
        sagittal_shortcut.activated.connect(lambda: self.orientation_combo.setCurrentIndex(2))
        
        # Slice navigation shortcuts
        next_slice_shortcut = QShortcut(QKeySequence("Right"), self)
        next_slice_shortcut.activated.connect(self.next_slice)
        
        prev_slice_shortcut = QShortcut(QKeySequence("Left"), self)
        prev_slice_shortcut.activated.connect(self.prev_slice)
        
        # Jump shortcuts
        first_slice_shortcut = QShortcut(QKeySequence("Home"), self)
        first_slice_shortcut.activated.connect(lambda: self.slice_slider.setValue(0))
        
        last_slice_shortcut = QShortcut(QKeySequence("End"), self)
        last_slice_shortcut.activated.connect(lambda: self.slice_slider.setValue(self.slice_slider.maximum()))
    
    def next_slice(self):
        """Navigate to next slice"""
        current = self.slice_slider.value()
        if current < self.slice_slider.maximum():
            self.slice_slider.setValue(current + 1)
    
    def prev_slice(self):
        """Navigate to previous slice"""
        current = self.slice_slider.value()
        if current > 0:
            self.slice_slider.setValue(current - 1)
    
    def find_mri_folder(self):
        """Try to automatically find MRI folder"""
        # Look for a folder named "MRI" in the working directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mri_folder = os.path.join(current_dir, "MRI")
        
        if os.path.exists(mri_folder) and os.path.isdir(mri_folder):
            print(f"MRI folder found: {mri_folder}")
            self.statusBar.showMessage(f"✅ MRI folder found: {mri_folder}", 3000)
            self.load_mri_files(mri_folder)
        else:
            print("MRI folder not found. Please select manually.")
            self.statusBar.showMessage("📂 Please select MRI folder to begin analysis", 0)
    
    def select_folder(self):
        """Select folder containing MRI files"""
        folder = QFileDialog.getExistingDirectory(
            self, 
            "Select MRI Folder", 
            "", 
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if folder:
            self.statusBar.showMessage(f"📁 Loading MRI files from: {folder}", 0)
            self.load_mri_files(folder)
    
    def load_mri_files(self, folder_path):
        """Load all MRI files from specified folder"""
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Find .nii and .nii.gz files
        nii_files = glob.glob(os.path.join(folder_path, "*.nii"))
        nii_files += glob.glob(os.path.join(folder_path, "*.nii.gz"))
        
        if not nii_files:
            self.progress_bar.setVisible(False)
            QMessageBox.warning(
                self, 
                "⚠️ No Files Found", 
                f"No NIfTI files found in selected folder:\n\n{folder_path}\n\nPlease select a folder containing .nii or .nii.gz files."
            )
            self.statusBar.showMessage("❌ No MRI files found in selected folder", 5000)
            return
        
        # Clear file list
        self.mri_files = []
        self.mri_paths = []
        self.mri1_combo.clear()
        self.mri2_combo.clear()
        
        # Add files with progress
        self.progress_bar.setRange(0, len(nii_files))
        for i, nii_file in enumerate(nii_files):
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()  # Keep UI responsive
            
            file_name = os.path.basename(nii_file)
            name = os.path.splitext(file_name)[0]
            if name.endswith('.nii'):  # Clean second extension for .nii.gz
                name = os.path.splitext(name)[0]
                
            self.mri_files.append(name)
            self.mri_paths.append(nii_file)
            print(f"MRI file found: {name} - {nii_file}")
        
        # Update ComboBoxes
        self.mri1_combo.addItems(self.mri_files)
        self.mri2_combo.addItems(self.mri_files)
        
        # Select different files
        if len(self.mri_files) > 1:
            self.mri2_combo.setCurrentIndex(1)
        
        # Hide progress and update status
        self.progress_bar.setVisible(False)
        self.statusBar.showMessage(f"✅ Loaded {len(nii_files)} MRI files successfully", 5000)
        
        # Enable controls if files were loaded
        if self.mri_files:
            self.mri1_combo.setEnabled(True)
            self.mri2_combo.setEnabled(True)
            self.orientation_combo.setEnabled(True)
            self.slice_slider.setEnabled(True)
    
    def change_orientation(self, index):
        """Change image orientation"""
        orientations = ['axial', 'coronal', 'sagittal']
        orientation = orientations[index]
        
        self.mri1_canvas.update_orientation(orientation)
        self.mri2_canvas.update_orientation(orientation)
        self.diff_canvas.update_orientation(orientation)
        
        self.update_slice_slider()
    
    def update_slice_slider(self):
        """Update slice slider"""
        # Get maximum slice count according to selected orientation
        max_slice = max(self.mri1_canvas.get_max_slice(), self.mri2_canvas.get_max_slice())
        
        if max_slice > 0:
            current_slice = min(self.slice_slider.value(), max_slice - 1)
            self.slice_slider.setMaximum(max_slice - 1)
            self.slice_slider.setValue(current_slice)
            self.slice_value_label.setText(f"{current_slice + 1}/{max_slice}")
        else:
            self.slice_slider.setMaximum(0)
            self.slice_slider.setValue(0)
            self.slice_value_label.setText("0/0")
    
    def change_slice(self, slice_num):
        """Change displayed slice"""
        self.mri1_canvas.update_slice(slice_num)
        self.mri2_canvas.update_slice(slice_num)
        self.diff_canvas.update_slice(slice_num)
        
        # Update slice label with enhanced formatting
        max_slice = max(self.mri1_canvas.get_max_slice(), self.mri2_canvas.get_max_slice())
        self.slice_value_label.setText(f"{slice_num + 1:03d}/{max_slice:03d}")
        
        # Update status bar with current slice info
        orientation_names = ['Axial', 'Coronal', 'Sagittal']
        current_orientation = orientation_names[self.orientation_combo.currentIndex()]
        self.statusBar.showMessage(f"📊 {current_orientation} slice {slice_num + 1} of {max_slice}", 2000)
    
    def update_mri_views(self):
        """Update MRI images"""
        idx1 = self.mri1_combo.currentIndex()
        idx2 = self.mri2_combo.currentIndex()
        
        if idx1 < 0 or idx2 < 0 or len(self.mri_paths) == 0:
            return
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 4)  # 4 steps: load MRI1, load MRI2, update views, complete
        
        try:
            # Step 1: Load first MRI
            self.progress_bar.setValue(1)
            self.statusBar.showMessage("📥 Loading primary MRI...", 0)
            QApplication.processEvents()
            
            mri1_path = self.mri_paths[idx1]
            mri1_name = self.mri_files[idx1]
            print(f"Loading MRI 1: {mri1_name} - {mri1_path}")
            mri1_nifti = nib.load(mri1_path)
            mri1_data = mri1_nifti.get_fdata()
            
            # Step 2: Load second MRI
            self.progress_bar.setValue(2)
            self.statusBar.showMessage("📥 Loading comparison MRI...", 0)
            QApplication.processEvents()
            
            mri2_path = self.mri_paths[idx2]
            mri2_name = self.mri_files[idx2]
            print(f"Loading MRI 2: {mri2_name} - {mri2_path}")
            mri2_nifti = nib.load(mri2_path)
            mri2_data = mri2_nifti.get_fdata()
            
            # Step 3: Update images
            self.progress_bar.setValue(3)
            self.statusBar.showMessage("🖼️ Updating visualizations...", 0)
            QApplication.processEvents()
            
            self.mri1_canvas.set_data(mri1_data, f"Primary: {mri1_name}")
            self.mri2_canvas.set_data(mri2_data, f"Comparison: {mri2_name}")
            self.diff_canvas.set_data(mri1_data, mri2_data, f"Diff: {mri1_name} vs {mri2_name}")
            
            # Step 4: Update slice slider
            self.progress_bar.setValue(4)
            self.update_slice_slider()
            
            # Complete
            self.progress_bar.setVisible(False)
            self.statusBar.showMessage(f"✅ Successfully loaded and compared: {mri1_name} vs {mri2_name}", 5000)
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            error_msg = f"❌ Error loading MRI files:\n\n{str(e)}\n\nPlease check if the files are valid NIfTI format."
            QMessageBox.critical(self, "Error Loading MRI Files", error_msg)
            self.statusBar.showMessage("❌ Error loading MRI files", 5000)
            print(f"Error: {e}")


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Advanced MRI Viewer")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Alzheimer's Research Lab")
    
    # Set application style to modern dark theme
    app.setStyle('Fusion')
    
    # Create and show the main window
    viewer = MRIViewer()
    viewer.show()
    
    # Center the window on screen
    screen = app.primaryScreen().availableGeometry()
    window_geometry = viewer.frameGeometry()
    window_geometry.moveCenter(screen.center())
    viewer.move(window_geometry.topLeft())
    
    # Start the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
