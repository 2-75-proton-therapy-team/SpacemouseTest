#!/bin/bash

# Set up python virtual environment
python -m venv manual_alignment_venv
./manual_alignment_venv/Scripts/Activate.ps1

# Install dependencies for colorization
pip install numpy
pip install tifffile
pip install dicom-numpy 
pip install opencv-python
pip install pydicom
pip install matplotlib
pip install scikit-learn
pip install scipy
pip install SimpleITK
pip install pyqt6

# Install dependencies for spacemouse compact joystick input
pip install keyboard
pip install spacenavigator
pip install pywinusb

deactivate
