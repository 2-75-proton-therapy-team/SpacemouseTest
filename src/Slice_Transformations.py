from Volume_Transformations import *

from scipy import ndimage
import numpy as np
# import time

class SliceTransformations:
    
    @staticmethod
    def rotate(image, angle):
        """
        Rotate an image by a given angle.
        """
        return ndimage.rotate(image, angle, reshape=False)

    @staticmethod
    def translate(image, translation):
        """
        Translate an image by a given x and y offset.
        """
        return ndimage.shift(image, (translation[1], translation[0]))
    
    @staticmethod
    def update_transformation_matrix(transformation_matrix, translation, angle, axes):
        """
        Updates a transformation matrix by a given translation and rotation.
        Args:
            transformation_matrix: The current transformation matrix.
            translation: The translation vector (e.g. [x, y, z]).
            angles: The angles to be rotated by (e.g. [roll, pitch, yaw]).
            axes: The axes to rotate around (e.g. "axial", "sagittal", "coronal").
        Returns:
            The updated transformation matrix.
        """
        x,y,z = 0,0,0
        if axes == "axial": 
            y,x = translation
        elif axes == "sagittal":
            z,y = translation
        else: # axes == "coronal"
            z,x = translation
        
        T = VolumeTransformations.get_translation_matrix([x,y,z])
        R = VolumeTransformations.get_rotation_matrix(axes, [angle, 0,0])
        return T @ R @ transformation_matrix