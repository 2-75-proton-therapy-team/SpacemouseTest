from scipy import ndimage
import numpy as np
import time
import concurrent.futures

class VolumeTransformations:
    @staticmethod
    def get_translation_matrix(translation):
        """
        Returns a translation matrix for a given translation vector.
        Args:
            translation: The translation vector (e.g. [x, y, z]).
        Returns:
            The translation matrix.
        """

        T = np.array([
            [1, 0, 0, translation[0]],
            [0, 1, 0, translation[1]],
            [0, 0, 1, translation[2]],
            [0, 0, 0, 1]
        ])

        return T
    
    @staticmethod
    def get_rotation_matrix(axis, angles, degrees=True):
        """
        Returns a rotation matrix for a given set of angles.
        Args:
            angles: The angles to be rotated by (e.g. [roll, pitch, yaw]).
        Returns:
            The rotation matrix.
        """

        if degrees:
            angles = np.radians(angles)

        # Extract angles
        if axis == "axial":
            yaw, pitch, roll = angles
        elif axis == "sagittal":
            roll, yaw, pitch = angles
        else:
            pitch, yaw, roll = angles

        # Rotation matrices
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(roll), -np.sin(roll), 0],
            [0, np.sin(roll), np.cos(roll), 0],
            [0, 0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch), 0],
            [0, 1, 0, 0],
            [-np.sin(pitch), 0, np.cos(pitch), 0],
            [0, 0, 0, 1]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0, 0],
            [np.sin(yaw), np.cos(yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Combined rotation: first about x, then y, then z
        R = Rz @ Ry @ Rx

        return R

    @staticmethod
    def rotate(volume, axis, angles, degrees=True):
        """
        Rotates a volume by a given angle around a given axis.
        Args:
            volume: The volume to be rotated.
            angles: The angles to be rotated by (e.g. [roll, pitch, yaw]).
            degrees: Whether the angles are given in degrees or radians.
        Returns:
            The rotated volume.
        """
    
        rotated_v = volume.copy().astype(np.uint8)

        # # Extract angles
        # if axis == "axial":
        #     yaw, pitch, roll = angles
        #     z,y,x = 0,1,2
        # elif axis == "sagittal":
        #     roll, yaw, pitch = angles
        #     x,z,y = 0,1,2
        # else:
        #     pitch, yaw, roll = angles
        #     y,z,x = 0,1,2

        # def rotate_one_axis(slice, angle):
        #     return ndimage.rotate(slice, angle, reshape=False)
        
        # start = time.time()
        # for index, angle in zip([x, y, z], [roll, pitch, yaw]):
        #     if index == 0:
        #         rotated_v_slices = [rotated_v[i,:,:] for i in range(rotated_v.shape[index])]
        #     elif index == 1:
        #         rotated_v_slices = [rotated_v[:,i,:] for i in range(rotated_v.shape[index])]
        #     else:
        #         rotated_v_slices = [rotated_v[:,:,i] for i in range(rotated_v.shape[index])]
            
        #     with concurrent.futures.ThreadPoolExecutor() as executor:
        #         rotated_v = np.transpose(np.array(list(executor.map(lambda x: rotate_one_axis(x, angle), rotated_v_slices))),[index]+[i for i in range(3) if i != index])

        # print(time.time() - start)
        # return rotated_v
        # start = time.time()
        # for index, angle in enumerate([roll, pitch, yaw]):
        #     rotated_v = ndimage.rotate(rotated_v, 
        #                                angle, 
        #                                axes=tuple([i for i in range(3) if i != index]),
        #                                reshape=False,
        #                                order=1, 
        #                                mode='constant', 
        #                                cval=0)
        # print(time.time() - start)
        # return rotated_v

        R = VolumeTransformations.get_rotation_matrix(axis, angles, degrees)

        center = [rotated_v.shape[0] // 2, rotated_v.shape[1] // 2, rotated_v.shape[2] // 2]

        # Translation matrices to move center to origin and back
        translate_to_origin = np.array([[1, 0, 0, -center[0]],
                                        [0, 1, 0, -center[1]],
                                        [0, 0, 1, -center[2]],
                                        [0, 0, 0, 1]])
        translate_back = np.array([[1, 0, 0, center[0]],
                                   [0, 1, 0, center[1]],
                                   [0, 0, 1, center[2]],
                                   [0, 0, 0, 1]])

        combined_transform = translate_back @ R @ translate_to_origin
        rotated_v = ndimage.affine_transform(rotated_v, 
                                             combined_transform[:3,:3], 
                                             offset=combined_transform[:3, 3],
                                             output_shape=rotated_v.shape,
                                             order=1,  # Bilinear interpolation
                                             mode='constant',  # Set out-of-bounds values to a constant
                                             cval=0  # Black color for constant values
                                             )
        return rotated_v

    @staticmethod
    def translate(volume, shifts):
        """
        Translates a volume by a given amount in each direction.
        Args:
            volume: The volume to be translated.
            shifts: The amount to translate.
        Returns:
            The translated volume.
        """
        translated_v = ndimage.shift(volume, 
                                     shifts, 
                                     order=1, # Bilinear interpolation
                                     mode='constant', # Set out-of-bounds values to a constant
                                     cval=0 # Black color for constant values
                                     )
        return translated_v

    @staticmethod
    def scale(volume, scale):
        """
        Scales the volume by a given factor.
        Args:
            volume: The volume to be scaled.
            scale: The factor to scale the volume by.
        Returns:
            The scaled volume.
        """ 
        scaled_v = ndimage.zoom(volume, scale)
        return scaled_v
    
    @staticmethod
    def decompose(transformation_matrix, decimal=3):
        """
        Decomposes a transformation matrix into translation and rotation.
        Args:
            transformation_matrix: The transformation matrix to decompose.
        Returns:
            The translation and rotation vectors.
        """
        translations = np.round(transformation_matrix[:3, 3], decimal)

        # # Use SVD to decompose rotation and scaling
        # U, S, Vt = np.linalg.svd(transformation_matrix[:3, :3])
        # rotation_matrix = np.dot(U, Vt)
        # scaling = np.diag(S)

        scales = np.sqrt(np.sum(np.square(transformation_matrix[:3, :3]), axis=0))
        rotation_matrix = transformation_matrix[:3, :3] / scales

        roll = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
        pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(np.square(rotation_matrix[2, 1]) + np.square(rotation_matrix[2, 2])))
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        rotations = np.round(np.array([np.degrees(roll), np.degrees(pitch), np.degrees(yaw)]),decimal)

        return translations, rotations
    
    @staticmethod
    def transform(volume, translation, angles, axes):
        """
        Transforms a volume by a given translation and rotation.
        Args:
            volume: The volume to be transformed.
            translation: The translation vector (e.g. [x, y, z]).
            angles: The angles to be rotated by (e.g. [roll, pitch, yaw]).
            axes: The axes to rotate around (e.g. "axial", "sagittal", "coronal").
        Returns:
            The transformed volume.
        """
        T = VolumeTransformations.get_translation_matrix(translation)

        center = np.array([volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2])
        translate_to_origin = VolumeTransformations.get_translation_matrix(-center)
        translate_back = VolumeTransformations.get_translation_matrix(center)

        R = VolumeTransformations.get_rotation_matrix(axes, angles)
        
        A = (translate_back @ R @ translate_to_origin) @ T

        start = time.time()
        V = VolumeTransformations.transform_by_matrix(volume, A)
        print(time.time() - start)
        return V
    
    @staticmethod
    def transform_by_matrix(volume, transformation_matrix):
        """
        Transforms a volume by a given transformation matrix.
        Args:
            volume: The volume to be transformed.
            transformation_matrix: The transformation matrix.
        Returns:
            The transformed volume.
        """
        A = transformation_matrix
        V = ndimage.affine_transform(volume,
                                     A[:3,:3],
                                     offset=A[:3, 3],
                                     output_shape=volume.shape,
                                     order=1,  # Bilinear interpolation
                                     mode='constant',  # Set out-of-bounds values to a constant
                                     cval=0  # Black color for constant values
                                     )
        return V

    @staticmethod
    def update_transformation_matrix(transformation_matrix, translation, angles, axes):
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
        T = VolumeTransformations.get_translation_matrix(translation)
        R = VolumeTransformations.get_rotation_matrix(axes, angles)
        return T @ R @ transformation_matrix