import tifffile
import cv2
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from image_processing import *
import SimpleITK as sitk

class Slices:

    def __init__(self, **kwargs):
        """
        Initialize a Slices object
        """
        if "path" in kwargs:
            path = kwargs["path"]

            # tif file
            if path.endswith(".TIF"):
                print("Read TIFF File")
                self.slices = self._read_tif_file(path)
            
            # dicom file
            elif path.endswith(".dcm") or path.endswith(".DCM"):
                print("Read DCM File")
                self.slices, self.dimensions, self.orientation = self._read_dicom_file(path)

            # dicom folder 
            elif os.path.isdir(path) and os.listdir(path)[0].endswith(".dcm"):
                print("Reading DICOM files in Folder")
                self.slices, self.dimensions, self.orientation = self._read_dicom_folder(path)

        if "slices_vals" in kwargs:
            self.slices = kwargs["slices_vals"]["slices"]
            self.dimensions = kwargs["slices_vals"]["dimensions"]
            self.orientation = kwargs["slices_vals"]["orientation"]

        if "preprocess_fx" in kwargs:
            preprocess_fx = kwargs["preprocess_fx"]
            self.slices = preprocess_fx(self.slices)

    def _read_tif_file(self, tif_file_path):
        """
        Read a TIFF file into a 3D NumPy array.
        """
        with tifffile.TiffFile(tif_file_path) as tif:
            image_array = tif.asarray()
        return image_array

    def _read_dicom_file(self, dicom_file_path):
        """
        Read a DICOM file into a 3D Numpy array
        """
        slice = pydicom.dcmread(dicom_file_path)
        dimensions = [slice.SliceThickness, *(slice.PixelSpacing)]
        orientation = self._determine_orientation(slice.ImageOrientationPatient)
        return self._get_pixels_hu([slice]), dimensions, orientation

    
    def _read_dicom_folder(self, dicom_dir_path):
        """
        Read a folder of DICOM file into a 3D NumPy array.
        """
        slices = [pydicom.dcmread(dicom_dir_path + '/' + s) for s in os.listdir(dicom_dir_path)]

        # Order based on z (because orientation is axial)
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        
        dimensions = [slices[0].SliceThickness, *(slices[0].PixelSpacing)]
        orientation = self._determine_orientation(slices[0].ImageOrientationPatient)
        return self._get_pixels_hu(slices), dimensions, orientation

    def _get_pixels_hu(self,slices):
        """
        Convert List of Dicom Datasets to numpy array.
        """
        image = np.stack([s.pixel_array for s in slices])
        image = image.astype(np.int16)
        image[image == -2000] = 0
        intercept = slices[0].RescaleIntercept
        slope = slices[0].RescaleSlope
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
        image += np.int16(intercept)
        return np.flip(np.array(image, dtype=np.int16), axis=0)
    
    def _determine_orientation(self, iop, threshold=1):
        """ 
        Determine the anatomical orientation of a DICOM slice. 
        Args:
            threshold: Threshold for cosine similarity (how close to 1 or 0 the cosines need to be)
        """
        # Define the standard anatomical directions
        axial_cosines = np.array([[1, 0, 0], [0, 1, 0]])
        sagittal_cosines = np.array([[0, 0, 1], [0, 1, 0]])
        coronal_cosines = np.array([[1, 0, 0], [0, 0, 1]])

        # Calculate differences from each standard plane
        axial_diff = np.sum(np.abs(axial_cosines - np.array(iop).reshape(2, 3)))
        sagittal_diff = np.sum(np.abs(sagittal_cosines - np.array(iop).reshape(2, 3)))
        coronal_diff = np.sum(np.abs(coronal_cosines - np.array(iop).reshape(2, 3)))

        # If min difference > threshold, no orientation assigned
        if min(axial_diff, sagittal_diff, coronal_diff) > threshold:
            return "Oblique or undefined orientation"
    
        # Determine the closest plane
        if axial_diff < sagittal_diff and axial_diff < coronal_diff:
            return "axial"
        elif sagittal_diff < coronal_diff:
            return "sagittal"
        else:
            return "coronal"
    
    def get_slices(self):
        return self.slices
    
    def get_dimensions(self):
        return self.dimensions
    
    def get_slice_thickness(self):
        return self.dimensions[0]
    
    def get_pixel_spacing(self):
        return self.dimensions[1:]
    
    def get_orientation(self):
        return self.orientation
    
    def get_all_vals(self):
        return self.__dict__
    
    def copy(self):
        return Slices(slices_vals=self.__dict__)
    
    def resize(self, new_dimensions, new=True):
        """
        Given the desired new dimensions (Slice Thickness, Pixel Spacing), return new Slices Object.
        """
        slices_obj = self.copy() if new else self

        # Convert the NumPy array to a SimpleITK image
        sitk_image = sitk.GetImageFromArray(slices_obj.get_slices())
        sitk_image.SetSpacing(slices_obj.get_dimensions()[::-1])

        # Define a resample filter
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetOutputSpacing(new_dimensions)
        output_size = [int(round(old_size*old_dim/new_dim)) for old_size, old_dim, new_dim in zip(sitk_image.GetSize(), sitk_image.GetSpacing(), new_dimensions)]
        resample.SetSize(output_size)
        transform = sitk.Transform() # Identity Matrix
        resample.SetTransform(transform)

        # Run the resampling
        resampled_image = resample.Execute(sitk_image)
        resampled_array = sitk.GetArrayFromImage(resampled_image)

        slices_obj.dimensions = new_dimensions
        slices_obj.slices = resampled_array

        return slices_obj
    
    def change_orientation(self, new_orientation, new=True):
        """
        Given the desired new orientation of the slices, return new Slices Object.
        """

        slices_obj = self.copy() if new else self
        if self.orientation == new_orientation:
            return slices_obj

        if self.orientation == "axial":
            z,y,x = [0,1,2]
            z_spacing, y_spacing, x_spacing = slices_obj.get_dimensions()
        elif self.orientation == "sagittal":
            x,z,y = [0,1,2]
            x_spacing, z_spacing, y_spacing = slices_obj.get_dimensions()
        elif self.orientation == "coronal":
            y,z,x = [0,1,2]
            y_spacing, z_spacing, x_spacing = slices_obj.get_dimensions()

        if new_orientation == "axial":
            slices_obj.slices = np.transpose(slices_obj.get_slices(), (z, y, x))
            slices_obj.dimensions = [z_spacing, y_spacing, x_spacing]
        elif new_orientation == "sagittal":
            slices_obj.slices = np.transpose(slices_obj.get_slices(), (x, z, y))
            slices_obj.dimensions = [x_spacing, z_spacing, y_spacing]
        elif new_orientation == "coronal":
            slices_obj.slices = np.transpose(slices_obj.get_slices(), (y, z, x))
            slices_obj.dimensions = [y_spacing, z_spacing, x_spacing]
            
        slices_obj.orientation = new_orientation

        return slices_obj
    
    def get_x_ray_sim(self, orientation=None, isSquare=True, show=False):
        """
        Simulate X-Ray view from 3D Volume.
        """
        if orientation and orientation != self.orientation:
            slices = self.change_orientation(orientation).get_slices()
        else:
            slices = self.slices

        # Sum Over Depth
        x_ray_sim = np.sum(slices, axis=0)

        # Normalization
        x_ray_sim_max = np.max(x_ray_sim)
        x_ray_sim_min = np.min(x_ray_sim)
        x_ray_sim = (x_ray_sim - x_ray_sim_min) / (x_ray_sim_max - x_ray_sim_min)

        # Post processing
        x_ray_sim = cv2.convertScaleAbs(x_ray_sim, alpha=(255.0))

        if isSquare:
            height, width = x_ray_sim.shape
            side_length = max(x_ray_sim.shape)
            canvas = np.zeros((side_length, side_length), dtype=np.uint8)

            # Calculate the position to paste the resized image
            x_offset = (side_length - width) // 2
            y_offset = (side_length - height) // 2

            # Paste the resized image onto the canvas
            canvas[y_offset:y_offset+height, x_offset:x_offset+width] = x_ray_sim
            x_ray_sim = canvas

        x_ray_sim = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8)).apply(x_ray_sim)

        # High-Pass Filter using Gaussian Blur
        blurred = cv2.GaussianBlur(x_ray_sim, (15, 15), 10)
        x_ray_sim = cv2.addWeighted(x_ray_sim, 1.5, blurred, -0.5, 0)

        # Canny Edge Detection
        edges = cv2.Canny(x_ray_sim, threshold1=150, threshold2=200)

        x_ray_sim = cv2.addWeighted(x_ray_sim, 1.0, edges, 0.5, 0)

        if show:
            cv2.imshow(f"{orientation} x-ray slice", x_ray_sim)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return x_ray_sim
    
    def view(self, 
             preprocess_fx=None, 
             start=0, stop=None, step=1, 
             wait=50,
             orientation=None, viewer="cv2"):
        """
        View 3D volume.
        """
        if orientation and orientation != self.orientation:
            slices = self.change_orientation(orientation).get_slices()
        else:
            slices = self.slices

        if stop is None:
            stop = len(slices)

        if preprocess_fx:
            slices = preprocess_fx(slices)

        if viewer == "cv2": # cv2 window 
            for slice in slices[start:stop:step]:
                cv2.imshow("slice", slice)
                cv2.waitKey(wait)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else: # plt interactive window
            plt.ion() 
            fig = plt.figure() 
            ax = fig.add_subplot(111) 

            for slice in slices[start:stop:step]:
                ax.imshow(slice, cmap=plt.cm.gray)
                plt.pause(wait/100)
                ax.clear()
            plt.show()


