from Volume_Transformations import *
from Slice_Transformations import *
from Slices import *
from image_processing import *

import keyboard
import spacenavigator
import os

isStopped = False
total_x_shift = 0
total_y_shift = 0
total_z_shift = 0

total_roll = 0
total_pitch = 0
total_yaw = 0

def keyboard_stop(event):
  global isStopped 
  isStopped = True

if __name__ == "__main__":
    
    curr_dir = os.getcwd()
    dicom_folder_path = os.path.join(curr_dir, 'public_data', 'CQ500CT3', 'Unknown Study', 'CT PLAIN THIN')# "public_data/CQ500CT3/Unknown Study/CT PLAIN THIN"
    
    fixed_volume = Slices(path=dicom_folder_path, preprocess_fx=increase_contrast)
    fixed_volume = fixed_volume.change_orientation("sagittal")
    fixed_slices = fixed_volume.get_slices()

    moving_volume = fixed_volume.copy()
    moving_slices = moving_volume.get_slices()

    moving_slices = VolumeTransformations.translate(moving_slices, [0,5,10]) # x=0, z=5, y=10 translation
    z_indent = moving_volume.get_slice_thickness()

    slice_index = fixed_slices.shape[0] // 2
    moved_slice_index = slice_index
    fixed_slice = fixed_slices[slice_index]    
    moving_slice = moving_slices[slice_index]

    # Listen for the 'esc' key
    keyboard.on_press_key("esc", keyboard_stop) 

    while not isStopped:
        if spacenavigator.open():
            state = spacenavigator.read()
            print(f"x:{state.x}, y:{state.y}, z:{state.z}, roll: {state.roll}, pitch: {state.pitch}, yaw: {state.yaw}")

            if state.x or state.y:
                moving_slice = SliceTransformations.translate(moving_slice, [state.x, -state.y])
                total_x_shift += state.x
                total_y_shift -= state.y
            
            if state.z and abs(state.z) > .5:
                moved_slice_index += 1 if state.z > 0 else -1
                total_z_shift += z_indent*(1 if state.z > 0 else -1)
                moving_slice = moving_slices[moved_slice_index]
                moving_slice = SliceTransformations.translate(moving_slice, [total_x_shift, total_y_shift])
            
            if state.yaw:
                moving_slice = SliceTransformations.rotate(moving_slice, state.yaw)
        else: 
            print("No device detected")

        cv2.imshow("Overlayed Slices", imshowpair_diff(fixed_slice, moving_slice))
        cv2.waitKey(10)

    cv2.waitKey(1)
    cv2.destroyAllWindows()



