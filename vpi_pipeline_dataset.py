import cv2
import numpy as np
from gstreamer.gstreamer_base_code import __gstreamer_pipeline
#from rectification.stereo_rectification_calibrated import stereo_rectification_calibrated
from stereo_rectification_calibrated import stereo_rectification_calibrated
import vpi
import os
from tqdm import tqdm
from vpi_pipeline import vpi_pipeline

def get_start_index(image_list, start_filename):
    """Find the index of the starting filename in the sorted list of image filenames."""
    try:
        return image_list.index(start_filename)
    except ValueError:
        print(f"File {start_filename} not found in the list.")
        return None

if __name__ == "__main__":
    # Paths to the folders containing the stereo images
    path = "/media/seaclear/639f8c93-fac2-4b84-a4fe-b261541674e9/lab_tests/lab_test_2908/29-08-2024_14:36:52"

    # Join the base path with folder names
    left_folder = os.path.join(path, 'right')
    right_folder = os.path.join(path, 'left')
    depth_folder = os.path.join(path, 'depth')
    out_img_folder = os.path.join(path, 'color')
    disparity_folder = os.path.join(path, 'disparity')

    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(out_img_folder, exist_ok=True)
    os.makedirs(disparity_folder, exist_ok=True)

    # List all files in the left and right folders
    left_images = sorted(os.listdir(left_folder))
    right_images = sorted(os.listdir(right_folder))

    # Define the starting filename (e.g., '000001.png')
    start_filename = '14:37:05.671.jpg'  # Adjust this to your starting filename

    # Find the start index
    start_index = get_start_index(left_images, start_filename)
    if start_index is None:
        # Exit if the start file is not found
        exit()

    for i in tqdm(range(start_index, len(left_images))):
        depth_path = os.path.join(depth_folder, str(i).zfill(6)+'.png')
        color_out_path = os.path.join(out_img_folder, str(i).zfill(6)+'.jpg')
        disparity_out_path = os.path.join(disparity_folder, str(i).zfill(6)+'.png')

        # Load the left and right images
        left_image_path = os.path.join(left_folder, left_images[i])
        right_image_path = os.path.join(right_folder, right_images[i])

        left_image = cv2.imread(left_image_path)
        right_image = cv2.imread(right_image_path)

        if left_image is None or right_image is None:
            print(f"Skipping pair {i}: Unable to load one of the images.")
            continue
        
        depth_map, color, disparity, stereo = vpi_pipeline(left_image, right_image)
        
        color = cv2.resize(color, (960, 480), interpolation=cv2.INTER_LINEAR)
        disparity = cv2.resize(disparity, (960, 480), interpolation=cv2.INTER_LINEAR)

        postProcStream = vpi.Stream()
        with postProcStream, vpi.Backend.CUDA:

                vpi_depth_map = vpi.asimage(depth_map)

                # Resize the image
                resized_depth = vpi_depth_map.rescale((960, 480))  # Resize to half the original size

                median_filtered = resized_depth.median_filter((11, 11))  # 5x5 kernel

                # Apply Median Filter
                for i in range(10):
                    median_filtered = median_filtered.median_filter((2, 2))  # 5x5 kernel

                # Apply Bilateral Filter
                bilateral_filtered = median_filtered.bilateral_filter(9, 21, 11)  # Bilateral filter with kernel size 9, sigma_color 0.1, sigma_space 2.0

                # Convert back to OpenCV image for saving/displaying
                result_depth = bilateral_filtered.cpu()
                #result_depth = bilateral_filtered.convert(vpi.Format.U16, scale=65535.0 / (32 * max_disparity)).cpu()

        cv2.imwrite(depth_path, result_depth)
        cv2.imwrite(disparity_out_path, disparity)
        cv2.imwrite(color_out_path, color)
