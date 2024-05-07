import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap

def increase_contrast(slices, show=False):
    """
    Increase the contrast of the real slices.
    """
    contrast_slices = cv2.convertScaleAbs(slices)
    contrast_slices[contrast_slices < 20] = 0
    contrast_slices[contrast_slices > 255] = 255
    contrast_slices = cv2.bitwise_not(contrast_slices)

    if show:
        if len(slices.shape) == 3:
            # if 3D volume submitted, display middle slice
            slice_index = slices.shape[0] // 2
            cv2.imshow("Original", slices[slice_index])
            cv2.imshow("Increased Contrast", contrast_slices[slice_index])
        else:
            cv2.imshow("Original", slices)
            cv2.imshow("Increased Contrast", contrast_slices)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return contrast_slices

def test_increase_contrast(slice, show=False):
    """
    Increase the contrast of the Rigaku test slices.
    """
    # K-means clustering to increase contrast
    k = 3
    clustered_image = kmeans_clustering(slice,k)

    max_min = np.min(slice)
    for i in range(k):
        cluster_vals = slice[clustered_image == i]
        max_min = np.min(cluster_vals) if np.min(cluster_vals) > max_min else max_min

    slice[slice > max_min] = max_min

    # Normalization
    slice = slice.astype(np.float32)
    min_element = np.min(slice)
    max_element = np.max(slice)
    contrast_slice = (slice - min_element) / (max_element - min_element) * 255
    contrast_slice = contrast_slice.astype(np.uint8)
    
    # Histogram equalization
    contrast_slice = cv2.equalizeHist(contrast_slice)
    contrast_slice = cv2.addWeighted(contrast_slice,2.3,np.zeros(contrast_slice.shape, contrast_slice.dtype),0,10)
    contrast_slice = cv2.bitwise_not(contrast_slice)

    if show:
        if len(slice.shape) == 3:
            # if 3D volume submitted, display middle slice
            slice_index = slice.shape[0] // 2
            cv2.imshow("Original", slice[slice_index])
            cv2.imshow("Increased Contrast", contrast_slice[slice_index])
        else:
            cv2.imshow("Original", slice)
            cv2.imshow("Increased Contrast", contrast_slice)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return contrast_slice

def colorize(slice, colors, show=False):
    """
    Convert the grayscale images to the requested colormap
    """
    # Create a custom colormap
    color_scheme = "_".join(colors)
    color_values = {
        "black":(0, 0, 0, 1),  # Black
        "green":(0, 1, 0, 1),  # Green
        "magenta": (1, 0, 1, 1)  # Magenta
    }

    # Create a colormap from the color values
    cmap = LinearSegmentedColormap.from_list(color_scheme, [color_values[color] for color in colors])

    # Apply the colormap to the normalized image
    colored_slice = plt.get_cmap(cmap)(slice)

    # The result of plt.get_cmap() is RGBA, convert it to RGB
    colored_slice_rgb = (colored_slice[..., :3] * 255).astype(np.uint8)

    # Convert from RGB to BGR format for OpenCV
    colored_slice_bgr = cv2.cvtColor(colored_slice_rgb, cv2.COLOR_RGB2BGR)

    if show:
        # Display the image with OpenCV
        cv2.imshow(f"Color Map: {color_scheme}", colored_slice_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return colored_slice_bgr


def imshowpair_diff(image1, image2, show=False):
    """
    Given 2 images, overlap them and show the differences in images.
    """
    colored_slice1 = colorize(image1, ["black", "green"])
    colored_slice2 = colorize(image2, ["black", "magenta"])
    
    # Resize images to match (using the smallest dimensions)
    height = min(colored_slice1.shape[0], colored_slice2.shape[0])
    width = min(colored_slice1.shape[1], colored_slice2.shape[1])
    colored_slice1 = cv2.resize(colored_slice1, (width, height))
    colored_slice2 = cv2.resize(colored_slice2, (width, height))

    difference = cv2.add(colored_slice1, colored_slice2)
    
    # # Calculate absolute difference
    # difference = cv2.absdiff(colored_slice1, colored_slice2)
    
    # # Enhance difference for visualization (optional)
    # _, difference = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
    
    if show:
        # Display the difference
        cv2.imshow("IMG1", colored_slice1)
        cv2.imshow("IMG2", colored_slice2)
        cv2.imshow("difference", difference)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return difference

def kmeans_clustering(slice, k=3, show=False):
    """
    Perform Segmentation through KMeans Clustering
    """
    voxels = slice.astype(np.float32).reshape(-1, 1)

    # Initialize the KMeans object
    kmeans = KMeans(n_clusters=k)

    # Fit the model to the voxel data
    kmeans.fit(voxels)

    # Get the cluster labels for each voxel
    labels = kmeans.labels_

    # Reshape the labels back to the original CT scan shape
    clustered_ct_image = labels.reshape(slice.shape)

    if show:
        # Visualize the clustered CT scan using 3D plot
        fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_subplot(111)

        colors = ['r', 'g', 'b', "m", "y", "c", "k", "w"]  # you can add more colors if needed

        # Scatter plot each voxel with its cluster color
        for i in range(k):
            x = np.where(clustered_ct_image == i)[0]
            y = np.where(clustered_ct_image == i)[1]
            # z = np.where(clustered_ct_image == i)[2]
            ax.scatter(x, y, c=colors[i], marker='.', alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        plt.title('Clustered CT Scan')
        plt.show()
        plt.savefig('segmented_image.png')

    return clustered_ct_image