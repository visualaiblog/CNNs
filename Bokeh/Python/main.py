"""
Author: Ritik Bompilwar
Organization: Visual AI Blog

Description:
This Python script demonstrates the creation of a Depth of Field (DOF) Bokeh effect using a custom convolution function.
"""

import numpy as np
import cv2

# Define a custom convolution function
def custom_convolution(input_image, kernel):
    kernel_height, kernel_width = kernel.shape
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    # Get the dimensions of the input image
    image_height, image_width, channels = input_image.shape

    # Create an empty output image
    output_image = np.zeros_like(input_image)

    # Perform convolution for each color channel
    for channel in range(channels):
        for i in range(padding_height, image_height - padding_height):
            for j in range(padding_width, image_width - padding_width):
                sum_color = 0
                for m in range(-padding_height, padding_height + 1):
                    for n in range(-padding_width, padding_width + 1):
                        sum_color += input_image[i + m, j + n, channel] * kernel[m + padding_height, n + padding_width]
                output_image[i, j, channel] = np.uint8(sum_color)

    return output_image

# Function to create a synthetic depth map
def createDepthMap(image):
    depthMap = np.zeros(image.shape[:2], dtype=np.uint8)

    # Define the depth values for foreground and background
    foregroundDepth = 100
    backgroundDepth = 255

    # Define the region as foreground (center of the image)
    centerX = image.shape[1] // 2
    centerY = image.shape[0] // 2
    rectWidth = image.shape[1] // 4
    rectHeight = image.shape[0] // 4

    # Set the circular region as foreground
    cv2.circle(depthMap, (centerX, centerY), rectWidth, foregroundDepth, -1)

    # Set the remaining region as background
    depthMap[depthMap == 0] = backgroundDepth

    return depthMap

# Function to create a Gaussian blur kernel
def gaussian_blur_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

# Function to apply a depth-based blur
def applyDepthBlur(input_image, depthMap, blurAmount):
    # Create a Gaussian blur kernel with the specified blurAmount
    kernel_size = 2 * blurAmount + 1
    sigma = blurAmount / 2.0
    gaussian_kernel = gaussian_blur_kernel(kernel_size, sigma)

    # Apply the Gaussian blur using the custom convolution function
    return custom_convolution(input_image, gaussian_kernel)

# Load the input image using OpenCV (replace with your own file path)
input_image = cv2.imread('path/to/your/input/image.png')

# Create a synthetic depth map
depthMap = createDepthMap(input_image)

# Define the blur amount (adjust as needed)
blurAmount = 5

# Apply depth-based blur to create the DOF bokeh effect
output_image = applyDepthBlur(input_image, depthMap, blurAmount)

# Display the updated image using OpenCV
cv2.imshow('DOF Bokeh Effect with Custom Convolution', output_image)
cv2.waitKey(0)

# Save the updated image using OpenCV (replace with your own file path)
cv2.imwrite('path/to/your/output/image.png', output_image)
cv2.destroyAllWindows()
