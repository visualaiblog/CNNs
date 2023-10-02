/*
 Author: Ritik Bompilwar
 Organization: Visual AI Blog

 Description:
 This C++ program demonstrates the creation of a Depth of Field (DOF) Bokeh effect using a custom convolution function.
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

// Define convolution function for color images
void convolution(const cv::Mat& inputImage, const std::vector<std::vector<std::vector<double>>>& kernel, cv::Mat& outputImage) {
    int kernelHeight = kernel.size();
    int kernelWidth = kernel[0].size();
    int paddingHeight = kernelHeight / 2;
    int paddingWidth = kernelWidth / 2;

    // Add padding to the input image
    cv::Mat paddedImage;
    cv::copyMakeBorder(inputImage, paddedImage, paddingHeight, paddingHeight, paddingWidth, paddingWidth, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // Create an empty output image
    outputImage = cv::Mat::zeros(inputImage.size(), inputImage.type());

    // Perform convolution
    for (int i = paddingHeight; i < paddedImage.rows - paddingHeight; ++i) {
        for (int j = paddingWidth; j < paddedImage.cols - paddingWidth; ++j) {
            double sumB = 0;
            double sumG = 0;
            double sumR = 0;
            for (int m = -paddingHeight; m <= paddingHeight; ++m) {
                for (int n = -paddingWidth; n <= paddingWidth; ++n) {
                    sumB += paddedImage.at<cv::Vec3b>(i + m, j + n)[0] * kernel[m + paddingHeight][n + paddingWidth][0];
                    sumG += paddedImage.at<cv::Vec3b>(i + m, j + n)[1] * kernel[m + paddingHeight][n + paddingWidth][1];
                    sumR += paddedImage.at<cv::Vec3b>(i + m, j + n)[2] * kernel[m + paddingHeight][n + paddingWidth][2];
                }
            }
            outputImage.at<cv::Vec3b>(i - paddingHeight, j - paddingWidth)[0] = cv::saturate_cast<uchar>(sumB);
            outputImage.at<cv::Vec3b>(i - paddingHeight, j - paddingWidth)[1] = cv::saturate_cast<uchar>(sumG);
            outputImage.at<cv::Vec3b>(i - paddingHeight, j - paddingWidth)[2] = cv::saturate_cast<uchar>(sumR);
        }
    }
}

// Function to create a synthetic depth map
cv::Mat createDepthMap(const cv::Mat& image) {
    cv::Mat depthMap(image.size(), CV_8U);

    // Define the depth values for foreground and background
    int foregroundDepth = 100;
    int backgroundDepth = 255;

    // Define the region as foreground (center of the image)
    int centerX = image.cols / 2;
    int centerY = image.rows / 2;
    int rectWidth = image.cols / 4;
    int rectHeight = image.rows / 4;

    // Set the circular region as foreground
    cv::circle(depthMap, cv::Point(centerX, centerY), rectWidth, cv::Scalar(foregroundDepth), -1);

    // Set the remaining region as background
    cv::rectangle(depthMap, cv::Rect(0, 0, image.cols, image.rows), cv::Scalar(backgroundDepth), -1);

    return depthMap;
}

// Function to apply a depth-based blur
void applyDepthBlur(const cv::Mat& inputImage, const cv::Mat& depthMap, cv::Mat& outputImage, int blurAmount) {
    // Create a Gaussian kernel for blurring
    std::vector<std::vector<std::vector<double>>> kernel(2 * blurAmount + 1, std::vector<std::vector<double>>(2 * blurAmount + 1, std::vector<double>(3, 0.0)));

    // Fill the Gaussian kernel
    double sigma = blurAmount / 2.0;
    double sum = 0.0;
    for (int i = -blurAmount; i <= blurAmount; ++i) {
        for (int j = -blurAmount; j <= blurAmount; ++j) {
            double value = exp(-(i * i + j * j) / (2.0 * sigma * sigma)) / (2.0 * CV_PI * sigma * sigma);
            kernel[i + blurAmount][j + blurAmount][0] = value;
            kernel[i + blurAmount][j + blurAmount][1] = value;
            kernel[i + blurAmount][j + blurAmount][2] = value;
            sum += value;
        }
    }

    // Normalize the kernel
    for (int i = 0; i < 2 * blurAmount + 1; ++i) {
        for (int j = 0; j < 2 * blurAmount + 1; ++j) {
            kernel[i][j][0] /= sum;
            kernel[i][j][1] /= sum;
            kernel[i][j][2] /= sum;
        }
    }

    // Apply convolution with the Gaussian kernel based on the depth map
    convolution(inputImage, kernel, outputImage);
}

int main() {
    // Load the input image using OpenCV (replace with your own file path)
    cv::Mat inputImage = cv::imread("path/to/your/input/image.jpg");

    if (inputImage.empty()) {
        std::cerr << "Error: Could not open or find the input image." << std::endl;
        return -1;
    }

    // Create a synthetic depth map
    cv::Mat depthMap = createDepthMap(inputImage);

    // Define the blur amount (adjust as needed)
    int blurAmount = 5;

    // Apply depth-based blur to create the DOF bokeh effect
    cv::Mat outputImage;
    applyDepthBlur(inputImage, depthMap, outputImage, blurAmount);

    // Display or save the updated image as needed (replace with your own file path)
    cv::imwrite("path/to/your/output/image.png", outputImage);
    cv::imshow("DOF Bokeh Effect", outputImage);
    cv::waitKey(0);

    return 0;
}
