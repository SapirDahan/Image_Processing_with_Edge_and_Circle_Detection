
# Image Processing with Edge Detection, Blurring, and Hough Circle Detection

This project demonstrates various image processing techniques including edge detection, blurring, and Hough Circle detection using OpenCV.

## Table of Contents
- [Introduction](#introduction)
- [Image Processing Techniques](#image-processing-techniques)
  - [Blurring](#blurring)
  - [Edge Detection](#edge-detection)
    - [Sobel Edge Detection](#sobel-edge-detection)
    - [Zero Crossing Simple Edge Detection](#zero-crossing-simple-edge-detection)
    - [Zero Crossing LOG Edge Detection](#zero-crossing-log-edge-detection)
    - [Canny Edge Detection](#canny-edge-detection)
  - [Hough Circle Detection](#hough-circle-detection)
- [Dependencies](#dependencies)
- [Usage](#usage)


## Introduction

This project provides implementations and demonstrations of various image processing techniques including blurring, edge detection, and Hough Circle detection using OpenCV in Python.

## Image Processing Techniques

### Blurring

Blurring is used to reduce noise and detail in an image. This project includes:
- Custom Gaussian blurring implementation
- OpenCV's built-in Gaussian blurring

### Edge Detection

Edge detection highlights the boundaries within images. This project includes the following edge detection methods:

#### Sobel Edge Detection

Uses the Sobel operator to calculate the gradient of the image intensity at each pixel, highlighting regions of high spatial frequency that correspond to edges.

#### Zero Crossing Simple Edge Detection

Detects edges by finding zero crossings in the Laplacian of the image. This method is sensitive to noise and can detect edges in images with varying intensity.

#### Zero Crossing LOG Edge Detection

Applies a Laplacian of Gaussian (LOG) filter to the image and then finds zero crossings. This method combines Gaussian smoothing with Laplacian edge detection to reduce noise.

#### Canny Edge Detection

A multi-stage algorithm to detect a wide range of edges in images, involving noise reduction, gradient calculation, non-maximum suppression, double thresholding, and edge tracking by hysteresis.

### Hough Circle Detection

Detects circles in an image using the Hough Transform. This method can identify circular shapes by transforming points in the image space to the parameter space and finding intersections that correspond to circles.

## Dependencies

- Python 3.x
- OpenCV
- Matplotlib

To install the required dependencies, run:
```bash
pip install opencv-python matplotlib
```

## Usage

1. **Blurring Demo**: Demonstrates blurring using custom and OpenCV implementations.
2. **Edge Detection Demo**: Demonstrates edge detection using various methods.
3. **Hough Circle Detection Demo**: Demonstrates circle detection using the Hough Transform.

