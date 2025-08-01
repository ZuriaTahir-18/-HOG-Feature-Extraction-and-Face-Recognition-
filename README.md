# HOG Feature Extraction and Face Detection in Python

This project demonstrates how to extract **Histogram of Oriented Gradients (HOG)** features from images and use them for **template matching** and **face detection**. It also includes comparison with **Haar Cascade** detection for robust results.

## Objective

To implement a custom pipeline that extracts HOG features and detects similar regions using template matching. It also includes face detection using OpenCV's built-in Haar cascade classifier and comparison through visualization.

## Key Features

- Custom implementation of:
  - Gradient filters (Sobel)
  - Gradient magnitude and orientation
  - Orientation histograms
  - Block normalization
  - HOG descriptor
- Template matching via sliding window and cosine similarity
- Non-maximum suppression to reduce overlaps
- Face detection using Haar Cascade classifier
- Side-by-side visualization of results

## File Structure

```text
├── hog.py               # Main script containing all functions and execution flow
├── target.png           # Target image (grayscale)
├── template.png         # Template image for matching
├── Cameraman.tif        # Sample image to demonstrate HOG extraction
├── README.md            # You're reading this!
