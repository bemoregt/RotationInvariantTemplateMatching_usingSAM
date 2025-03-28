# Rotation Invariant Template Matching using SAM

A Python application that demonstrates rotation-invariant template matching using Meta AI's Segment Anything Model (SAM).

## Overview

This application leverages the powerful segmentation capabilities of the Segment Anything Model (SAM) to perform rotation-invariant template matching for object detection in images. It provides a simple GUI interface where users can:

1. Select an object in an image by clicking on it
2. Use SAM to segment the object automatically
3. Create a template from the segmented object
4. Find similar objects in the image at various rotation angles (rotation-invariant matching)

## Features

- Interactive GUI using Tkinter
- Real-time segmentation with SAM
- Automatic template creation from segmented objects
- Rotation-invariant template matching (0-179 degrees with 1-degree intervals)
- Results visualization with bounding boxes

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- PyTorch
- Pillow (PIL)
- Meta AI's Segment Anything Model (SAM)

## Installation

1. Clone this repository:
```
git clone https://github.com/bemoregt/RotationInvariantTemplateMatching_usingSAM.git
cd RotationInvariantTemplateMatching_usingSAM
```

2. Install the required packages:
```
pip install opencv-python numpy torch torchvision pillow
```

3. Install SAM following the instructions at: https://github.com/facebookresearch/segment-anything

4. Download the SAM model checkpoint (e.g., `sam_vit_b_01ec64.pth`)

## Usage

1. Update the image path and SAM checkpoint path in the code

2. Run the application:
```
python main.py
```

3. When the GUI appears:
   - Click on an object in the image to segment it with SAM
   - Press 'm' to perform rotation-invariant template matching
   - Press 'r' to reset/reload the original image

## How It Works

1. The application initializes a window with the original image
2. When you click on an object, SAM generates a segmentation mask
3. A template is created from the segmented region
4. When you press 'm', the application:
   - Rotates the template at multiple angles (0-179°)
   - Performs template matching for each rotation
   - Filters and groups matching results
   - Draws bounding boxes around detected matches

## License

MIT

## References

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [OpenCV Template Matching](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)