# Real-time Object Detection System

This project implements a real-time object detection system using the DETR (DEtection TRansformer) model. It provides real-time object detection through your webcam with bounding boxes and confidence scores.

## Features

- Real-time object detection using webcam
- High-accuracy detection using DETR model
- Bounding box visualization with confidence scores
- Support for common objects (people, cars, animals, etc.)
- Easy-to-use interface

## Requirements

- Python 3.8 or higher
- Webcam
- CUDA-capable GPU (optional, but recommended for better performance)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/object-detection.git
cd object-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install the package in development mode:
```bash
pip install -e .
```

## Usage

1. Run the detection script:
```bash
python run_detection.py
```

2. The webcam will open and start detecting objects in real-time
3. Press 'q' to quit the detection

## Project Structure

```
object-detection/
├── run_detection.py      # Main detection script
├── train_dino.py         # Training script (if needed)
├── visualize.py          # Visualization utilities
├── requirements.txt      # Project dependencies
├── setup.py             # Package setup file
└── README.md            # This file
```

## Model Details

The project uses the DETR (DEtection TRansformer) model from Facebook Research, which is a state-of-the-art object detection model. The model is pre-trained on the COCO dataset and can detect 80 different object classes.

## Performance

The model provides:
- Real-time detection (depending on your hardware)
- High accuracy object detection
- Confidence scores for each detection
- Bounding box visualization

## Customization

You can customize the detection by:
1. Adjusting the confidence threshold in `run_detection.py`
2. Modifying the visualization colors and styles
3. Adding custom object classes (requires model fine-tuning)

## Troubleshooting

Common issues and solutions:

1. Webcam not opening:
   - Check if your webcam is properly connected
   - Try changing the camera index in `run_detection.py`

2. Slow performance:
   - Use a GPU if available
   - Reduce the input image size
   - Adjust the confidence threshold

3. Model loading errors:
   - Check your internet connection
   - Verify the model path
   - Ensure all dependencies are installed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Facebook Research for the DETR model
- Hugging Face for the Transformers library
- OpenCV for computer vision utilities 