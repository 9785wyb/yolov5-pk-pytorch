# Improved YOLOv5 for Early Parkinson's Disease Detection

This repository contains an enhanced YOLOv5 model for the early diagnosis of Parkinson's disease from medical images. The improvements focus on increasing detection accuracy and reducing false positives and negatives.

## Features

- **Coordinate Attention (CA) Mechanism**: Enhances feature focus by integrating spatial information into channel attention.
- **Dynamic Depthwise Convolutions**: Improves image representation in the Neck layer.
- **Decoupled Head Layers**: Increases detection effectiveness by separating classification and localization tasks.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Improved-YOLOv5-Parkinsons.git
   cd Improved-YOLOv5-Parkinsons


Install dependencies:
pip install -r requirements.txt


Training:
python train.py --data data/your_dataset.yaml --cfg models/yolov5s.yaml --weights '' --name yolov5s_results

Evaluation：
python val.py --data data/your_dataset.yaml --weights runs/train/yolov5s_results/weights/best.pt

Model Improvements
The modified YOLOv5 model with improvements can be found in the model folder. These improvements include:

Integration of the Coordinate Attention (CA) mechanism
Use of dynamic depthwise convolutions in the Neck layer
Replacement of coupled head layers with decoupled head layers for better detection accuracy

