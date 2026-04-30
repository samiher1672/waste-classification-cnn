# waste-classification-cnn
Smart Waste Classification System using CNN - Intelligent Systems Final Project

Group 7 - Intelligent Systems Final Project

## Overview
An AI-based system for classifying waste into **Glass**, **Organic**, and **Paper** categories using a Convolutional Neural Network (CNN).

## Model Performance
- **Test Accuracy**: 90.5%
- **Classes**: Glass (91.4% F1), Organic (88.4% F1), Paper (91.5% F1)

## Files
- `train_model.py` - CNN model training script
- `demo.py` - Live demo for presentation
- `waste_classifier_ui.py` - GUI application with Tkinter
- `predict.py` - Single image prediction
- `evaluate.py` - Classification report and confusion matrix
- `final_results.py` - Generate results summary
- `waste_model.h5` - Trained model file

## How to Run
1. Install dependencies: `pip install tensorflow matplotlib seaborn scikit-learn pillow`
2. Run the GUI: `python waste_classifier_ui.py`
3. Or run the demo: `python demo.py`

## Team
- Samuel Amihere - Model Design & Training
- Mickel Murenzi - Data Collection & Preprocessing Testing, Evaluation & Presentation
- Abdulazeez Dauda - Data Collection & Preprocessing
