EMG Signal Classification for Prosthetic Control 🤖
📌 Project Overview

This project focuses on classifying EMG (Electromyography) signals to recognize hand gestures such as Open, Close, and Grip. This is useful for prosthetic hand control systems.

📂 Dataset
Source: Kaggle
Dataset Name: EMG Dataset for Hand Gesture Recognition
Features: EMG signal channels (channel1–channel8)
Target: Gesture label

⚙️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

🔄 Project Workflow
Load dataset
Preprocess data (cleaning, normalization)
Split dataset into training and testing
Train model (Random Forest)
Evaluate model performance
Generate confusion matrix
Predict gesture from sample input

🤖 Model Used
Random Forest Classifier

📊 Results
Accuracy: ~18–20%
Dataset contains multiple gesture classes (0–35), making classification complex

📈 Output
Confusion Matrix Visualization
Classification Report
Sample Prediction

🚀 How to Run
pip install pandas numpy matplotlib seaborn scikit-learn
python emg_model.py

📌 Conclusion

The model successfully classifies EMG signals, though accuracy is moderate due to complex multi-class data. This project demonstrates the application of machine learning in biomedical signal processing.

👩‍💻 Author

Revathi Guggilla
