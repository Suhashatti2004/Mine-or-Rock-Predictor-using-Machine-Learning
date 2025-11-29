# Mine-or-Rock-Predictor-using-Machine-Learning
This project is a beginner-friendly machine learning model that classifies sonar signals as either Mine (M) or Rock (R) using the Sonar Mines vs Rocks dataset. It applies Logistic Regression to analyze underwater acoustic signals and predict whether the object detected by sonar is dangerous (mine) or harmless (rock).

🚀 Project Overview
This project reads sonar signal data (60 numeric features per sample), trains a Logistic Regression classifier, evaluates its accuracy, and predicts the category of new input samples. It is ideal for beginners learning:
How to work with datasets
Preprocessing data
Training ML models
Making predictions
Evaluating model performance

🧠 Technologies Used
Python
NumPy
Pandas
Scikit-Learn
Logistic Regression

📂 Project Structure
├── sonar_data.csv          # Dataset (not included in repo)
├── Machine_pro_1.py        # Main Python script
└── README.md               # Project description

📊 Features
✔ Loads and processes sonar dataset
✔ Splits data into training and testing sets
✔ Trains Logistic Regression model
✔ Evaluates accuracy
✔ Accepts custom input for prediction
✔ Outputs whether the object is a Mine or Rock

🔍 Prediction Example
======= FINAL PREDICTION =======
Predicted class: M
Result → The object is predicted to be a **MINE**.
Confidence (Mine, Rock): [0.52 0.48]
================================

🎯 Project Goal
The goal of this project is to help beginners understand how machine learning can be applied to real-world problems like underwater object detection using sonar signals.

📘 How to Run
Clone the repository
Install dependencies
pip install numpy pandas scikit-learn
Update the dataset path in the script
Run the script
python Machine_pro_1.py

🤝 Contributions
Contributions, improvements, and suggestions are welcome!

⭐ Show Your Support

If you like this project, consider giving it a ⭐ on GitHub!
