# Diabetes Prediction using Neural Network

This project builds a neural network model to predict whether a patient is diabetic or not using the popular **Pima Indians Diabetes Dataset**. It includes all essential machine learning steps such as data cleaning, visualization, training, evaluation, and hyperparameter tuning.

## Project Highlights

- Data preprocessing and scaling
- Visualization: heatmaps and class distributions
- Multiple neural network architectures (small, large, regularized)
- Dropout & L2 regularization to prevent overfitting
- Training & validation loss plots
- Hyperparameter tuning with GridSearchCV
- Model evaluation using Accuracy, F1, ROC-AUC
- Saved final model (`finalmodel_2.h5`) and scaler (`scaler_2.pkl`)

---

## Setup Instructions

1. **Clone this repository:**
```bash
git clone https://github.com/yourusername/diabetes-prediction-nn.git
cd diabetes-prediction-nn
```

2. **Install the required Python libraries:**
```bash
pip install -r requirements.txt
```

## How to Run the Model

1. **Run the notebook:**
   - Open `diabetes_prediction` or `final_model_2.ipynb` in Jupyter Notebook or VS Code and run all cells step by step.

2. **Evaluate on test set:**
   - At the end of the notebook, the model will be evaluated using accuracy, confusion matrix, and ROC curve.

3. **Saved files:**
   - `finalmodel_2.h5`: the trained Keras model
   - `scaler_2.pkl`: the fitted StandardScaler for preprocessing

To use the model in production, simply load both the `.h5` model and `scaler_2.pkl` file and feed new data.

## Files in This Repo

| File | Description |
|------|-------------|
| `diabetes_prediction` | Final notebook with complete pipeline |
| `finalmodel_2.h5` | Trained neural network model |
| `scaler_2.pkl` | Saved StandardScaler for input preprocessing |
| `requirements.txt` | List of required Python libraries |
| `README.md` | This readme file |

## License

This project is open-source and available under the MIT License.
