# Wine Quality Classification

This repository contains a machine learning project for classifying wine quality based on various physicochemical properties. The dataset used is from the University of California, Irvine, and can be found on Kaggle.

## Dataset

The dataset is sourced from Kaggle:
- [Red Wine Quality Dataset](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009)

The dataset includes features such as acidity, alcohol content, and more, with a target variable indicating wine quality.

## Project Overview

The project involves several key steps:
1. **Data Exploration**: Visualize and understand the relationships between features.
   - Use `sns.pairplot` to visualize feature relationships and determine which features might be informative for classification.

2. **Train-Test Split**: Divide the dataset into training and testing sets to evaluate model performance.

3. **Scaling**: Standardize the dataset using `StandardScaler` to ensure all features are on a similar scale.

4. **Modeling**: Apply various classifiers including K-Nearest Neighbors (KNN) and Random Forest.
   - Use `KNeighborsClassifier` for KNN classification.
   - Use `RandomForestClassifier` and perform hyperparameter optimization with `GridSearchCV`.

5. **Hyperparameter Optimization**: Optimize model parameters using `GridSearchCV` to find the best combination of hyperparameters.
   - Test different values for `n_estimators`, `max_features`, and `bootstrap` parameters for the Random Forest model.

6. **Multi-Class Classification**: Classify wine into categories: "Bad", "Normal", and "Good".
   - Create a mapping from quality scores to these categories.

7. **Multi-Label Classification**: Create a multi-label classifier for alcohol content and wine quality.
   - Use binary classification for each label and compute confusion matrices for evaluation.

8. **Evaluation**: Use various metrics to evaluate model performance:
   - **Accuracy Metrics**: Compute accuracy, recall, precision, and confusion matrices for classification models.
   - **ROC Curve**: Plot ROC curves to evaluate binary classification performance.

## Files

- `data_import.ipynb`: Jupyter notebook for data exploration, preprocessing, and modeling.
- `requirements.txt`: List of Python dependencies required to run the project.
- `README.md`: Documentation for the project.

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/asparmar14/Classification-task

2. **Navigate to the project directory:**
   ```bash
   cd wine-quality-classification
3. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook data_import.ipynb

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UCI Machine Learning Repository
- Kaggle
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurélien Géron
