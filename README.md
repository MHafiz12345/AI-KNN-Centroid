# 🧠 Stroke Prediction with Custom KNN and Centroid-Based Models

This project implements a stroke prediction system using a dataset of patient health records. The core objective is to predict the likelihood of a stroke using distance-based classifiers — all manually implemented without relying on external ML libraries.

The project walks through the full ML pipeline: data loading, manual preprocessing, feature engineering, and evaluation using custom classifiers (KNN and Centroid Classifier) with various distance metrics.

## 📂 Dataset

The dataset is publicly available on Kaggle:  
👉 [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download)

Each row represents an individual with features like:

- `gender`, `age`, `hypertension`, `heart_disease`
- `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`
- `bmi`, `smoking_status`
- `stroke` (Target: 1 = Stroke, 0 = No Stroke)

## ⚙️ Manual Data Preprocessing

The dataset is cleaned and preprocessed **without using any external ML preprocessing libraries**. All steps are coded manually:

### 🔸 Step-by-step Processing:

1. **Invalid Record Removal**: Rows with `gender = Other` are dropped
2. **Missing BMI Handling**: Missing `bmi` values are filled with the manually computed mean
3. **Categorical Encoding**: All categorical features are manually one-hot encoded using custom logic
4. **Column Renaming**: Some binary columns are renamed for better clarity (e.g., `hypertension_0` → `no_hypertension`)
5. **Feature Scaling**: Numerical features (`age`, `bmi`, `avg_glucose_level`) are manually standardized using:

   ![z = \frac{x - \mu}{\sigma}](https://latex.codecogs.com/svg.latex?z%20%3D%20%5Cfrac%7Bx%20-%20%5Cmu%7D%7B%5Csigma%7D)

6. **ID Column Dropped**: The `id` column is removed to avoid data leakage

The processed data is saved as `stroke_data_processed.csv`.

## 🔢 Distance Metrics Used

Multiple distance metrics are implemented from scratch to explore their effect on classification performance:

### 📏 Euclidean Distance

![Euclidean Distance Formula](https://github.com/user-attachments/assets/a409fb38-71d3-4ee3-830c-7b5b5480ed32)

### 📏 Cosine Similarity (converted to distance via sorting)

![Cosine Similarity Formula](https://github.com/user-attachments/assets/19ff55e4-7a1d-4735-8573-1f06b3f2cc8b)

### 📏 Mahalanobis Distance

![Mahalanobis Distance Formula](https://github.com/user-attachments/assets/a97f3e87-ddd2-4cca-a8d3-0e8f7aa22a5f)

## 🧠 Models Implemented

### ✅ K-Nearest Neighbors (KNN)

KNN is implemented using all three distance metrics:

- `KNN_euclidean`
- `KNN_cosine`
- `KNN_mahalanobis`

Prediction is done using **majority voting** from the top K nearest neighbors.

### ✅ Centroid-Based Classifier

This model calculates the centroid of each class and assigns new samples based on the **nearest centroid** using:

- Euclidean Distance
- Cosine Similarity
- Mahalanobis Distance

## 🔁 Cross Validation

To evaluate model performance:

- **5-Fold Cross Validation** is implemented from scratch
- Dataset is split into K folds
- Each model is trained and tested across all folds
- Metrics used:
  - **Accuracy**
  - **Macro-averaged Precision**

## 📈 Example Outputs

During cross-validation, outputs show accuracy and precision per fold:

```
KNN (Cosine) With K = 3
Fold 1 Accuracy: 0.9430 | Precision: 0.7689
Fold 2 Accuracy: 0.9517 | Precision: 0.7924
...
```

This helps identify the best combination of distance metric and model type.

## 🛠️ How to Run

1. Clone the repo and download the dataset from Kaggle
2. Place the CSV file in the expected directory (`My Drive` if running in Colab)
3. Run the preprocessing notebook to generate `stroke_data_processed.csv`
4. Run the model training notebook

Notebooks:
- `Manual Data Preprocessing.ipynb`
- `KNN & Centroid - Manual.ipynb`

## 🧪 Tested Models Overview

| Model                     | Distance    | Cross-Validation | Notes                             |
|---------------------------|-------------|------------------|-----------------------------------|
| KNN (k=3,5,7)             | Euclidean   | ✅               | Manually implemented              |
| KNN (k=3,5,7)             | Cosine      | ✅               | Works well with standardized features |
| KNN (k=3,5,7)             | Mahalanobis | ✅               | Sensitive to covariance           |
| Centroid Classifier       | Euclidean   | ✅               | Fast and simple                   |
| Centroid Classifier       | Cosine      | ✅               | Good when angle matters           |
| Centroid Classifier       | Mahalanobis | ✅               | Adds context from data spread     |

## 🧰 Requirements

- Python (3.6+)
- NumPy
- pandas
- matplotlib
- seaborn
- Google Colab (for Drive access)
