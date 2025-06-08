# Machine Learning Project Summary:

This report details the outcomes of a team project designed to apply and evaluate various machine learning techniques. The project was divided into three distinct analyses: a classification task to predict heart disease, a regression task to model credit card balances, and an image classification task to identify fonts. The following sections outline the objectives, methods, and results for each part, as specified by the coursework requirements.

---

## Part I: Classification Analysis

### Objective

The primary goal of this analysis was to construct and evaluate a classifier to predict the presence of heart disease (AHD) in patients, based on the `Heart.csv` dataset. The project required comparing three distinct classification models to identify the one with the highest predictive accuracy.

### Data & Preprocessing

- **Dataset**: The `Heart.csv` dataset contained 303 patient observations with 14 features, including age, sex, cholesterol, and other medical measurements.
- **Cleaning**: Observations with missing values in the `Ca` and `Thal` columns were removed, resulting in a clean dataset of 297 observations.
- **Encoding & Scaling**:
  - Target variable `AHD` was label encoded ('No' = 0, 'Yes' = 1).
  - Numerical features (`RestBP`, `Chol`, `MaxHR`) were standardized.
  - Categorical features such as `ChestPain` and `Thal` were one-hot encoded.
- **Data Split**: Data was split into training (75%) and testing (25%) sets using stratification to maintain class balance.

### Methodology & Results

Three different classifiers were trained and tuned using cross-validation to maximize performance.

#### Classifiers Investigated:

- **Logistic Regression (LR)**: L2 penalty with `C=10.0`.
- **K-Nearest Neighbors (KNN)**: Optimal `k=17` using Manhattan distance.
- **Linear Discriminant Analysis (LDA)**: Implemented with the SVD solver.

#### Performance Comparison:

- **Test Accuracy**: 
  - LR: 84.0%  
  - LDA: 84.0%  
  - KNN: 82.7%
- **ROC & AUC**:
  - LDA had the highest AUC: **0.9343**
  - LR AUC: 0.9286

#### Final Prediction:

Using the LDA model, a prediction was made for a 55-year-old woman with specific medical attributes.  
**Prediction**: 'No' for heart disease  
**Probability of 'Yes'**: 4.64%

---

## Part II: Regression Analysis

### Objective

This section aimed to identify key predictors of credit card balance using the `Credit.csv` dataset. The analysis compared Lasso regression against Decision Tree and Random Forest models.

### Methodology & Results

#### 1. Lasso Regression

- **Model Setup**:
  - Included all predictors and 55 interaction terms.
  - Categorical variables were dummy-encoded; numerical predictors standardized.
- **Parameter Tuning**:
  - Optimal λ = 0.916 (via 5-fold CV), shrinking 50% of coefficients to zero.
- **Performance**:
  - Test MSE: **4,715.30**  
  - Avg prediction error: ~$68.67  
  - Predicted balance for specified individual: **$287.30**

#### 2. Tree-Based Methods

- **Decision Tree**:
  - `max_depth=3`
  - Top predictors: `Rating`, `Limit`, `Income`, `Student`
- **Random Forest**:
  - 200 trees, no depth limit
  - Test MSE: **11,964.14**
  - Predicted balance: **$613.87**

### Conclusion

Lasso regression outperformed Random Forest significantly, with a much lower test MSE (4,715.30 vs. 11,964.14). Therefore, **Lasso was selected as the superior model** for this regression task.

---

## Part III: Image Classification

### Objective

To build a robust classifier capable of identifying the font of a given text image. This was a multi-class classification task with 2,000–3,000 unique font classes.

### Data Construction & Preprocessing

- **Image Cropping**: Using a sliding window, 50 square 32x32 crops were extracted per image, generating a large labeled dataset.
- **Labeling**: Each crop was labeled with the original font identity.
- **Augmentation**: Random rotations and translations were applied to boost generalization.
- **Normalization & Splitting**:
  - Pixel values normalized.
  - Data split: 70% training, 15% validation, 15% test.

### Model & Results

- **Model Architecture**:
  - Convolutional Neural Network (CNN)
  - 3 convolutional blocks + fully connected layers
  - Dropout rate: 0.6
- **Performance**:
  - **Test Accuracy**: 80.19%
  - **AUC**: ~0.9998 — excellent class discrimination

---
