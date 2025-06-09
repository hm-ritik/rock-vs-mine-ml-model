# 🔍 Sonar Rock vs Mine Classification Model

This project is a Machine Learning-based classification model that predicts whether 
an object detected by sonar is a **Rock** or a **Mine**, using the **Sonar dataset** from the UCI Machine Learning Repository.

- **Algorithm Used**: Logistic Regression (Binary Classification)
- **Steps Followed**:
  1. Data Loading and Exploration
  2. Label Encoding of Target (`R`/`M`)
  3. Feature Scaling using `StandardScaler`
  4. Train-Test Split (90% Train, 10% Test)
  5. Model Training using `LogisticRegression`
  6. Performance Evaluation

---

## 📈 Performance

| Metric            | Value         |
|-------------------|---------------|
| Train Accuracy     | ~93.6%        |
| Test Accuracy      | ~85.7%        |
