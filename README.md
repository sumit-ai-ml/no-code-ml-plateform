# 🚀 No-Code ML Platform

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.32.0-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-v1.4.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A powerful, user-friendly machine learning platform that enables data analysis and model training without writing a single line of code.

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation)

</div>

## 🌟 Features

### 📊 Data Analysis
- **File Upload Support**
  - CSV files (.csv)
  - Excel files (.xlsx, .xls)
- **Exploratory Data Analysis (EDA)**
  - Statistical summaries
  - Data visualization
  - Correlation analysis
- **Categorical Data Analysis**
  - Value counts
  - Distribution plots
  - Frequency analysis

### 🔧 Data Preprocessing
- **Missing Value Handling**
  - Mean/Median/Mode imputation
  - Custom value imputation
- **Categorical Encoding**
  - One-Hot Encoding
  - Label Encoding
- **Feature Scaling**
  - StandardScaler
  - MinMaxScaler

### 🤖 Machine Learning
- **Classification Models**
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - AdaBoost
  - SVM
  - Decision Tree
  - K-Nearest Neighbors
  - Naive Bayes (Gaussian, Multinomial, Bernoulli)

- **Regression Models**
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Elastic Net
  - Random Forest
  - Gradient Boosting
  - AdaBoost
  - SVM
  - Decision Tree
  - K-Nearest Neighbors

### 📈 Model Evaluation
- **Validation Methods**
  - Train-Test Split (80:20)
  - 5-Fold Cross Validation
- **Performance Metrics**
  - Classification: Accuracy, Precision, Recall, F1 Score
  - Regression: R² Score, MSE, RMSE, MAE
- **Visualizations**
  - ROC Curves
  - Confusion Matrices
  - Actual vs Predicted Plots

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sumit-ai-ml/no-code-ml-plateform.git
   cd no-code-ml-plateform
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Navigate the Interface**
   - Use the sidebar to access different features
   - Upload your dataset through the main page
   - Follow the intuitive workflow for analysis and modeling

3. **Workflow**
   ```
   Data Upload → EDA → Preprocessing → Model Training → Evaluation
   ```

## 📚 Documentation

### Data Upload
- Supported formats: CSV, Excel
- Maximum file size: 200MB
- Encoding: UTF-8

### Data Preprocessing
- Automatic detection of data types
- Smart handling of missing values
- One-click encoding of categorical variables

### Model Training
- Automatic problem type detection
- Hyperparameter tuning options
- Model comparison capabilities

### Visualization
- Interactive plots
- Downloadable results
- Customizable parameters

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Streamlit for the amazing web framework
- Scikit-learn for the machine learning algorithms
- Plotly for interactive visualizations

## 📧 Contact

Sumit Pandey - [@sumit-ai-ml](https://github.com/sumit-ai-ml)

Project Link: [https://github.com/sumit-ai-ml/no-code-ml-plateform](https://github.com/sumit-ai-ml/no-code-ml-plateform)

---

<div align="center">
Made with ❤️ by Sumit Pandey
</div> 