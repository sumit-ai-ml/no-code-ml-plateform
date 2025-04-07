import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score,
    roc_curve, auc, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, mean_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns
import time

st.set_page_config(page_title="Model Training - No-Code ML Platform", layout="wide")

# Check if data is available in session state
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("Please upload a dataset in the main page first.")
    st.stop()

def plot_training_history(history, metric_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history['train'], name='Training'))
    fig.add_trace(go.Scatter(y=history['val'], name='Validation'))
    fig.update_layout(
        title=f'Training and Validation {metric_name}',
        xaxis_title='Epoch',
        yaxis_title=metric_name
    )
    st.plotly_chart(fig)

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash')))
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=700
    )
    st.plotly_chart(fig)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Negative', 'Positive'],
                    y=['Negative', 'Positive'],
                    title="Confusion Matrix")
    st.plotly_chart(fig)

def get_model(problem_type, model_type, hyperparameters=None):
    if hyperparameters is None:
        hyperparameters = {}
        
    if problem_type == "Classification":
        if model_type == "Logistic Regression":
            return LogisticRegression(max_iter=1000, **hyperparameters)
        elif model_type == "Random Forest":
            return RandomForestClassifier(**hyperparameters)
        elif model_type == "Gradient Boosting":
            return GradientBoostingClassifier(**hyperparameters)
        elif model_type == "AdaBoost":
            return AdaBoostClassifier(**hyperparameters)
        elif model_type == "SVM":
            return SVC(probability=True, **hyperparameters)
        elif model_type == "Decision Tree":
            return DecisionTreeClassifier(**hyperparameters)
        elif model_type == "K-Nearest Neighbors":
            return KNeighborsClassifier(**hyperparameters)
        elif model_type == "Gaussian Naive Bayes":
            return GaussianNB(**hyperparameters)
        elif model_type == "Multinomial Naive Bayes":
            return MultinomialNB(**hyperparameters)
        elif model_type == "Bernoulli Naive Bayes":
            return BernoulliNB(**hyperparameters)
    else:  # Regression
        if model_type == "Linear Regression":
            return LinearRegression(**hyperparameters)
        elif model_type == "Ridge Regression":
            return Ridge(**hyperparameters)
        elif model_type == "Lasso Regression":
            return Lasso(**hyperparameters)
        elif model_type == "Elastic Net":
            return ElasticNet(**hyperparameters)
        elif model_type == "Random Forest":
            return RandomForestRegressor(**hyperparameters)
        elif model_type == "Gradient Boosting":
            return GradientBoostingRegressor(**hyperparameters)
        elif model_type == "AdaBoost":
            return AdaBoostRegressor(**hyperparameters)
        elif model_type == "SVM":
            return SVR(**hyperparameters)
        elif model_type == "Decision Tree":
            return DecisionTreeRegressor(**hyperparameters)
        elif model_type == "K-Nearest Neighbors":
            return KNeighborsRegressor(**hyperparameters)

def get_hyperparameters(problem_type, model_type):
    hyperparameters = {}
    
    if problem_type == "Classification":
        if model_type == "Logistic Regression":
            hyperparameters["C"] = st.slider("Regularization strength (C)", 0.1, 10.0, 1.0)
            hyperparameters["solver"] = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
        elif model_type == "Random Forest":
            hyperparameters["n_estimators"] = st.slider("Number of trees", 10, 200, 100)
            hyperparameters["max_depth"] = st.slider("Maximum depth", 1, 20, 5)
        elif model_type == "Gradient Boosting":
            hyperparameters["n_estimators"] = st.slider("Number of trees", 10, 200, 100)
            hyperparameters["learning_rate"] = st.slider("Learning rate", 0.01, 0.5, 0.1)
        elif model_type == "AdaBoost":
            hyperparameters["n_estimators"] = st.slider("Number of estimators", 10, 200, 50)
            hyperparameters["learning_rate"] = st.slider("Learning rate", 0.01, 0.5, 0.1)
        elif model_type == "SVM":
            hyperparameters["C"] = st.slider("Regularization parameter (C)", 0.1, 10.0, 1.0)
            hyperparameters["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        elif model_type == "Decision Tree":
            hyperparameters["max_depth"] = st.slider("Maximum depth", 1, 20, 5)
            hyperparameters["min_samples_split"] = st.slider("Minimum samples split", 2, 20, 2)
        elif model_type == "K-Nearest Neighbors":
            hyperparameters["n_neighbors"] = st.slider("Number of neighbors", 1, 20, 5)
            hyperparameters["weights"] = st.selectbox("Weights", ["uniform", "distance"])
    else:  # Regression
        if model_type == "Linear Regression":
            pass  # No hyperparameters
        elif model_type == "Ridge Regression":
            hyperparameters["alpha"] = st.slider("Alpha (regularization strength)", 0.1, 10.0, 1.0)
        elif model_type == "Lasso Regression":
            hyperparameters["alpha"] = st.slider("Alpha (regularization strength)", 0.1, 10.0, 1.0)
        elif model_type == "Elastic Net":
            hyperparameters["alpha"] = st.slider("Alpha (regularization strength)", 0.1, 10.0, 1.0)
            hyperparameters["l1_ratio"] = st.slider("L1 ratio", 0.0, 1.0, 0.5)
        elif model_type == "Random Forest":
            hyperparameters["n_estimators"] = st.slider("Number of trees", 10, 200, 100)
            hyperparameters["max_depth"] = st.slider("Maximum depth", 1, 20, 5)
        elif model_type == "Gradient Boosting":
            hyperparameters["n_estimators"] = st.slider("Number of trees", 10, 200, 100)
            hyperparameters["learning_rate"] = st.slider("Learning rate", 0.01, 0.5, 0.1)
        elif model_type == "AdaBoost":
            hyperparameters["n_estimators"] = st.slider("Number of estimators", 10, 200, 50)
            hyperparameters["learning_rate"] = st.slider("Learning rate", 0.01, 0.5, 0.1)
        elif model_type == "SVM":
            hyperparameters["C"] = st.slider("Regularization parameter (C)", 0.1, 10.0, 1.0)
            hyperparameters["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        elif model_type == "Decision Tree":
            hyperparameters["max_depth"] = st.slider("Maximum depth", 1, 20, 5)
            hyperparameters["min_samples_split"] = st.slider("Minimum samples split", 2, 20, 2)
        elif model_type == "K-Nearest Neighbors":
            hyperparameters["n_neighbors"] = st.slider("Number of neighbors", 1, 20, 5)
            hyperparameters["weights"] = st.selectbox("Weights", ["uniform", "distance"])
    
    return hyperparameters

def train_model(X, y, problem_type, model_type, validation_method, n_splits=5, hyperparameters=None):
    if validation_method == "Train-Test Split":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if problem_type == "Classification" else None
        )
    else:  # Cross Validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
    
    scaler = StandardScaler()
    
    # Get the model with the selected hyperparameters
    model = get_model(problem_type, model_type, hyperparameters)
    
    if problem_type == "Classification":
        if validation_method == "Train-Test Split":
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            st.success(f"Accuracy: {accuracy:.4f}")
            st.success(f"Precision: {precision:.4f}")
            st.success(f"Recall: {recall:.4f}")
            st.success(f"F1 Score: {f1:.4f}")
            
            # Plot ROC curve
            plot_roc_curve(y_test, y_pred_proba)
            
            # Plot confusion matrix
            plot_confusion_matrix(y_test, y_pred)
            
            # Classification report
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            
            return model, accuracy, "Accuracy"
        else:
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                model.fit(X_train_scaled, y_train)
                score = model.score(X_val_scaled, y_val)
                cv_scores.append(score)
            
            mean_cv_score = np.mean(cv_scores)
            st.success(f"Cross-validation Accuracy: {mean_cv_score:.4f} (±{np.std(cv_scores):.4f})")
            return model, mean_cv_score, "CV Accuracy"
    
    else:  # Regression
        if validation_method == "Train-Test Split":
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.success(f"R² Score: {r2:.4f}")
            st.success(f"MSE: {mse:.4f}")
            st.success(f"RMSE: {rmse:.4f}")
            st.success(f"MAE: {mae:.4f}")
            
            # Plot actual vs predicted
            fig = px.scatter(x=y_test, y=y_pred, 
                           title="Actual vs Predicted Values",
                           labels={'x': 'Actual', 'y': 'Predicted'})
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                   y=[y_test.min(), y_test.max()],
                                   mode='lines', name='Perfect Prediction'))
            st.plotly_chart(fig)
            
            return model, r2, "R² Score"
        else:
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                model.fit(X_train_scaled, y_train)
                score = model.score(X_val_scaled, y_val)
                cv_scores.append(score)
            
            mean_cv_score = np.mean(cv_scores)
            st.success(f"Cross-validation R² Score: {mean_cv_score:.4f} (±{np.std(cv_scores):.4f})")
            return model, mean_cv_score, "CV R² Score"

def compare_all_models(X, y, problem_type, validation_method):
    # Define all models to compare
    if problem_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "SVM": SVC(probability=True),
            "Decision Tree": DecisionTreeClassifier(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Gaussian Naive Bayes": GaussianNB()
        }
        
        # Special models that need different preprocessing
        special_models = {
            "Multinomial Naive Bayes": MultinomialNB(),
            "Bernoulli Naive Bayes": BernoulliNB()
        }
    else:  # Regression
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Elastic Net": ElasticNet(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            "SVM": SVR(),
            "Decision Tree": DecisionTreeRegressor(),
            "K-Nearest Neighbors": KNeighborsRegressor()
        }
        special_models = {}
    
    # Initialize results dictionary
    results = {}
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Prepare data for standard models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train and evaluate standard models
    for i, (model_name, model) in enumerate(models.items()):
        status_text.text(f"Training {model_name}...")
        
        try:
            if validation_method == "Train-Test Split":
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, 
                    stratify=y if problem_type == "Classification" else None
                )
                
                model.fit(X_train, y_train)
                
                # Get training metrics
                if problem_type == "Classification":
                    y_train_pred = model.predict(X_train)
                    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
                    
                    train_accuracy = accuracy_score(y_train, y_train_pred)
                    train_precision = precision_score(y_train, y_train_pred, average='weighted')
                    train_recall = recall_score(y_train, y_train_pred, average='weighted')
                    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                    
                    # Get validation metrics
                    y_test_pred = model.predict(X_test)
                    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    test_accuracy = accuracy_score(y_test, y_test_pred)
                    test_precision = precision_score(y_test, y_test_pred, average='weighted')
                    test_recall = recall_score(y_test, y_test_pred, average='weighted')
                    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
                    
                    results[model_name] = {
                        "score": test_accuracy,
                        "metric": "Accuracy",
                        "details": {
                            "train": {
                                "accuracy": train_accuracy,
                                "precision": train_precision,
                                "recall": train_recall,
                                "f1": train_f1,
                                "y_true": y_train,
                                "y_pred": y_train_pred,
                                "y_pred_proba": y_train_pred_proba
                            },
                            "val": {
                                "accuracy": test_accuracy,
                                "precision": test_precision,
                                "recall": test_recall,
                                "f1": test_f1,
                                "y_true": y_test,
                                "y_pred": y_test_pred,
                                "y_pred_proba": y_test_pred_proba
                            }
                        }
                    }
                else:
                    # Get training metrics
                    y_train_pred = model.predict(X_train)
                    train_r2 = r2_score(y_train, y_train_pred)
                    train_mse = mean_squared_error(y_train, y_train_pred)
                    train_rmse = np.sqrt(train_mse)
                    train_mae = mean_absolute_error(y_train, y_train_pred)
                    
                    # Get validation metrics
                    y_test_pred = model.predict(X_test)
                    test_r2 = r2_score(y_test, y_test_pred)
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    test_rmse = np.sqrt(test_mse)
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    
                    results[model_name] = {
                        "score": test_r2,
                        "metric": "R² Score",
                        "details": {
                            "train": {
                                "r2": train_r2,
                                "mse": train_mse,
                                "rmse": train_rmse,
                                "mae": train_mae,
                                "y_true": y_train,
                                "y_pred": y_train_pred
                            },
                            "val": {
                                "r2": test_r2,
                                "mse": test_mse,
                                "rmse": test_rmse,
                                "mae": test_mae,
                                "y_true": y_test,
                                "y_pred": y_test_pred
                            }
                        }
                    }
            else:  # Cross Validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=5)
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                
                results[model_name] = {
                    "score": mean_score,
                    "std": std_score,
                    "metric": "CV Score"
                }
        except Exception as e:
            st.warning(f"Error training {model_name}: {str(e)}")
            results[model_name] = {
                "score": 0,
                "metric": "Error"
            }
        
        # Update progress
        progress_bar.progress((i + 1) / (len(models) + len(special_models)))
    
    # Train and evaluate special models (if any)
    if special_models:
        # For MultinomialNB and BernoulliNB, we need to ensure non-negative values
        # We'll use MinMaxScaler instead of StandardScaler
        from sklearn.preprocessing import MinMaxScaler
        minmax_scaler = MinMaxScaler()
        X_minmax = minmax_scaler.fit_transform(X)
        
        for i, (model_name, model) in enumerate(special_models.items()):
            status_text.text(f"Training {model_name}...")
            
            try:
                if validation_method == "Train-Test Split":
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_minmax, y, test_size=0.2, random_state=42, 
                        stratify=y if problem_type == "Classification" else None
                    )
                    
                    model.fit(X_train, y_train)
                    
                    if problem_type == "Classification":
                        # Get training metrics
                        y_train_pred = model.predict(X_train)
                        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
                        
                        train_accuracy = accuracy_score(y_train, y_train_pred)
                        train_precision = precision_score(y_train, y_train_pred, average='weighted')
                        train_recall = recall_score(y_train, y_train_pred, average='weighted')
                        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                        
                        # Get validation metrics
                        y_test_pred = model.predict(X_test)
                        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
                        
                        test_accuracy = accuracy_score(y_test, y_test_pred)
                        test_precision = precision_score(y_test, y_test_pred, average='weighted')
                        test_recall = recall_score(y_test, y_test_pred, average='weighted')
                        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
                        
                        results[model_name] = {
                            "score": test_accuracy,
                            "metric": "Accuracy",
                            "details": {
                                "train": {
                                    "accuracy": train_accuracy,
                                    "precision": train_precision,
                                    "recall": train_recall,
                                    "f1": train_f1,
                                    "y_true": y_train,
                                    "y_pred": y_train_pred,
                                    "y_pred_proba": y_train_pred_proba
                                },
                                "val": {
                                    "accuracy": test_accuracy,
                                    "precision": test_precision,
                                    "recall": test_recall,
                                    "f1": test_f1,
                                    "y_true": y_test,
                                    "y_pred": y_test_pred,
                                    "y_pred_proba": y_test_pred_proba
                                }
                            }
                        }
                    else:
                        # Get training metrics
                        y_train_pred = model.predict(X_train)
                        train_r2 = r2_score(y_train, y_train_pred)
                        train_mse = mean_squared_error(y_train, y_train_pred)
                        train_rmse = np.sqrt(train_mse)
                        train_mae = mean_absolute_error(y_train, y_train_pred)
                        
                        # Get validation metrics
                        y_test_pred = model.predict(X_test)
                        test_r2 = r2_score(y_test, y_test_pred)
                        test_mse = mean_squared_error(y_test, y_test_pred)
                        test_rmse = np.sqrt(test_mse)
                        test_mae = mean_absolute_error(y_test, y_test_pred)
                        
                        results[model_name] = {
                            "score": test_r2,
                            "metric": "R² Score",
                            "details": {
                                "train": {
                                    "r2": train_r2,
                                    "mse": train_mse,
                                    "rmse": train_rmse,
                                    "mae": train_mae,
                                    "y_true": y_train,
                                    "y_pred": y_train_pred
                                },
                                "val": {
                                    "r2": test_r2,
                                    "mse": test_mse,
                                    "rmse": test_rmse,
                                    "mae": test_mae,
                                    "y_true": y_test,
                                    "y_pred": y_test_pred
                                }
                            }
                        }
                else:  # Cross Validation
                    cv_scores = cross_val_score(model, X_minmax, y, cv=5)
                    mean_score = np.mean(cv_scores)
                    std_score = np.std(cv_scores)
                    
                    results[model_name] = {
                        "score": mean_score,
                        "std": std_score,
                        "metric": "CV Score"
                    }
            except Exception as e:
                st.warning(f"Error training {model_name}: {str(e)}")
                results[model_name] = {
                    "score": 0,
                    "metric": "Error"
                }
            
            # Update progress
            progress_bar.progress((len(models) + i + 1) / (len(models) + len(special_models)))
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    st.subheader("Model Comparison Results")
    
    # Create a DataFrame for results
    if validation_method == "Train-Test Split":
        results_df = pd.DataFrame({
            "Model": list(results.keys()),
            "Score": [results[model]["score"] for model in results],
            "Metric": [results[model]["metric"] for model in results]
        })
    else:
        results_df = pd.DataFrame({
            "Model": list(results.keys()),
            "Score": [results[model]["score"] for model in results],
            "Std": [results[model]["std"] for model in results],
            "Metric": [results[model]["metric"] for model in results]
        })
    
    # Sort by score (descending)
    results_df = results_df.sort_values("Score", ascending=False)
    
    # Display results table
    st.dataframe(results_df)
    
    # Plot results
    if validation_method == "Train-Test Split":
        fig = px.bar(results_df, x="Model", y="Score", 
                    title=f"Model Comparison by {results_df['Metric'].iloc[0]}",
                    labels={"Score": results_df['Metric'].iloc[0]})
    else:
        fig = px.bar(results_df, x="Model", y="Score", 
                    error_y="Std",
                    title=f"Model Comparison by {results_df['Metric'].iloc[0]}",
                    labels={"Score": results_df['Metric'].iloc[0]})
    
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)
    
    # Display detailed metrics and plots for each model
    if validation_method == "Train-Test Split":
        st.subheader("Detailed Model Performance")
        
        for model_name in results_df["Model"]:
            if "details" in results[model_name]:
                with st.expander(f"{model_name} Details"):
                    if problem_type == "Classification":
                        details = results[model_name]["details"]
                        
                        # Display training metrics
                        st.subheader("Training Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{details['train']['accuracy']:.4f}")
                        with col2:
                            st.metric("Precision", f"{details['train']['precision']:.4f}")
                        with col3:
                            st.metric("Recall", f"{details['train']['recall']:.4f}")
                        with col4:
                            st.metric("F1 Score", f"{details['train']['f1']:.4f}")
                        
                        # Display validation metrics
                        st.subheader("Validation Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{details['val']['accuracy']:.4f}")
                        with col2:
                            st.metric("Precision", f"{details['val']['precision']:.4f}")
                        with col3:
                            st.metric("Recall", f"{details['val']['recall']:.4f}")
                        with col4:
                            st.metric("F1 Score", f"{details['val']['f1']:.4f}")
                        
                        # Plot ROC curves
                        st.subheader("ROC Curves")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Training ROC Curve")
                            plot_roc_curve(details['train']['y_true'], details['train']['y_pred_proba'])
                        with col2:
                            st.write("Validation ROC Curve")
                            plot_roc_curve(details['val']['y_true'], details['val']['y_pred_proba'])
                        
                        # Plot confusion matrices
                        st.subheader("Confusion Matrices")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Training Confusion Matrix")
                            plot_confusion_matrix(details['train']['y_true'], details['train']['y_pred'])
                        with col2:
                            st.write("Validation Confusion Matrix")
                            plot_confusion_matrix(details['val']['y_true'], details['val']['y_pred'])
                        
                        # Classification reports
                        st.subheader("Classification Reports")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Training Classification Report")
                            st.text(classification_report(details['train']['y_true'], details['train']['y_pred']))
                        with col2:
                            st.write("Validation Classification Report")
                            st.text(classification_report(details['val']['y_true'], details['val']['y_pred']))
                    else:
                        details = results[model_name]["details"]
                        
                        # Display training metrics
                        st.subheader("Training Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R² Score", f"{details['train']['r2']:.4f}")
                        with col2:
                            st.metric("MSE", f"{details['train']['mse']:.4f}")
                        with col3:
                            st.metric("RMSE", f"{details['train']['rmse']:.4f}")
                        with col4:
                            st.metric("MAE", f"{details['train']['mae']:.4f}")
                        
                        # Display validation metrics
                        st.subheader("Validation Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R² Score", f"{details['val']['r2']:.4f}")
                        with col2:
                            st.metric("MSE", f"{details['val']['mse']:.4f}")
                        with col3:
                            st.metric("RMSE", f"{details['val']['rmse']:.4f}")
                        with col4:
                            st.metric("MAE", f"{details['val']['mae']:.4f}")
                        
                        # Plot actual vs predicted
                        st.subheader("Actual vs Predicted Plots")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Training Actual vs Predicted")
                            fig = px.scatter(x=details['train']['y_true'], y=details['train']['y_pred'], 
                                           title="Training: Actual vs Predicted Values",
                                           labels={'x': 'Actual', 'y': 'Predicted'})
                            fig.add_trace(go.Scatter(x=[details['train']['y_true'].min(), details['train']['y_true'].max()], 
                                                   y=[details['train']['y_true'].min(), details['train']['y_true'].max()],
                                                   mode='lines', name='Perfect Prediction'))
                            st.plotly_chart(fig)
                        with col2:
                            st.write("Validation Actual vs Predicted")
                            fig = px.scatter(x=details['val']['y_true'], y=details['val']['y_pred'], 
                                           title="Validation: Actual vs Predicted Values",
                                           labels={'x': 'Actual', 'y': 'Predicted'})
                            fig.add_trace(go.Scatter(x=[details['val']['y_true'].min(), details['val']['y_true'].max()], 
                                                   y=[details['val']['y_true'].min(), details['val']['y_true'].max()],
                                                   mode='lines', name='Perfect Prediction'))
                            st.plotly_chart(fig)
    
    # Return the best model
    best_model_name = results_df["Model"].iloc[0]
    if best_model_name in special_models:
        best_model = special_models[best_model_name]
    else:
        best_model = models[best_model_name]
    best_score = results_df["Score"].iloc[0]
    
    st.success(f"Best model: {best_model_name} with {results_df['Metric'].iloc[0]}: {best_score:.4f}")
    
    return best_model, best_score, results_df['Metric'].iloc[0]

def main():
    st.title("Model Training")
    
    # Get the dataframe from session state
    df = st.session_state.df
    
    # Problem Type Selection
    problem_type = st.selectbox("Select problem type", ["Classification", "Regression"])
    
    # Feature Selection
    st.subheader("Select Features")
    feature_cols = st.multiselect("Select feature columns", df.columns)
    
    target_col = st.selectbox("Select target column", df.columns)
    
    # Validation Method Selection
    validation_method = st.radio(
        "Select validation method",
        ["Train-Test Split (80:20)", "5-Fold Cross Validation"]
    )
    
    # Model Selection
    if problem_type == "Classification":
        model_options = [
            "Logistic Regression", "Random Forest", "Gradient Boosting", 
            "AdaBoost", "SVM", "Decision Tree", "K-Nearest Neighbors",
            "Gaussian Naive Bayes", "Multinomial Naive Bayes", "Bernoulli Naive Bayes"
        ]
    else:  # Regression
        model_options = [
            "Linear Regression", "Ridge Regression", "Lasso Regression", 
            "Elastic Net", "Random Forest", "Gradient Boosting", 
            "AdaBoost", "SVM", "Decision Tree", "K-Nearest Neighbors"
        ]
    
    # Train Single Model Section
    st.subheader("Train Single Model")
    
    model_type = st.selectbox("Select model type", model_options)
    
    # Get hyperparameters for the selected model
    hyperparameters = get_hyperparameters(problem_type, model_type)
    
    if st.button("Train Selected Model"):
        if len(feature_cols) > 0:
            X = df[feature_cols]
            y = df[target_col]
            
            model, metric_value, metric_name = train_model(
                X, y, problem_type, model_type,
                "Train-Test Split" if validation_method == "Train-Test Split (80:20)" else "Cross Validation",
                hyperparameters=hyperparameters
            )
            
            if metric_name:
                st.success(f"Model trained successfully! {metric_name}: {metric_value:.4f}")
        else:
            st.error("Please select at least one feature column.")
    
    # Compare All Models Section
    st.subheader("Compare All Models")
    
    if st.button("Compare All Models"):
        if len(feature_cols) > 0:
            X = df[feature_cols]
            y = df[target_col]
            
            best_model, best_score, metric_name = compare_all_models(
                X, y, problem_type,
                "Train-Test Split" if validation_method == "Train-Test Split (80:20)" else "Cross Validation"
            )
            
            st.success(f"Best model: {best_model.__class__.__name__} with {metric_name}: {best_score:.4f}")
        else:
            st.error("Please select at least one feature column.")

if __name__ == "__main__":
    main() 