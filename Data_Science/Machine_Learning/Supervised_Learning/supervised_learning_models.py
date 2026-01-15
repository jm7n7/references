"""
Supervised Learning Models in Machine Learning
This file contains template implementations for various classification and regression models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

# ============================================================================
# CLASSIFICATION MODELS
# ============================================================================

# 1. LOGISTIC REGRESSION
def logistic_regression_example(X_train, X_test, y_train, y_test):
    """
    Logistic Regression for binary and multiclass classification.
    """
    from sklearn.linear_model import LogisticRegression
    
    # Initialize and train
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'  # or 'liblinear', 'saga', etc.
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Logistic Regression - Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return model, y_pred


# 2. DECISION TREE CLASSIFIER
def decision_tree_example(X_train, X_test, y_train, y_test):
    """
    Decision Tree Classifier - non-linear, interpretable model.
    """
    from sklearn.tree import DecisionTreeClassifier
    
    # Initialize and train
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        criterion='gini'  # or 'entropy'
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree - Accuracy: {accuracy:.4f}")
    
    return model, y_pred


# 3. RANDOM FOREST CLASSIFIER
def random_forest_example(X_train, X_test, y_train, y_test):
    """
    Random Forest - ensemble of decision trees.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Initialize and train
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    feature_importance = model.feature_importances_
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest - Accuracy: {accuracy:.4f}")
    
    return model, y_pred, feature_importance


# 4. SUPPORT VECTOR MACHINE (SVM)
def svm_example(X_train, X_test, y_train, y_test):
    """
    Support Vector Machine for classification.
    """
    from sklearn.svm import SVC
    
    # Scale features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train
    model = SVC(
        kernel='rbf',  # 'linear', 'poly', 'rbf', 'sigmoid'
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM - Accuracy: {accuracy:.4f}")
    
    return model, scaler, y_pred


# 5. K-NEAREST NEIGHBORS (KNN)
def knn_example(X_train, X_test, y_train, y_test):
    """
    K-Nearest Neighbors classifier.
    """
    from sklearn.neighbors import KNeighborsClassifier
    
    # Scale features (important for KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',  # or 'distance'
        algorithm='auto',
        metric='minkowski'
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN - Accuracy: {accuracy:.4f}")
    
    return model, scaler, y_pred


# 6. NAIVE BAYES
def naive_bayes_example(X_train, X_test, y_train, y_test):
    """
    Naive Bayes classifier - good for text classification.
    """
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    
    # Gaussian Naive Bayes (for continuous features)
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Naive Bayes - Accuracy: {accuracy:.4f}")
    
    return model, y_pred


# 7. GRADIENT BOOSTING CLASSIFIER
def gradient_boosting_example(X_train, X_test, y_train, y_test):
    """
    Gradient Boosting Classifier - ensemble boosting method.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Initialize and train
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    feature_importance = model.feature_importances_
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Gradient Boosting - Accuracy: {accuracy:.4f}")
    
    return model, y_pred, feature_importance


# 8. XGBOOST CLASSIFIER
def xgboost_example(X_train, X_test, y_train, y_test):
    """
    XGBoost - optimized gradient boosting library.
    Requires: pip install xgboost
    """
    try:
        import xgboost as xgb
        
        # Initialize and train
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            eval_metric='mlogloss'
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        feature_importance = model.feature_importances_
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print(f"XGBoost - Accuracy: {accuracy:.4f}")
        
        return model, y_pred, feature_importance
    except ImportError:
        print("XGBoost not installed. Install with: pip install xgboost")
        return None, None, None


# 9. NEURAL NETWORK (MLP)
def neural_network_example(X_train, X_test, y_train, y_test):
    """
    Multi-Layer Perceptron (Neural Network) classifier.
    """
    from sklearn.neural_network import MLPClassifier
    
    # Scale features (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='constant',
        max_iter=500,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Neural Network - Accuracy: {accuracy:.4f}")
    
    return model, scaler, y_pred


# ============================================================================
# REGRESSION MODELS
# ============================================================================

# 10. LINEAR REGRESSION
def linear_regression_example(X_train, X_test, y_train, y_test):
    """
    Linear Regression for continuous target variables.
    """
    from sklearn.linear_model import LinearRegression
    
    # Initialize and train
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Linear Regression - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    return model, y_pred


# 11. RIDGE REGRESSION
def ridge_regression_example(X_train, X_test, y_train, y_test):
    """
    Ridge Regression - L2 regularization.
    """
    from sklearn.linear_model import Ridge
    
    # Initialize and train
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Ridge Regression - R²: {r2:.4f}, MSE: {mse:.4f}")
    
    return model, y_pred


# 12. LASSO REGRESSION
def lasso_regression_example(X_train, X_test, y_train, y_test):
    """
    Lasso Regression - L1 regularization (feature selection).
    """
    from sklearn.linear_model import Lasso
    
    # Initialize and train
    model = Lasso(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Lasso Regression - R²: {r2:.4f}, MSE: {mse:.4f}")
    
    return model, y_pred


# 13. DECISION TREE REGRESSOR
def decision_tree_regressor_example(X_train, X_test, y_train, y_test):
    """
    Decision Tree Regressor.
    """
    from sklearn.tree import DecisionTreeRegressor
    
    # Initialize and train
    model = DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Decision Tree Regressor - R²: {r2:.4f}, MSE: {mse:.4f}")
    
    return model, y_pred


# 14. RANDOM FOREST REGRESSOR
def random_forest_regressor_example(X_train, X_test, y_train, y_test):
    """
    Random Forest Regressor.
    """
    from sklearn.ensemble import RandomForestRegressor
    
    # Initialize and train
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    feature_importance = model.feature_importances_
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest Regressor - R²: {r2:.4f}, MSE: {mse:.4f}")
    
    return model, y_pred, feature_importance


# 15. SUPPORT VECTOR REGRESSION (SVR)
def svr_example(X_train, X_test, y_train, y_test):
    """
    Support Vector Regression.
    """
    from sklearn.svm import SVR
    
    # Scale features (important for SVR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train
    model = SVR(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        epsilon=0.1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"SVR - R²: {r2:.4f}, MSE: {mse:.4f}")
    
    return model, scaler, y_pred


# 16. GRADIENT BOOSTING REGRESSOR
def gradient_boosting_regressor_example(X_train, X_test, y_train, y_test):
    """
    Gradient Boosting Regressor.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Initialize and train
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    feature_importance = model.feature_importances_
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Gradient Boosting Regressor - R²: {r2:.4f}, MSE: {mse:.4f}")
    
    return model, y_pred, feature_importance


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def prepare_data(df, target_column, test_size=0.2, random_state=42):
    """
    Prepare data for training: split features and target, encode labels if needed.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of the target column
        test_size: Proportion of test set
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode categorical target if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def hyperparameter_tuning_example(X_train, y_train, model_type='random_forest'):
    """
    Example of hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to tune ('random_forest', 'svm', 'logistic', etc.)
    
    Returns:
        Best model and best parameters
    """
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    
    elif model_type == 'svm':
        from sklearn.svm import SVC
        model = SVC(random_state=42)
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
    
    elif model_type == 'logistic':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def cross_validation_example(model, X, y, cv=5):
    """
    Example of cross-validation for model evaluation.
    
    Args:
        model: Trained model
        X: Features
        y: Target
        cv: Number of folds
    
    Returns:
        Cross-validation scores
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    return scores


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create sample data
    from sklearn.datasets import make_classification, make_regression
    
    print("=" * 60)
    print("CLASSIFICATION EXAMPLES")
    print("=" * 60)
    
    # Classification example
    X_clf, y_clf = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=10, n_classes=3, random_state=42
    )
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    # Run classification models
    print("\n1. Logistic Regression:")
    logistic_regression_example(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
    
    print("\n2. Decision Tree:")
    decision_tree_example(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
    
    print("\n3. Random Forest:")
    random_forest_example(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
    
    print("\n4. SVM:")
    svm_example(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
    
    print("\n5. KNN:")
    knn_example(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
    
    print("\n6. Naive Bayes:")
    naive_bayes_example(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
    
    print("\n7. Gradient Boosting:")
    gradient_boosting_example(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
    
    print("\n8. Neural Network:")
    neural_network_example(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
    
    print("\n" + "=" * 60)
    print("REGRESSION EXAMPLES")
    print("=" * 60)
    
    # Regression example
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=20, n_informative=10,
        noise=10, random_state=42
    )
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Run regression models
    print("\n1. Linear Regression:")
    linear_regression_example(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
    
    print("\n2. Ridge Regression:")
    ridge_regression_example(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
    
    print("\n3. Lasso Regression:")
    lasso_regression_example(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
    
    print("\n4. Decision Tree Regressor:")
    decision_tree_regressor_example(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
    
    print("\n5. Random Forest Regressor:")
    random_forest_regressor_example(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
    
    print("\n6. SVR:")
    svr_example(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
    
    print("\n7. Gradient Boosting Regressor:")
    gradient_boosting_regressor_example(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING EXAMPLE")
    print("=" * 60)
    
    # Hyperparameter tuning example
    print("\nRandom Forest Hyperparameter Tuning:")
    best_model, best_params = hyperparameter_tuning_example(
        X_train_clf, y_train_clf, model_type='random_forest'
    )

