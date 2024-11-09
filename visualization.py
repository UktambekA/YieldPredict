import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for Random Forest model in the stacking ensemble
    """
    rf_model = model.named_estimators_['rf']
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance.head(15), x='importance', y='feature')
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    return plt

def plot_learning_curve(learning_curve_data):
    """
    Plot learning curve to analyze model performance
    """
    plt.figure(figsize=(10, 6))
    plt.plot(learning_curve_data['train_sizes'], 
             learning_curve_data['train_scores'], 
             label='Training MAE')
    plt.plot(learning_curve_data['train_sizes'], 
             learning_curve_data['valid_scores'], 
             label='Validation MAE')
    plt.xlabel('Training Examples')
    plt.ylabel('Mean Absolute Error')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    return plt

def plot_predictions_vs_actual(y_true, y_pred, title='Predictions vs Actual'):
    """
    Create scatter plot of predicted vs actual values
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 
             'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    return plt

def plot_residuals(y_true, y_pred):
    """
    Plot residuals to analyze prediction errors
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Count')
    plt.title('Distribution of Residuals')
    return plt