import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedMultipleLinearRegression:
    def __init__(self, model_type='ols'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()

    def preprocess_data(self, X, y):
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_model(self, X_train, y_train):
        if self.model_type == 'ols':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif self.model_type == 'lasso':
            self.model = Lasso(alpha=1.0)
        elif self.model_type == 'elastic_net':
            self.model = ElasticNet(alpha=1.0, l1_ratio=0.5)
        else:
            raise ValueError("Invalid model type")

        self.model.fit(X_train, y_train)
        logging.info(f"Model {self.model_type} trained successfully")

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Model performance - MSE: {mse:.4f}, R-squared: {r2:.4f}")
        return mse, r2

    def analyze_coefficients(self, feature_names):
        coefficients = pd.DataFrame({'Feature': feature_names, 'Coefficient': self.model.coef_})
        coefficients = coefficients.sort_values(by='Coefficient', key=abs, ascending=False)
        logging.info("Model coefficients:\n" + str(coefficients))
        return coefficients

    def plot_residuals(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.savefig('residual_plot.png')
        logging.info("Residual plot saved as 'residual_plot.png'")

    def predict(self, X_new):
        X_scaled = self.scaler.transform(X_new)
        return self.model.predict(X_scaled)

# Example usage
if __name__ == "__main__":
    # Load your data here
    # data = pd.read_csv('your_data.csv')
    # X = data.drop('target_column', axis=1)
    # y = data['target_column']

    regressor = AdvancedMultipleLinearRegression(model_type='ridge')
    # X_scaled, y = regressor.preprocess_data(X, y)
    # X_train, X_test, y_train, y_test = regressor.split_data(X_scaled, y)
    # regressor.train_model(X_train, y_train)
    # mse, r2 = regressor.evaluate_model(X_test, y_test)
    # coefficients = regressor.analyze_coefficients(X.columns)
    # regressor.plot_residuals(X_test, y_test)
    # predictions = regressor.predict(X_new)
