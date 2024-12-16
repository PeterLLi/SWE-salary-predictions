import numpy as np
import pandas as pd
import re
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


class Main:
    def __init__(self):
        self.model_dataset = None
        self.salary_data = pd.read_csv("us-software-engineer-jobs-zenrows.csv")
        self.xgb_model = XGBRegressor(
            n_estimators=1000, # number of boosting rounds
            learning_rate=0.005, # shrinkage to balance training speed and performance
            max_depth=12, # maximum tree depth to model complexity
            min_child_weight=4, # minimum sum of weights in a child to avoid overfitting
            reg_alpha=1.0, # L1 regularization for sparsity and smoothness
            reg_lambda=0.5, # L2 regularization for sparsity and smoothness
            subsample=0.7, # percentage of samples used for training each tree
            colsample_bytree=0.7, # percentage of features used for each tree
            random_state=42 # ensures reproducibility
        )
        self.mse_scores = []
        self.r2_scores = []
        self.last_y_test = None
        self.last_y_pred = None

    def preprocessing(self):
        """Preprocess the salary data by selecting key features and standardizing salaries."""
        # Filter rows where the 'types' column contains "full" (case insensitive)
        # filtered_data = self.salary_data[self.salary_data['types'].str.contains(r'full-time', flags=re.IGNORECASE, na=False, regex=True)]
        filtered_data = self.salary_data[
            self.salary_data['types'].str.match(r'^\s*full-time\s*$', flags=re.IGNORECASE, na=False)
        ]
        # Create a subset of the filtered data with selected columns and drop rows with missing values
        self.model_dataset = filtered_data[['title', 'company', 'salary', 'location']].copy()
        self.model_dataset = self.model_dataset.dropna()

        print(f"Number of entries after feature selection and dropping nulls: {len(self.model_dataset)}")

        # Enhanced title features
        self.model_dataset['is_senior'] = self.model_dataset['title'].str.lower().str.contains(
            r'senior|sr\.?|staff|principal|lead|architect')
        self.model_dataset['is_lead'] = self.model_dataset['title'].str.lower().str.contains(
            r'vice|manager|head|director|chief')
        self.model_dataset['is_mid'] = self.model_dataset['title'].str.lower().str.contains(
            r'(?:\s|^)iii\b|(?:\s|^)iv\b|level\s*[345]|software engineer\s*[345]|\s+3\b|\s+4\b|\s+5\b')
        self.model_dataset['is_junior'] = self.model_dataset['title'].str.lower().str.contains(
            r'junior|jr\.?|entry|associate|(?:\s|^)i\b|(?:\s|^)ii\b|level\s*[12]|software engineer\s*[12]|\s+1\b|\s+2\b')
        self.model_dataset['is_fullstack'] = self.model_dataset['title'].str.lower().str.contains(
            'full.?stack|full.?end')
        self.model_dataset['is_frontend'] = self.model_dataset['title'].str.lower().str.contains(
            'front.?end|react|angular|vue|ui|web')
        self.model_dataset['is_backend'] = self.model_dataset['title'].str.lower().str.contains(
            'back.?end|api|golang|java|python')
        self.model_dataset['is_ml'] = self.model_dataset['title'].str.lower().str.contains(
            'machine|learning|artificial|ml|ai|data')
        self.model_dataset['is_cloud'] = self.model_dataset['title'].str.lower().str.contains(
            'sre|cloud|aws|azure|gcp|devops|infrastructure')
        self.model_dataset['is_mobile'] = self.model_dataset['title'].str.lower().str.contains(
            'mobile|ios|android|flutter|react')
        self.model_dataset['is_embedded'] = self.model_dataset['title'].str.lower().str.contains(
            'embedded|hardware|firmware|iot')
        self.model_dataset['is_security'] = self.model_dataset['title'].str.lower().str.contains(
            'security|crypto|blockchain|security')
        self.model_dataset['is_data'] = self.model_dataset['title'].str.lower().str.contains(
            'data|etl|pipeline|hadoop|spark')
        self.model_dataset['is_game'] = self.model_dataset['title'].str.lower().str.contains(
            'game|unity|unreal|gaming')

        # Remote work feature
        self.model_dataset['is_remote'] = self.model_dataset['location'].str.lower().str.contains('remote', na=False)

        # Tech hub locations
        tech_hubs = {
            'tier_1': ['san francisco', 'san jose', 'seattle', 'new york', 'boston'],
            'tier_2': ['austin', 'denver', 'chicago', 'los angeles', 'san diego', 'portland', 'atlanta'],
        }

        self.model_dataset['is_tier1_hub'] = self.model_dataset['location'].str.lower().apply(
            lambda x: any(hub in x for hub in tech_hubs['tier_1'])
        )
        self.model_dataset['is_tier2_hub'] = self.model_dataset['location'].str.lower().apply(
            lambda x: any(hub in x for hub in tech_hubs['tier_2'])
        )

        processed_salaries = []

        for salary in self.model_dataset['salary']:
            salary = str(salary).lower()

            # Modified regex to capture the whole number including commas
            numbers = [float(num.replace(',', '')) for num in re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', salary)]
            if not numbers:
                processed_salaries.append(None)
                continue

            # Handle ranges - take average if it's a range
            if '-' in salary:
                final_number = sum(numbers) / len(numbers)
            else:
                final_number = numbers[0]

            # Convert to yearly salary based on rate
            if 'hour' in salary:
                final_number = final_number * 40 * 52
            elif 'week' in salary:
                final_number = final_number * 52
            elif 'month' in salary:
                final_number = final_number * 12

            # processed_salaries.append(final_number)
            min_salary = 30000  # $30k minimum
            max_salary = 300000  # $300k maximum
            if min_salary <= final_number <= max_salary:
                processed_salaries.append(final_number)
            else:
                processed_salaries.append(None)  # Will be removed by dropna()

        self.model_dataset['processed_salary'] = processed_salaries
        self.model_dataset = self.model_dataset.dropna()
        print(f"Number of entries after preprocessing: {len(self.model_dataset)}")
        return self.model_dataset

    def data_embedding(self):
        """Generate unified embeddings for all text-based features."""
        transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Combine text fields into a single string
        self.model_dataset['Combined'] = (
                self.model_dataset['company'] + " " +
                self.model_dataset['title'] + " " +
                self.model_dataset['location']
        )

        # Generate embeddings for the combined text
        combined_embeddings = transformer_model.encode(self.model_dataset['Combined'].tolist())

        # Add all the boolean features
        title_location_features = np.column_stack([
            self.model_dataset['is_senior'],
            self.model_dataset['is_lead'],
            self.model_dataset['is_mid'],
            self.model_dataset['is_junior'],
            self.model_dataset['is_fullstack'],
            self.model_dataset['is_frontend'],
            self.model_dataset['is_backend'],
            self.model_dataset['is_ml'],
            self.model_dataset['is_cloud'],
            self.model_dataset['is_mobile'],
            self.model_dataset['is_embedded'],
            self.model_dataset['is_security'],
            self.model_dataset['is_data'],
            self.model_dataset['is_game'],
            self.model_dataset['is_tier1_hub'],
            self.model_dataset['is_tier2_hub'],
            self.model_dataset['is_remote']
        ])

        # Combine embeddings with title and location features
        X = np.hstack([combined_embeddings, title_location_features])

        # Get the processed salaries as the target variable
        y = self.model_dataset['processed_salary'].values

        return X, y

    def train_model(self, X, y, n_splits=5):
        """Train and evaluate the model using K-Fold cross-validation and holdout test set."""
        # First split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Set up K-Fold cross-validation on training data
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        self.mse_scores = []
        self.r2_scores = []

        print("Performing cross-validation on training data...")
        # Perform the K-Fold split
        for train_index, val_index in kf.split(X_train):
            X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

            # Train the model
            self.xgb_model.fit(X_fold_train, y_fold_train)

            # Predict on validation fold
            y_fold_pred = self.xgb_model.predict(X_fold_val)

            # Save the last fold's results
            self.last_y_test = y_fold_val
            self.last_y_pred = y_fold_pred

            # Calculate metrics
            mse = mean_squared_error(y_fold_val, y_fold_pred)
            r2 = r2_score(y_fold_val, y_fold_pred)

            self.mse_scores.append(mse)
            self.r2_scores.append(r2)

        # Print cross-validation results
        avg_mse = np.mean(self.mse_scores)
        std_mse = np.std(self.mse_scores)
        avg_r2 = np.mean(self.r2_scores)
        std_r2 = np.std(self.r2_scores)

        print(f"Cross-validation metrics (on training data):")
        print(f"Average MSE across folds: {avg_mse:.2f} ± {std_mse:.2f}")
        print(f"Average R² across folds: {avg_r2:.3f} ± {std_r2:.3f}")

        # Train final model on last fold and evaluate on test set
        print("\nEvaluating on holdout test set...")
        y_test_pred = self.xgb_model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"Test set metrics:")
        print(f"MSE on test set: {test_mse:.2f}")
        print(f"R² on test set: {test_r2:.3f}")

        return self.xgb_model, self.last_y_test, self.last_y_pred

    def analyze_data(self):
        """Analyze the processed salary data."""
        processed_data = self.preprocessing()

        salary_analysis = {
            'average_salary': processed_data['processed_salary'].mean(),
            'median_salary': processed_data['processed_salary'].median(),
            'min_salary': processed_data['processed_salary'].min(),
            'max_salary': processed_data['processed_salary'].max(),
            'total_positions': len(processed_data),
        }

        return salary_analysis

    def visualization(self):
        sns.set_theme()
        plt.figure(figsize=(10, 6))

        plt.scatter(range(len(self.model_dataset['processed_salary'])), self.model_dataset['processed_salary'],
                    alpha=0.5)

        plt.title('Software Engineer Salaries')
        plt.xlabel('Rank')
        plt.ylabel('Salary ($)')
        plt.show()

    def visualize_predictions(self):
        """Visualize model predictions vs actual values."""
        plt.figure(figsize=(10, 6))

        # Create a scatter plot of actual vs predicted values
        plt.scatter(self.last_y_test, self.last_y_pred, alpha=0.5)

        # Add a perfect prediction line
        min_val = min(min(self.last_y_test), min(self.last_y_pred))
        max_val = max(max(self.last_y_test), max(self.last_y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        plt.title('Predicted vs Actual Salaries')
        plt.xlabel('Actual Salary ($)')
        plt.ylabel('Predicted Salary ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == '__main__':
    main = Main()
    main.preprocessing()
    main.visualization()
    X, y = main.data_embedding()
    model, y_test, y_pred = main.train_model(X, y)
    main.visualize_predictions()