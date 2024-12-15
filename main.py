import numpy as np
import pandas as pd
import re
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


class Main:
    def __init__(self):
        self.model_dataset = None
        self.salary_data = pd.read_csv("us-software-engineer-jobs-zenrows.csv")
        self.predictor = None

    def preprocessing(self):
        """Preprocess the salary data by selecting key features and standardizing salaries."""
        self.model_dataset = self.salary_data[['title', 'company', 'salary', 'location', 'remote_work_model']].copy()
        self.model_dataset = self.model_dataset.dropna()

        # Enhanced title features
        self.model_dataset['is_senior'] = self.model_dataset['title'].str.lower().str.contains(
            'senior|sr\.|sr|staff|principal')
        self.model_dataset['is_lead'] = self.model_dataset['title'].str.lower().str.contains(
            'vice|lead|manager|head|director|chief')
        self.model_dataset['is_junior'] = self.model_dataset['title'].str.lower().str.contains(
            'junior|jr\.|jr|entry|associate')
        self.model_dataset['is_fullstack'] = self.model_dataset['title'].str.lower().str.contains(
            'full.?stack|full.?end')
        self.model_dataset['is_frontend'] = self.model_dataset['title'].str.lower().str.contains(
            'front.?end|react|angular|vue|ui|web')
        self.model_dataset['is_backend'] = self.model_dataset['title'].str.lower().str.contains(
            'back.?end|api|golang|java|python')
        self.model_dataset['is_ml'] = self.model_dataset['title'].str.lower().str.contains(
            'machine|learning|artificial|ml|ai|data sci')
        self.model_dataset['is_cloud'] = self.model_dataset['title'].str.lower().str.contains(
            'sre|cloud|aws|azure|gcp|devops|infrastructure')
        self.model_dataset['is_mobile'] = self.model_dataset['title'].str.lower().str.contains(
            'mobile|ios|android|flutter|react native')
        self.model_dataset['is_embedded'] = self.model_dataset['title'].str.lower().str.contains(
            'embedded|hardware|firmware|iot')
        self.model_dataset['is_security'] = self.model_dataset['title'].str.lower().str.contains(
            'security|crypto|blockchain|security')
        self.model_dataset['is_data'] = self.model_dataset['title'].str.lower().str.contains(
            'data|etl|pipeline|hadoop|spark')
        self.model_dataset['is_game'] = self.model_dataset['title'].str.lower().str.contains(
            'game|unity|unreal|gaming')

        # Remote work feature
        self.model_dataset['is_remote'] = self.model_dataset['location'].str.lower().str.contains('remote',na=False)

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

            processed_salaries.append(final_number)

        self.model_dataset['processed_salary'] = processed_salaries
        self.model_dataset = self.model_dataset.dropna()
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
        """Train and evaluate the model using K-Fold cross-validation."""
        # Directly use the preprocessed and scaled Company Score
        # score_scaled = self.model_dataset['Company Score'].values.reshape(-1, 1)

        # Add the scaled score to the feature matrix
        # X = np.hstack([X, score_scaled])

        # Define the model
        xgb_model = XGBRegressor(
            n_estimators=750,
            learning_rate=0.01,
            max_depth=12,
            min_child_weight=4,
            # reg_alpha=0.5,
            # reg_lambda=1.0,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42
        )

        # Set up K-Fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        mse_scores = []
        r2_scores = []

        # Variables to store the last fold's test data and predictions
        last_y_test = None
        last_y_pred = None

        # Perform the K-Fold split
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train the model
            xgb_model.fit(X_train, y_train)

            # Predict on the test fold
            y_pred = xgb_model.predict(X_test)

            # Save the last fold's results
            last_y_test = y_test
            last_y_pred = y_pred

            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mse_scores.append(mse)
            r2_scores.append(r2)

        # Calculate average metrics
        avg_mse = np.mean(mse_scores)
        std_mse = np.std(mse_scores)
        avg_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)

        print(f"XGBoost Average MSE across folds: {avg_mse:.2f} ± {std_mse:.2f}")
        print(f"XGBoost Average R² across folds: {avg_r2:.3f} ± {std_r2:.3f}")

        xgb_model.fit(X, y)

        return xgb_model, last_y_test, last_y_pred

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

        # Sort salaries and plot them
        # sorted_salaries = sorted(self.model_dataset['processed_salary'])
        plt.scatter(range(len(self.model_dataset['processed_salary'])), self.model_dataset['processed_salary'],
                    alpha=0.5)

        plt.title('Software Engineer Salaries (Sorted)')
        plt.xlabel('Rank')
        plt.ylabel('Salary ($)')
        plt.show()


if __name__ == '__main__':
    main = Main()
    main.preprocessing()
    X, y = main.data_embedding()
    model, y_test, y_pred = main.train_model(X, y)

