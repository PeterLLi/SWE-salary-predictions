import os
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


class Main:
    def __init__(self):
        self.model_dataset = None
        self.salary_data = pd.read_csv("us-software-engineer-jobs-zenrows.csv")
        self.model_file = "xgb_model.pkl"  # File to save the trained model
        self.xgb_model = XGBRegressor(
            n_estimators=1000,  # number of boosting rounds
            learning_rate=0.005,  # shrinkage to balance training speed and performance
            max_depth=12,  # maximum tree depth to model complexity
            min_child_weight=4,  # minimum sum of weights in a child to avoid overfitting
            reg_alpha=1.0,  # L1 regularization for sparsity and smoothness
            reg_lambda=0.5,  # L2 regularization for sparsity and smoothness
            subsample=0.7,  # percentage of samples used for training each tree
            colsample_bytree=0.7,  # percentage of features used for each tree
            random_state=42  # ensures reproducibility
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

        print(f"Number of entries after feature selection: {len(self.model_dataset)}")

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

    def load_model(self):
        """Load the model if it exists."""
        if os.path.exists(self.model_file):
            print("Loading saved model...")
            self.xgb_model = joblib.load(self.model_file)
            return True
        return False

    def save_model(self):
        """Save the trained model."""
        print("Saving model...")
        joblib.dump(self.xgb_model, self.model_file)

    def train_model(self, X, y, n_splits=5):
        """Train and evaluate the model using K-Fold cross-validation and holdout test set."""
        # Check if a saved model exists
        if self.load_model():
            print("Model loaded. Skipping training...")
            return self.xgb_model, self.last_y_test, self.last_y_pred

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

        # Save the trained model
        self.save_model()

        # Visualize results
        self.visualize_predictions()

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
        sorted_salaries = sorted(self.model_dataset['processed_salary'])
        plt.scatter(range(len(sorted_salaries)), sorted_salaries,
                    alpha=0.5)

        plt.title('Software Engineer Salaries (sorted)')
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

    def predict_single_entry(self, entry):
        """Predict salary for a single data entry."""
        # Generate boolean features
        is_senior = bool(re.search(r'senior|sr\.?|staff|principal|lead|architect', entry['title'].lower()))
        is_lead = bool(re.search(r'vice|manager|head|director|chief', entry['title'].lower()))
        is_mid = bool(
            re.search(r'(?:\s|^)iii\b|(?:\s|^)iv\b|level\s*[345]|software engineer\s*[345]|\s+3\b|\s+4\b|\s+5\b',
                      entry['title'].lower()))
        is_junior = bool(re.search(
            r'junior|jr\.?|entry|associate|(?:\s|^)i\b|(?:\s|^)ii\b|level\s*[12]|software engineer\s*[12]|\s+1\b|\s+2\b',
            entry['title'].lower()))
        is_fullstack = bool(re.search(r'full.?stack|full.?end', entry['title'].lower()))
        is_frontend = bool(re.search(r'front.?end|react|angular|vue|ui|web', entry['title'].lower()))
        is_backend = bool(re.search(r'back.?end|api|golang|java|python', entry['title'].lower()))
        is_ml = bool(re.search(r'machine|learning|artificial|ml|ai|data', entry['title'].lower()))
        is_cloud = bool(re.search(r'sre|cloud|aws|azure|gcp|devops|infrastructure', entry['title'].lower()))
        is_mobile = bool(re.search(r'mobile|ios|android|flutter|react', entry['title'].lower()))
        is_embedded = bool(re.search(r'embedded|hardware|firmware|iot', entry['title'].lower()))
        is_security = bool(re.search(r'security|crypto|blockchain|security', entry['title'].lower()))
        is_data = bool(re.search(r'data|etl|pipeline|hadoop|spark', entry['title'].lower()))
        is_game = bool(re.search(r'game|unity|unreal|gaming', entry['title'].lower()))

        # Remote work feature
        is_remote = 'remote' in entry['location'].lower()

        # Tech hub locations
        tech_hubs = {
            'tier_1': ['san francisco', 'san jose', 'seattle', 'new york', 'boston'],
            'tier_2': ['austin', 'denver', 'chicago', 'los angeles', 'san diego', 'portland', 'atlanta'],
        }

        is_tier1_hub = any(hub in entry['location'].lower() for hub in tech_hubs['tier_1'])
        is_tier2_hub = any(hub in entry['location'].lower() for hub in tech_hubs['tier_2'])

        # Combine text fields into a single string
        combined_text = f"{entry['company']} {entry['title']} {entry['location']}"

        # Generate embedding for the combined text
        transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
        combined_embedding = transformer_model.encode([combined_text])  # Shape: (1, 384)

        # Combine all boolean features into an array
        title_location_features = np.array([
            is_senior, is_lead, is_mid, is_junior, is_fullstack, is_frontend, is_backend,
            is_ml, is_cloud, is_mobile, is_embedded, is_security, is_data, is_game,
            is_tier1_hub, is_tier2_hub, is_remote
        ]).reshape(1, -1)  # Shape: (1, 17)

        # Combine embeddings with boolean features
        features = np.hstack([combined_embedding, title_location_features])  # Shape: (1, 401)

        # Predict using the trained XGBoost model
        predicted_salary = self.xgb_model.predict(features)

        return predicted_salary[0]

    def get_user_input(self):
        """
        Continuously prompts the user to input job details for salary prediction.
        Exits when the user types 'exit' or 'quit'.
        """
        while True:
            print("\n--- Enter Job Details ---")

            # Get user input with validation
            title = input("Enter a job title (or type 'exit' to quit): ").strip()
            if title.lower() in {'exit', 'quit'}:
                print("Exiting... Goodbye!")
                break

            company = input("Enter a company: ").strip()
            location = input("Enter a location: ").strip()

            # Create a sample entry for prediction
            sample_entry = {
                'title': title,
                'company': company,
                'location': location,
            }

            # Predict salary
            try:
                predicted_salary = main.predict_single_entry(sample_entry)
                print(f"\nPredicted Salary: ${predicted_salary:,.2f}")
            except Exception as e:
                print(f"Error in prediction: {e}")

            # Ask user if they want to enter another position
            again = input("\nDo you want to predict another salary? (yes/no): ").strip().lower()
            if again not in {'yes', 'y'}:
                print("Exiting... Goodbye!")
                break


if __name__ == '__main__':
    main = Main()
    main.preprocessing()
    main.visualization()
    X, y = main.data_embedding()
    model, y_test, y_pred = main.train_model(X, y)

    main.get_user_input()
