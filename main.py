import numpy as np
import pandas as pd
import re
import sklearn as sk
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


class Main:
    def __init__(self):
        self.salary_data = pd.read_csv("software_engineer_salaries.csv")
        self.null_dataset = self.salary_data[self.salary_data.isnull().any(axis=1)]
        self.model_dataset = self.salary_data.dropna().copy()

    def preprocessing(self):
        """Preprocess the salary data by cleaning and transforming salary values."""
        processed_salaries = []
        # self.model_dataset = self.model_dataset.drop(columns=['Company Score'])
        # self.model_dataset = self.model_dataset.drop(columns=['Company'])
        self.model_dataset = self.model_dataset.drop(columns=['Date'])

        for salary in self.model_dataset['Salary']:
            if pd.isna(salary):
                processed_salaries.append(None)
                continue

            # Remove special characters but keep "Per Hour"
            cleaned = (salary.replace('$', '').replace('¬†', '')
                       .replace('(Employer est.)', '').replace('(Glassdoor est.)', '')
                       .strip())

            # Check if hourly rate - is a boolean
            is_hourly = 'Per Hour' in cleaned

            # Extract numbers using regex
            numbers = re.findall(r'(\d+(?:\.\d+)?)', cleaned)
            if not numbers:
                processed_salaries.append(None)
                continue

            # Convert to float and calculate average
            numbers = [float(num) for num in numbers]
            avg_salary = sum(numbers) / len(numbers)

            # Convert hourly to yearly if needed
            if is_hourly:
                processed_salaries.append(avg_salary * 40 * 52)  # 40 hours/week * 52 weeks/year
            else:
                processed_salaries.append(avg_salary * 1000)  # Convert K to actual value

        # Add processed salaries as a new column
        self.model_dataset['processed_salary'] = processed_salaries

        # Remove rows where salary processing failed
        self.model_dataset = self.model_dataset.dropna(subset=['processed_salary'])
        self.model_dataset = self.model_dataset.drop(columns=['Salary'])

        scaler = sk.preprocessing.StandardScaler()
        self.model_dataset['processed_salary'] = scaler.fit_transform(self.model_dataset[['processed_salary']])
        return self.model_dataset

    def data_embedding(self):
        """Generate embeddings for company name, location, and job title."""
        transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Generate embeddings for each column
        company_embeddings = transformer_model.encode(self.model_dataset['Company'].tolist())
        location_embeddings = transformer_model.encode(self.model_dataset['Location'].tolist())
        job_title_embeddings = transformer_model.encode(self.model_dataset['Job Title'].tolist())

        # Combine embeddings into a single feature vector for each row
        X = np.hstack([company_embeddings, location_embeddings, job_title_embeddings])

        # Get the processed salaries as the target variable
        y = self.model_dataset['processed_salary'].values

        return X, y

    def train_model(self, X, y, n_splits=5):
        """Train and evaluate the model using K-Fold cross-validation."""
        # Scale the Company Score
        score = self.model_dataset['Company Score'].values.reshape(-1, 1)
        scaler = sk.preprocessing.StandardScaler()
        score_scaled = scaler.fit_transform(score)

        # Add the scaled score to the feature matrix
        X = np.hstack([X, score_scaled])

        # Define the model
        xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, min_child_weight=3, random_state=42)

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

        print(f"Average MSE across folds: {avg_mse:.2f} ± {std_mse:.2f}")
        print(f"Average R² across folds: {avg_r2:.3f} ± {std_r2:.3f}")

        xgb_model.fit(X, y)

        return xgb_model, last_y_test, last_y_pred

    def predict_salary(self, model, new_data):
        """Predict salary for new job entries."""
        # Generate embeddings for new data
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        new_company_embeddings = embedding_model.encode(new_data['Company Name'].tolist())
        new_location_embeddings = embedding_model.encode(new_data['Location'].tolist())
        new_job_title_embeddings = embedding_model.encode(new_data['Job Title'].tolist())

        # Combine the embeddings into a single feature vector
        new_X = np.hstack([new_company_embeddings, new_location_embeddings, new_job_title_embeddings])

        # Predict the salary
        predicted_salary = model.predict(new_X)
        return predicted_salary

    def plot_predictions_vs_actual(self, y_test, y_pred):
        """Plot predicted salaries vs actual salaries."""
        plt.figure(figsize=(10, 6))

        # Scatter plot for actual vs predicted values
        plt.scatter(y_test, y_pred, alpha=0.5)

        # Line of perfect predictions (y = x)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--',
                 label="Perfect Prediction (y=x)")

        plt.title('Predictions vs Actual Salaries')
        plt.xlabel('Actual Salary ($)')
        plt.ylabel('Predicted Salary ($)')
        plt.legend()
        plt.show()

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
    # analysis = main.analyze_data()
    #
    # # Print each metric on its own line with formatting
    # print("Salary Analysis:")
    # print(f"Average Salary: ${analysis['average_salary']:,.2f}")
    # print(f"Median Salary: ${analysis['median_salary']:,.2f}")
    # print(f"Minimum Salary: ${analysis['min_salary']:,.2f}")
    # print(f"Maximum Salary: ${analysis['max_salary']:,.2f}")
    # print(f"Total Positions: {analysis['total_positions']}")

    # main.visualization()

    main.preprocessing()
    X, y = main.data_embedding()
    model, y_test, y_pred = main.train_model(X, y)
    main.plot_predictions_vs_actual(y_test, y_pred)