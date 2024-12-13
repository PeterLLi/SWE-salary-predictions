import numpy as np
import pandas as pd
import re
import sklearn as sk
from sklearn.model_selection import train_test_split
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
        self.model_dataset = self.model_dataset.drop(columns=['Company Score'])
        self.model_dataset = self.model_dataset.drop(columns=['Company'])
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
        # company_embeddings = transformer_model.encode(self.model_dataset['Company'].tolist())
        location_embeddings = transformer_model.encode(self.model_dataset['Location'].tolist())
        job_title_embeddings = transformer_model.encode(self.model_dataset['Job Title'].tolist())

        # Combine embeddings into a single feature vector for each row
        X = np.hstack([location_embeddings, job_title_embeddings])

        # Get the processed salaries as the target variable
        y = self.model_dataset['processed_salary'].values

        return X, y

    def train_model(self, X, y):
        """Train a gradient boosting model using XGBoost."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train model
        model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, min_child_weight=3, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.3f}")

        return model, y_test, y_pred

    def predict_salary(self, model, new_data, scaler):
        """Predict salary for new job entries."""
        # Generate embeddings for new data
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        new_location_embeddings = embedding_model.encode(new_data['Location'].tolist())
        new_job_title_embeddings = embedding_model.encode(new_data['Job Title'].tolist())

        # Combine the embeddings into a single feature vector
        new_X = np.hstack([new_location_embeddings, new_job_title_embeddings])

        # Predict the salary
        predicted_salary = model.predict(new_X)

        predicted_salary = scaler.inverse_transform(predicted_salary.reshape(-1, 1))

        return predicted_salary[0][0]

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


class PredictSalary:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict_salary(self, title, location):
        """Predict salary for new job entries."""
        # Generate embeddings for new data
        predict = self.format_data(title, location)
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        new_location_embeddings = embedding_model.encode(predict['Location'].tolist())
        new_job_title_embeddings = embedding_model.encode(predict['Job Title'].tolist())

        # Combine the embeddings into a single feature vector
        new_X = np.hstack([new_location_embeddings, new_job_title_embeddings])

        # Predict the salary
        predicted_salary = model.predict(new_X)

        predicted_salary = scaler.inverse_transform(predicted_salary.reshape(-1, 1))

        # return predicted_salary[0][0]
        print(f"Predicted Salary ($): {predicted_salary[0][0]}, for {title} in {location}")

    def format_data(self, title, location):
        new_data = pd.DataFrame({
            'Location': [location],
            'Job Title': [title]
        })
        return new_data


if __name__ == '__main__':
    main = Main()
    scaler = sk.preprocessing.StandardScaler()

    main.preprocessing()
    scaler.fit(main.model_dataset[['processed_salary']])

    X, y = main.data_embedding()
    model, y_test, y_pred = main.train_model(X, y)
    main.plot_predictions_vs_actual(y_test, y_pred)

    # Enter own values for location and title
    location = input("Please enter a location i.e. Milwaukee, WI\n")
    title = input("Please enter a job position i.e. Senior Software Engineer\n")

    # Predict the salary for new data
    PredictSalary(model, scaler).predict_salary(location, title)
