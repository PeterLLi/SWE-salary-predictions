import numpy as np
import pandas as pd
import re
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import torch


class Main:
    def __init__(self):
        self.salary_data = pd.read_csv("software_engineer_salaries.csv")
        self.null_dataset = self.salary_data[self.salary_data.isnull().any(axis=1)]
        self.model_dataset = self.salary_data.dropna().copy()

    def preprocessing(self):
        """Preprocess the salary data by cleaning and transforming salary values."""
        processed_salaries = []
        self.model_dataset = self.model_dataset.drop(columns=['Company Score'])
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

        return self.model_dataset

    def data_embedding(self):
        """Generate embeddings for company name, location, and job title."""
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Generate embeddings for each column
        company_embeddings = model.encode(self.model_dataset['Company'].tolist())
        location_embeddings = model.encode(self.model_dataset['Location'].tolist())
        job_title_embeddings = model.encode(self.model_dataset['Job Title'].tolist())

        # Combine embeddings into a single feature vector for each row
        X = np.hstack([company_embeddings, location_embeddings, job_title_embeddings])

        # Get the processed salaries as the target variable
        y = self.model_dataset['processed_salary'].values

        return X, y

    def train_model(self, X, y):
        """Train a linear regression model using the embeddings as features."""
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.2f}")

        return model

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
    model = main.train_model(X, y)
