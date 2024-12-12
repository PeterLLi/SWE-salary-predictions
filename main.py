import pandas as pd
import re
import sklearn as sk
import matplotlib.pyplot as plt

class Main:
    def __init__(self):
        self.salary_data = pd.read_csv("software_engineer_salaries.csv")
        self.null_dataset = self.salary_data[self.salary_data.isnull().any(axis=1)]
        self.model_dataset = self.salary_data.dropna().copy()

    def preprocessing(self):
        """Preprocess the salary data by cleaning and transforming salary values."""
        processed_salaries = []

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

        return self.model_dataset

    def split_data(self):
        print(self.model_dataset.columns)
        X = (self.model_dataset.drop(columns=['Company Score', 'Date', 'Salary', 'processed_salary']))
        print(X.columns)
        y = self.model_dataset['processed_salary']
        X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def scale_data(self):
        train, test, _, _ = self.split_data()
        scaler = sk.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(train)
        X_test = scaler.transform(test)
        return X_train, X_test

    def regressor_model_predict(self):
        regressor = sk.linear_model.LinearRegression()
        X_train, X_test = self.scale_data()
        _, _, y_train, _ = self.split_data()
        regressor.fit(X_train, y_train)
        return regressor.predict(X_test)

    def evaluate_model(self):
        _, _, _, y_test = self.split_data()
        y_pred = self.regressor_model_predict()
        mse_result = sk.metrics.mean_squared_error(y_test, y_pred)
        r2_result = sk.metrics.r2_score(y_test, y_pred)
        print('MSE: ', mse_result)
        print('R2: ', r2_result)


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


if __name__ == '__main__':
    main = Main()
    analysis = main.analyze_data()

    # Print each metric on its own line with formatting
    print("Salary Analysis:")
    print(f"Average Salary: ${analysis['average_salary']:,.2f}")
    print(f"Median Salary: ${analysis['median_salary']:,.2f}")
    print(f"Minimum Salary: ${analysis['min_salary']:,.2f}")
    print(f"Maximum Salary: ${analysis['max_salary']:,.2f}")
    print(f"Total Positions: {analysis['total_positions']}")

    main.evaluate_model()