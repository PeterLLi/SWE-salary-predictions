import pandas as pd
import re
import sklearn as sk

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