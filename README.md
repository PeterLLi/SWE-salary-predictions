# SWE-Salary-Predictions

## Overview
This project predicts software engineer salaries using a machine learning model. The code preprocesses data, trains a model, and evaluates it while providing performance metrics and visualizations.

## Requirements

### Dependencies
- All dependencies are listed in `requirements.txt`.
- Upon running the code, you will be prompted to install the libraries listed in the requirements file.

### Python Version
- **Python 3.11.11** is required as it supports the necessary version of PyTorch used by SentenceTransformer.

### Dataset
- The dataset must be placed in the same directory as `main.py`.
- Ensure the dataset file is named: `us-software-engineer-jobs-zenrows.csv` (this name is already used in the repository).

## Running the Model

1. **Setup**
   - Ensure all dependencies are installed by running:
     ```bash
     pip install -r requirements.txt
     ```
   - Confirm that Python 3.11.11 is installed on your system. We recommend creating a virtual environment through pyenv with this Python version.
   - Verify that the dataset file is correctly named and located in the appropriate directory.

2. **Execution**
   - Run the main script:
     ```bash
     python main.py
     ```
     OR
   - Open the file in PyCharm and execute the file. If you are running this through PyCharm and have created a virtual environment through pyenv, please be sure to select your virtual environment as the interpreter in PyCharm to ensure that there are no conflicts.

3. **Inputting an Entry**
   - After running the script, you will be prompted to input job details for salary prediction.
   - Enter the following details when prompted:
     - **Job Title**: Provide the job title, e.g., `Senior Software Engineer`.
     - **Company**: Enter the company name, e.g., `Google`.
     - **Location**: Specify the location, e.g., `San Francisco`.
   - Example interaction:
     ```
     Enter a job title: Senior Software Engineer
     Enter a company: Google
     Enter a location: San Francisco
     Predicted Salary: $150,000.00
     ```
   - To input another position, type `YES` when prompted, or type `no` to exit.

4. **Outputs**
   - The script will:
     - Preprocess the data.
     - Train the model.
     - Evaluate the model.
     - Output a .pkl file containing the model
     - Output two graphs and performance metrics to the console.

## Notes
- The entire project codebase is contained within `main.py`.
- Our submission provides a generated .pkl file. This is so that run times are within a reasonable time frame. To view all the generated graphs, please run main.py without the .pkl file in the directory.

