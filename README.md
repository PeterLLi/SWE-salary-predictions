# SWE-Salary-Predictions

## Overview
This project predicts software engineer salaries using a machine learning model. The code preprocesses data, trains a model, and evaluates it while providing performance metrics and visualizations.

## Requirements

### Dependencies
- All dependencies are listed in `requirements.txt`.
- Upon running the code, you will be prompted to install the libraries listed in the requirements file.

### Python Version
- **Python 3.11.11** is a hard requirement as this version supports the necessary version of PyTorch that is used by SentenceTransformer. Any newer releases i.e., 3.12.x or 3.13.x, will NOT work at the writing of this README, and any Python versions before 3.11.11 have NOT been tested.

### Dataset
- The dataset must be placed in the same directory as `main.py`.
- Ensure the dataset file is named: `us-software-engineer-jobs-zenrows.csv` (this name is already used in the repository). This dataset is included in the project submission and can be found in this repository. If, for some reason, neither of the versions works, links to the dataset on Kaggle within the report.

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
   - Open the file in PyCharm and execute the file.
      - If you created a virtual environment through pyenv in the previous step, please be sure to select this virtual environment as the interpreter in PyCharm. We recommend that you create a virtual environment through pyenv regardless if this is run within PyCharm or not as this prevents conflicts with other projects and ensures that SentenceTransformer will work as intended for reproducibility.

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
     - Output a .pkl file containing the model (if our provided .pkl is not put in the same directory as main.py).
     - Output two graphs and performance metrics to the console.

## Notes
- The entire project codebase is contained within `main.py`.
- Our submission on Canvas provides a .pkl file generated on our computers. This is so that run times will be within a reasonable time frame.
   - If you would like to view all the generated graphs, please run main.py without the .pkl file in the directory.
   - If you prefer run time and would prefer to just test the "entry input", please include our provided .pkl file in the same directory as main.py.
   - We highly recommend utilizing our provided .pkl file as this will drastically cut down the run time of the Python file.
      - Note that run times will still vary a bit depending on hardware specification with the provided .pkl file.

