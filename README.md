# SWE-Salary-Predictions

## Overview
This project predicts software engineer salaries using a machine learning model. The code preprocesses data, trains a model, and evaluates it while providing performance metrics and visualizations.

## Requirements

### Dependencies
- All dependencies are listed in `requirements.txt`.
- Upon running the code, you will be prompted to install the libraries listed in the requirements file.

### Python Version
- **Python 3.11** is required as it supports the necessary version of PyTorch used by SentenceTransformer.

### Dataset
- The dataset must be placed in the same directory as `main.py`.
- Ensure the dataset file is named: `us-software-engineer-jobs-zenrows.csv` (this name is already used in the repository).

## Running the Model

1. **Setup**
   - Ensure all dependencies are installed by running:
     ```bash
     pip install -r requirements.txt
     ```
   - Confirm that Python 3.11 is installed on your system.
   - Verify that the dataset file is correctly named and located in the appropriate directory.

2. **Execution**
   - Run the main script:
     ```bash
     python main.py
     ```

3. **Outputs**
   - The script will:
     - Preprocess the data.
     - Train the model.
     - Evaluate the model.
     - Output two graphs and performance metrics to the console.

## Notes
- The entire project codebase is contained within `main.py`.

