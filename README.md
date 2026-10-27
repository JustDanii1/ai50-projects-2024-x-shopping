# Shopping

This project is part of Harvard's CS50's Introduction to Artificial Intelligence with Python.

## Description

This program predicts whether online shopping visitors will complete a purchase using a K-Nearest Neighbors (KNN) machine learning model.

It uses data about browsing behavior, page visits, time spent on pages, visitor type, and other session characteristics to make predictions.

## How It Works

- Loads shopping session data from a CSV file.
- Converts the data into numerical features.
- Splits the dataset into training and testing sets.
- Trains a K-Nearest Neighbors classifier with one neighbor.
- Predicts whether a visitor will generate revenue.
- Evaluates the model using sensitivity and specificity.

## Technologies

- Python 3
- Scikit-learn
- Machine Learning
- K-Nearest Neighbors (KNN)
- CSV

## Installation

Install the required library:

```bash
pip install scikit-learn
```

## Usage

Run the program with a CSV dataset:

```bash
python shopping.py shopping.csv
```

The CSV file must contain the columns expected by the program, including `Administrative`, `Month`, `VisitorType`, `Weekend`, and `Revenue`.

## Output

The program displays:

- Number of correct predictions
- Number of incorrect predictions
- True Positive Rate
- True Negative Rate

## Course

Harvard CS50's Introduction to Artificial Intelligence with Python — Shopping Project.
