# Salary Prediction Using Machine Learning

> A beginner-friendly Linear Regression project built with Python and Scikit-learn.
> Submitted as part of an IBM BOB Internship Project.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [How to Run](#how-to-run)
6. [How It Works](#how-it-works)
7. [Results](#results)
8. [Technologies Used](#technologies-used)

---

## Project Overview

This project demonstrates how to build a simple **salary prediction system** using
**Linear Regression** — one of the most fundamental algorithms in Machine Learning.

Given a person's **years of experience**, the model predicts their expected annual **salary**.

---

## Dataset

| File | `salary_data.csv` |
|------|-------------------|
| Rows | 27 samples |
| Feature | `YearsExperience` — years of professional experience |
| Target | `Salary` — annual salary in USD |

The dataset contains a strong positive linear relationship between experience and salary,
making it an ideal starting point for learning regression.

---

## Project Structure

```
IBM_BOB_Salary_Project/
│
├── salary_data.csv          ← raw dataset
├── requirements.txt         ← Python package dependencies
├── README.md                ← this file
│
├── src/
│   ├── data_loader.py       ← load and validate CSV data
│   ├── train_model.py       ← train, evaluate, and save the model
│   └── predict.py           ← prediction helper function
│
├── app/
│   └── app.py               ← command-line user interface
│
├── models/
│   └── salary_model.pkl     ← saved trained model (generated on first run)
│
└── reports/
    ├── regression_plot.png  ← scatter plot with regression line (generated)
    ├── metrics.json         ← evaluation metrics in JSON (generated)
    └── project_report.html  ← static HTML project report
```

---

## Setup & Installation

### Prerequisites

- Python 3.8 or higher
- `pip` package manager

### 1. Clone / download the project

```bash
cd IBM_BOB_Salary_Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1 — Train the model

Run this **once** from the project root. It trains the model, prints evaluation
metrics, saves the model to `models/salary_model.pkl`, and saves a plot to
`reports/regression_plot.png`.

```bash
python src/train_model.py
```

### Step 2 — Launch the CLI application

```bash
python app/app.py
```

You will see a prompt:

```
Enter years of experience (e.g. 5.0): 
```

Type a number and press **Enter** to get an estimated salary. Type `quit` to exit.

---

## How It Works

```
CSV file  →  data_loader.py  →  train_model.py  →  salary_model.pkl
                                                         ↓
                                              predict.py (loads model)
                                                         ↓
                                                 app/app.py (CLI)
```

1. **Data Loading** — `data_loader.py` reads the CSV, validates columns, and checks for missing values.
2. **Training** — `train_model.py` splits the data 80/20, fits a `LinearRegression` from Scikit-learn, and evaluates it using R², MAE, and RMSE.
3. **Saving** — the trained model is serialised with `joblib` to `models/salary_model.pkl`.
4. **Prediction** — `predict.py` loads the saved model and provides a `predict_salary(years)` function.
5. **UI** — `app/app.py` wraps everything in a friendly command-line loop.

---

## Results

> Metrics are generated dynamically when you run `train_model.py`.  
> See `reports/metrics.json` and `reports/regression_plot.png` after training.

The model equation takes the form:

```
Salary = slope × YearsExperience + intercept
```

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Linear Regression model, train/test split, metrics |
| `matplotlib` | Regression scatter plot |
| `joblib` | Model serialisation (save / load) |

---

*Built with Python 3 · IBM BOB Internship Project*
