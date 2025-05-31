Titanic Survival Prediction
This project aims to predict the survival of passengers on the Titanic using machine learning techniques. The dataset contains various features about the passengers, such as age, gender, passenger class, and more. The goal is to build a model that can accurately predict whether a passenger survived or not based on these features.

Table of Contents
Project Overview

Dataset

Exploratory Data Analysis (EDA)

Data Preprocessing

Model Training

Evaluation

Results

Usage

Dependencies

License

Project Overview
The project involves:

Loading and exploring the Titanic dataset.

Performing exploratory data analysis to understand the data.

Preprocessing the data (handling missing values, encoding categorical variables).

Training a Random Forest Classifier to predict survival.

Evaluating the model's performance using accuracy and classification metrics.

Dataset
The dataset used in this project is the famous Titanic dataset, which includes information about passengers such as:

PassengerId: Unique identifier for each passenger.

Survived: Survival status (0 = No, 1 = Yes).

Pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd).

Name: Passenger's name.

Sex: Gender of the passenger.

Age: Age of the passenger.

SibSp: Number of siblings/spouses aboard.

Parch: Number of parents/children aboard.

Ticket: Ticket number.

Fare: Fare paid for the ticket.

Cabin: Cabin number.

Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

Exploratory Data Analysis (EDA)
Key insights from the EDA:

The survival rate was approximately 38%.

Female passengers had a significantly higher survival rate than males.

Passengers in higher classes (1st and 2nd) had a better chance of survival compared to those in the 3rd class.

Data Preprocessing
Steps taken to preprocess the data:

Dropped irrelevant columns (PassengerId, Name, Ticket, Cabin).

Filled missing values in the Age column with the median age.

Filled missing values in the Embarked column with the mode.

Encoded categorical variables (Sex and Embarked) using Label Encoding.

Model Training
A Random Forest Classifier was trained on the preprocessed data. The dataset was split into training and testing sets (80% training, 20% testing).

Evaluation
The model was evaluated using:

Accuracy Score: The percentage of correct predictions.

Classification Report: Includes precision, recall, and F1-score for each class.

Confusion Matrix: Shows the number of true positives, true negatives, false positives, and false negatives.

Results
The model achieved an accuracy of approximately 81% on the test set. Detailed performance metrics can be found in the classification report and confusion matrix.

Usage
Clone the repository:

bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
Install the required dependencies:

bash
pip install -r requirements.txt
Run the Jupyter notebook:

bash
jupyter notebook task1.ipynb
Dependencies
Python 3.x

pandas

numpy

seaborn

matplotlib

scikit-learn
