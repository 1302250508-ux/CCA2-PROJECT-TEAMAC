"""
Traditional Programming vs Machine Learning Programming
--------------------------------------------------------
Name : YOUR NAME HERE
PRN  : YOUR PRN HERE

Problem: Predict whether a student will Pass or Fail from
         hours studied, attendance % and assignments done.

Traditional programming : WE write the rules (if/else) by hand.
Machine learning        : the computer LEARNS the rules from labelled data.

Input : input_data.xlsx (sheet "Data")
Run   : python traditional_vs_ml.py
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

STUDENT_NAME = "YOUR NAME HERE"
PRN = "YOUR PRN HERE"

INPUT_FILE = "input_data.xlsx"
SHEET_NAME = "Data"
FEATURES = ["Hours_Studied", "Attendance_Pct", "Assignments_Done"]
LABEL = "Result"


# ---------------------------------------------------------------
# 1. TRADITIONAL PROGRAMMING - rules written by the programmer
# ---------------------------------------------------------------
def traditional_rule(row):
    """Hand-written rules. Change the thresholds to make them your own."""
    if row["Attendance_Pct"] >= 75 and row["Hours_Studied"] >= 4:
        return "Pass"
    if row["Assignments_Done"] >= 8 and row["Attendance_Pct"] >= 60:
        return "Pass"
    return "Fail"


# ---------------------------------------------------------------
# 2. MACHINE LEARNING - rules learned from data
# ---------------------------------------------------------------
def train_ml_model(x_train, y_train):
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(x_train, y_train)
    return model


def main():
    print(f"Name: {STUDENT_NAME} | PRN: {PRN}")
    print("=" * 60)

    # Read the input from the Excel file
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    print(f"Loaded {len(df)} rows from '{INPUT_FILE}' (sheet '{SHEET_NAME}')")
    print(df.head().to_string(index=False))

    x = df[FEATURES]
    y = df[LABEL]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    # Traditional approach
    trad_pred = x_test.apply(traditional_rule, axis=1)
    trad_acc = accuracy_score(y_test, trad_pred)

    # ML approach
    model = train_ml_model(x_train, y_train)
    ml_pred = model.predict(x_test)
    ml_acc = accuracy_score(y_test, ml_pred)

    print("\n--- Results on the same unseen test data ---")
    print(f"Traditional (hand-written rules) accuracy : {trad_acc:.2%}")
    print(f"Machine Learning (decision tree) accuracy : {ml_acc:.2%}")

    print("\n--- Rules the ML model learned by itself ---")
    print(export_text(model, feature_names=FEATURES))

    # Predict for brand-new students using BOTH approaches
    new_students = pd.DataFrame(
        {
            "Hours_Studied": [6.0, 2.0, 4.5],
            "Attendance_Pct": [90, 55, 70],
            "Assignments_Done": [9, 3, 6],
        }
    )
    new_students["Traditional"] = new_students.apply(traditional_rule, axis=1)
    new_students["ML"] = model.predict(new_students[FEATURES])
    print("--- Predictions for new students ---")
    print(new_students.to_string(index=False))

    print("\nKey idea: in traditional programming the rules go IN;")
    print("in machine learning the rules come OUT of the data.")


if __name__ == "__main__":
    main()
