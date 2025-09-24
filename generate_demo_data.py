import pandas as pd
import os

# Ensure folder exists
os.makedirs("data/samples", exist_ok=True)

# Employees and questions
employees = ["E101", "E102", "E103", "E104"]
questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]

# Pre-test answers
pre_answers = [
    ["A","B","C","D","A"],
    ["B","C","B","A","C"],
    ["C","A","D","B","B"],
    ["D","D","A","C","A"]
]

# Post-test answers (some improvement)
post_answers = [
    ["A","B","C","D","A"],
    ["B","B","B","A","C"],
    ["C","A","C","B","B"],
    ["D","C","A","C","A"]
]

# Create DataFrames
pre_demo = pd.DataFrame(
    [(emp, q, ans) for emp, ans_list in zip(employees, pre_answers) for q, ans in zip(questions, ans_list)],
    columns=["employee_id", "question_id", "answer"]
)

post_demo = pd.DataFrame(
    [(emp, q, ans) for emp, ans_list in zip(employees, post_answers) for q, ans in zip(questions, ans_list)],
    columns=["employee_id", "question_id", "answer"]
)

# Save files
pre_demo.to_excel("data/samples/pre_demo.xlsx", index=False)
post_demo.to_excel("data/samples/post_demo.xlsx", index=False)

print("✅ Demo files created in data/samples/")
