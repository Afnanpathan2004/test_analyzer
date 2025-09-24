# Test Analysis App

A simple **Streamlit app** to analyze pre-test and post-test results of employees.

---

## Features
- Upload **pre-test** and **post-test** Excel/CSV files.
- Optional **answer key** for scoring.
- Automatically calculates:
  - Employee scores
  - Score improvement
  - Employees who gave pre-only, post-only, and both tests
  - Per-question statistics
- Download full report as Excel.

---

## Expected file format

**Test files (pre/post)**:
| employee_id | question_id | answer |
|------------|------------|--------|
| E101       | Q1         | A      |
| E101       | Q2         | C      |

**Answer key** (optional):
| question_id | correct_answer |
|------------|----------------|
| Q1         | A              |
| Q2         | C              |

Column names are case-insensitive. The app tries to auto-detect common names.

---

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
