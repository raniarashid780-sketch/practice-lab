# Practice Lab 🐍

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Structured practice covering Python, data manipulation, visualization, SQL, and machine learning — progressing toward NLP and LLM engineering.

## Project Overview

This repository is a structured learning path covering Python fundamentals, data analysis, machine learning, SQL, ORM, API development, and hands-on practice. It is designed as a beginner-friendly collection of short lesson scripts and mini-projects rather than a single large application.

The current focus is on building practical skills in:
- Python basics and scripting
- Exploratory Data Analysis (EDA)
- NumPy, pandas, and matplotlib
- Scikit-learn and machine learning fundamentals
- SQL and SQLAlchemy
- FastAPI application development

## Current Repository Structure

| Folder | Contents | Status |
|---|---|---|
| `python-core/` | Variables, data types, conditions, loops, functions, OOP, files, pathlib, shutil | Completed |
| `NumPy/` | Arrays, indexing, broadcasting, reshaping, linear algebra, random operations | Completed |
| `pandas/` | Series/dataframes, CSV I/O, filtering, missing data, groupby, merge, pivot, datetime manipulation | Completed |
| `matplotlib/` | Plotting basics, chart types, styling, subplots | Completed |
| `eda-practice/` | Data quality audits, missing-value handling, outlier detection, univariate/bivariate analysis, question framing, feature-signal review | Completed |
| `sklearn-practice/` | Train/test splitting, stratification, baseline models, and classification metrics | In progress |
| `sql-practice/` | PostgreSQL exercises and SQL learning track | Completed |
| `sqlalchemy-practice/` | ORM basics, models, sessions, relationships, querying, Alembic setup | Completed |
| `fastapi-practice/` | FastAPI basics, validation, dependencies, async patterns, Swagger testing | Completed |
| `docker/` | Container basics, Docker setup, and image/container workflow practice | Completed |

## Learning Progress

### Python and Data Foundations
- Core Python scripting and logic practice in `python-core/`
- `NumPy/` for numerical operations and array thinking
- `pandas/` for data manipulation and dataset analysis
- `matplotlib/` for charting and data storytelling

### EDA Track
The `eda-practice/` folder is the active workstream for exploratory data analysis and real-world data understanding.

Current EDA topics include:
- Data quality audit
- Handling missing values and imputation
- Outlier detection
- Univariate and bivariate analysis
- Question framing and chart selection
- Feature signal and business interpretation

### Machine Learning Track
- Scikit-learn practice in `sklearn-practice/`, starting with stratified train/test splits and a majority-class baseline

### Database, API, and Docker Track
- SQL exercises in `sql-practice/`
- ORM work in `sqlalchemy-practice/`
- REST API learning in `fastapi-practice/`
- Docker and container workflow practice for deployment and environment setup

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python python-core/01_variables_and_print.py
python NumPy/09_numpy_basics.py
python pandas/20_pandas_series_dataframes.py
python sklearn-practice/day01_split_baseline.py
```

For EDA practice:
- Start with the raw data file : `dirty_cafe_sales.csv`
- Explore the scripts in `eda-practice/` in order
- Example:

```bash
python eda-practice/day01_data_quality_audit.py
```

For SQL practice:
- This project is set up for PostgreSQL, not SQLite.
- Use your own PostgreSQL database and client to run the SQL files in `sql-practice`.
- Example using `psql`:

```bash
cd sql-practice
psql -f day01.sql -d your_database_name
```

For SQLAlchemy practice:
- Use the environment installed from `requirements.txt`.
- Import SQLAlchemy in your scripts to practice connections, models, and ORM operations.
- Example:

```bash
python sqlalchemy-practice/day01_core_basics.py
```

For FastAPI practice:
- Activate the virtual environment before running the API.
- Start the development server from the `fastapi-practice` directory:

```bash
cd fastapi-practice
uvicorn day01_basics:app --reload --port 8001
```

- Open `http://127.0.0.1:8001/docs` to test the endpoints in Swagger UI.
- `GET /items/{item_id}` demonstrates typed path parameters and returns the item ID and its Python type.
- `GET /search` demonstrates a required query parameter and an optional query parameter with a default value.
- `GET /items/{item_id}/reviews` demonstrates a typed path parameter and an optional float query parameter.
- Invalid values such as `/items/abc` or `min_rating=abc` return HTTP `422` because FastAPI validation rejects values with the wrong type.

For Docker practice:
- Docker is included in the project requirements for container and image work.
- Run Docker commands from the project root or inside the relevant lesson folder.
- Example:

```bash
docker --version
cd fastapi-practice/day09_docker
docker build -t day09_docker .
```

## Author

Rania Rashid — BS Artificial Intelligence, Ghazi University DG Khan

## License

MIT — see [LICENSE](LICENSE) for details.