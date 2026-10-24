# Natural Language Processing Assignment 1

## Project Overview

This project focuses on **Amharic Natural Language Processing (NLP)** using the GPAC corpus. It explores n-gram language models to analyze word sequences, estimate probabilities, evaluate model performance, and develop a predictive-text application.

## Team Members

| No. | Full Name | Student ID |
|---:|---|---|
| 1 | Beamlak Fekadu | UGR/8928/15 |
| 2 | Bemnet Asseged | UGR/2591/15 |
| 3 | Belen Berehanu | UGR/________/15 |
| 4 | Etsub Nadew | UGR/________/15 |
| 5 | Haset | UGR/________/15 |

## Project Objectives

- Prepare and analyze the Amharic GPAC corpus.
- Build unigram, bigram, trigram, and four-gram language models.
- Calculate n-gram frequencies and probabilities.
- Analyze frequent n-grams and generate sentences using language models.
- Evaluate language models using perplexity and next-word prediction.
- Develop an interactive predictive-text application.

## Technologies

- **Python** — language modeling and data processing
- **SQLite** — storing and querying n-gram counts
- **Streamlit** — interactive web application
- **Pandas** — data handling and results export

## Current Implementation

The initial predictive-text application uses **unigram and bigram statistics** from the GPAC corpus to suggest likely next words based on the last word entered.

Planned extensions include integrating higher-order n-gram models, completing model evaluation, and improving interactive sentence continuation.

## Running the Application

### 1. Install the dependency

```bash
pip install streamlit
```

### 2. Prepare the database

The application requires the SQLite database containing the processed n-gram counts. Update the database path in `app.py` to match the local environment.

### 3. Run the application

```bash
streamlit run app/app.py
```

## Repository Structure

```text
Natural_Language_Processing_Assignment_1/
├── README.md
└── app/
    └── app.py
```

Additional notebooks, corpus-processing scripts, evaluation results, and documentation may be added as the project progresses.

## Note

The GPAC corpus and generated SQLite database are not included in this repository. Each team member should follow the team's shared instructions for accessing the corpus and preparing the required data files.
