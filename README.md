# IPL 2025 Sentiment Analysis

## (1) Problem Statement
Social media platforms like Twitter generate millions of opinions about trending topics every day. Analyzing the sentiment of these tweets manually is time-consuming and impractical. This project addresses the problem of automatically classifying tweets related to IPL 2025 as positive, neutral, or negative using machine learning classifiers.

## (2) Objective
- Collect 100 tweets related to IPL 2025 and manually label them as positive, neutral, or negative.
- Apply text preprocessing techniques to clean the tweet data.
- Train and evaluate three classifiers: Naïve Bayes, SVM, and Logistic Regression.
- Compare classifier performance using precision and recall metrics.
- Identify the best performing classifier for IPL 2025 tweet sentiment analysis.

## (3) Dataset
- Source: Manually collected from Twitter/X using keywords #IPL2025, #IPL, team and player names
- Features: Tweet text, Sentiment label (positive / neutral / negative)
- Size: 100 tweets — 40 positive, 30 neutral, 30 negative
- Split: 80 tweets for training, 20 tweets for testing

## (4) Methodology
1. Data Preprocessing — lowercasing, removing URLs, mentions, hashtags, punctuation, and stopwords
2. EDA — sentiment distribution bar chart and pie chart
3. Model Building — TF-IDF vectorisation (max 500 features, unigrams + bigrams), trained Naïve Bayes, SVM, and Logistic Regression
4. Evaluation — weighted precision and recall computed for each classifier, confusion matrices plotted

## (5) Results
| Classifier          | Precision | Recall |
|---------------------|-----------|--------|
| Naïve Bayes         | 0.5367    | 0.5000 |
| SVM                 | 0.6404    | 0.6000 |
| Logistic Regression | 0.7000    | 0.5500 |

- Best classifier: **SVM** — highest balanced precision and recall
- Logistic Regression achieved the highest precision but lower recall
- Naïve Bayes was the weakest performer on this dataset

## (6) How to Run
```bash
pip install pandas scikit-learn matplotlib seaborn numpy
jupyter notebook sentiment_analysis_ipl2025.ipynb
```

## (7) Conclusion
This project successfully performed sentiment analysis on 100 IPL 2025 tweets using three machine learning classifiers. SVM outperformed the others with a weighted precision of 0.64 and recall of 0.60, making it the most balanced classifier for this task. Future improvements could include using a larger dataset, BERT-based embeddings, and hyperparameter tuning for better accuracy.

## (8) Student's Details
- Name: Huzaifa Bhati
- Roll No: 13
- UIN: 231A064
- Year: TE-AIDS
