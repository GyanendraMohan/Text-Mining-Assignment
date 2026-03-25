# Text Mining and Analysis Assignment

This repository contains three distinct machine learning projects focused on text mining, sentiment classification, and numerical user clustering. The assignment covers supervised classification algorithms (Naive Bayes) and unsupervised clustering (K-Means) alongside visual data exploration.

## 1. Sentiment Analysis on Reviews
**File**: `sentiment labelled sentences/Sentiment_Analysis.ipynb`

### Overview
This notebook classifies customer sentences extracted from three platforms (Amazon, IMDb, and Yelp) as either **Positive (1)** or **Negative (0)**. 

### Methodology
- **Data Loading**: Concatenates predefined datasets into a single balanced Pandas DataFrame.
- **Preprocessing**: Lowers text capitalizations, strips punctuation, and utilizes NLTK's `WordNetLemmatizer` alongside `stopwords` to clean the text into pure tokens.
- **Visualizations**: Uses `WordCloud` to map the most frequent words corresponding to 'Positive' vs. 'Negative' sentiments.
- **Vectorization**: Transforms the clean sentences into numerical arrays using `TfidfVectorizer` (Term Frequency-Inverse Document Frequency).
- **Modeling**: A `MultinomialNB` (Naive Bayes) classifier runs on an 80/20 train-test split, returning classification reports and a visual confusion matrix.
- **Testing**: Includes interactive input blocks testing sample text logic in real-time.

---

## 2. SMS Spam Classification
**File**: `SMSSpamCollection/Spam_Classification.ipynb`

### Overview
This notebook focuses on differentiating between genuine ("Ham") and junk ("Spam") SMS messages using natural language text characteristics. 

### Methodology
- **Data Structuring**: Reads tab-separated values mapping a text segment to its spam label.
- **Exploratory Data Analysis**: Evaluates dataset balance through Seaborn count plots and displays Word Clouds exposing urgency keywords ("Free", "Call", "Txt") largely confined to the spam corpus.
- **Preprocessing**: Mirrors the previous NLP structure—extracting regular expressions, lowering characters, and dropping common linguistic stop words.
- **Modeling Strategy**: Like the sentiment analysis task, the model employs `TfidfVectorizer` capped at high-variance features combined with `MultinomialNB`.
- **Validation**: Assesses model accuracy on a 20% test slice and showcases the confusion matrix showing extremely high Spam identification rates. 

---

## 3. TripAdvisor User Clustering
**File**: `TripAdvisor/TripAdvisor_Clustering.ipynb`

### Overview
Unlike the previous two datasets, the TripAdvisor dataset is purely numerical. It contains ratings normalized from 0 to 5 across 10 distinct evaluation categories mapped by `User ID`. Since text-mining doesn't apply here, this project segments users via mathematical clustering.

### Methodology
- **Feature Scaling**: Drops 'User ID' identifiers and scales variance equally across the 10 numeric dimensions using `StandardScaler`. This provides uniform weighting for mathematical distance.
- **Elbow Method (Optimal K)**: Plots the Within-Cluster Sum of Squares (WCSS) against an increasing number of neighbors (K=1 through 10) to identify the defining variance 'elbow' (k=3).
- **K-Means Clustering**: Instantiates an unsupervised KMeans model, defining categorical assignments for user sub-segments. 
- **PCA Visualization**: Reduces the original 10-dimensional space down to 2 principal components to permit 2D scatter plotting (`seaborn.scatterplot`), painting distinct geometric clusters.
- **Cluster Understanding**: Employs a semantic heatmap displaying the average profile rating across every category split perfectly by their discovered unsupervised cluster logic. 

---

## Technical Dependencies
- Python 3.x
- `pandas` / `numpy` (Structuring)
- `matplotlib` / `seaborn` (Data Visualizations)
- `wordcloud` (Visual word frequencies)
- `nltk` (NLP string operations)
- `scikit-learn` (Algorithms, Scaling, Vectorization, Splitting, and PCA)
