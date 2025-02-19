Project Overview: Sentiment Analysis on Product Reviews

Objective:
The goal of this project was to analyze customer sentiment from product reviews using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The project aimed to build a classification model to predict review scores (1-5) based on customer feedback text.

Key Project Phases & Activities:

1. Data Extraction & Preparation:
   - Extracted raw text data from zip files containing Amazon product reviews.
   - Created a structured DataFrame with fields: ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, ReviewSummary, ReviewText.
   - Data cleaning steps included removing unnecessary fields, converting data types, handling missing values, and dropping duplicates.

2. Data Exploration & Visualization:
   - Conducted Exploratory Data Analysis (EDA).
   - Visualized data using seaborn and matplotlib to identify patterns and anomalies in ratings.

3. Text Preprocessing:
   - Lowercasing and removal of special characters.
   - Tokenization of review text.
   - Stopwords removal (using NLTK).
   - Stemming (Porter Stemmer) and Lemmatization (WordNet Lemmatizer).
   - Applied custom preprocessing functions for consistent text cleaning across datasets.

4. Feature Engineering (Vectorization):
   - Bag of Words (BOW): Converted text to numerical form using CountVectorizer.
   - TF-IDF (Term Frequency - Inverse Document Frequency): Transformed text into weighted features using TfidfVectorizer.

5. Model Building & Evaluation:
   - Data Splitting: Train-Test Split (75%-25%).
   - Models Implemented:
     - Logistic Regression (BOW and TF-IDF).
     - Decision Tree Classifier.
     - Support Vector Classifier (attempted but not finalized).
   - Evaluation Metrics:
     - Accuracy, Precision, Recall, F1-Score.
   - Best Model Performance:
     - Logistic Regression (TF-IDF): Accuracy ~73.5%.
     - Decision Tree Classifier: Accuracy ~75.2%.

6. Model Deployment & User Testing:
   - Built interactive prediction functionality to classify new user reviews.
   - Example Inputs:
     - Positive Review: "It was amazing, never tried it." → Predicted Score: 5
     - Negative Review: "It was a bad food." → Predicted Score: 1
   - Model Serialization: Used Joblib to save trained models for future use.

Technologies & Libraries Used:
- Python: pandas, numpy, re, nltk, seaborn, matplotlib, sklearn, tqdm.
- NLP: NLTK (tokenization, stopwords, stemming, lemmatization).
- Machine Learning: Scikit-learn (Logistic Regression, Decision Tree, SVC).
- Data Processing: CountVectorizer, TfidfVectorizer.
- Model Deployment: Joblib.

Key Achievements:
- Processed 568,000+ product reviews efficiently.
- Developed multiple models for sentiment classification, achieving up to 75% accuracy.
- Implemented end-to-end NLP pipeline: Data extraction → Cleaning → Preprocessing → Modeling → Prediction.
- Interactive user testing feature allowing real-time review sentiment prediction.
- Gained hands-on experience with text data, feature engineering, and model evaluation.

Key Skills Demonstrated:
- Data Cleaning & Preprocessing.
- Natural Language Processing (NLP) techniques.
- Text Vectorization: BOW, TF-IDF.
- Supervised Machine Learning Models.
- Model Evaluation & Hyperparameter Tuning.
- Data Visualization & EDA.
- Deployment Ready Model Saving (Joblib).

Final Outcome:
The project successfully built a sentiment analysis system capable of predicting product review ratings based on customer review text, achieving reliable performance with Logistic Regression and Decision Tree models.

