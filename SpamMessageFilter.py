# Importing libraries
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import warnings
warnings.filterwarnings('ignore')

# Downloading necessary NLTK requirements
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Loading our dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Dataset preview(df.head())
print("Dataset Preview:", df.head())

# Displaying info (df.info())
print("\nDataset Info:", df.info())

#shape of the dataset
print("\nDataset Shape:", df.shape)

# Checking duplicates and null values
print("\nChecking for duplicates:")
print(f"Number of duplicates: {df.duplicated().sum()}")

print("\nChecking for null values:")
print(f"Number of null values:\n{df.isnull().sum()}")

# Removing duplicates and null values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Label encoding
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Shape of dataset after cleaning
print("\nShape of the dataset after cleaning:", df.shape)

# Data visualization: Label Distributions of Spam and Ham Messages
plt.figure(figsize=(8, 5))
sns.countplot(x='label', data=df)
plt.title('Distribution of Spam and Ham Messages')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Text Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()  # Removing punctuation and lower case
    tokens = nltk.word_tokenize(text)  # text tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatization and removal of stopwords
    return ' '.join(tokens)

df['processed_message'] = df['message'].apply(preprocess_text)

# Character Length Distribution for Legitimate and Spam Messages
df['char_length'] = df['message'].apply(len)
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='char_length', hue='label', multiple='stack', bins=30)
plt.title('Character Length Distribution for Legitimate and Spam Messages')
plt.xlabel('Character Length')
plt.ylabel('Count')
plt.show()

# Pairplot for Data Visualization
sns.pairplot(df, hue='label')
plt.show()

# Comparison graph:before v/s after preprocessing
before_count = df['message'].str.split().str.len()
after_count = df['processed_message'].str.split().str.len()

plt.figure(figsize=(12, 6))
plt.hist(before_count, alpha=0.5, label='Before Preprocessing', bins=30)
plt.hist(after_count, alpha=0.5, label='After Preprocessing', bins=30)
plt.title('Comparison of Word Counts Before and After Preprocessing')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Word Cloud for Spam and Ham
plt.figure(figsize=(12, 6))
spam_words = ' '.join(df[df['label'] == 1]['processed_message'])
ham_words = ' '.join(df[df['label'] == 0]['processed_message'])

# Word Cloud for Spam
plt.subplot(1, 2, 1)
plt.title('Word Cloud for Spam Messages')
wordcloud_spam = WordCloud(width=800, height=400).generate(spam_words)
plt.imshow(wordcloud_spam, interpolation='bilinear')
plt.axis('off')

# Word Cloud for Ham
plt.subplot(1, 2, 2)
plt.title('Word Cloud for Ham Messages')
wordcloud_ham = WordCloud(width=800, height=400).generate(ham_words)
plt.imshow(wordcloud_ham, interpolation='bilinear')
plt.axis('off')

plt.show()

# Feature Extraction using Count vectorizer and TFIDF
count_vectorizer = CountVectorizer(max_features=3000)
tfidf_vectorizer = TfidfVectorizer(max_features=3000)

X_count = count_vectorizer.fit_transform(df['processed_message']).toarray()
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_message']).toarray()
y = df['label']

# Data Splitting
X_train_count, X_test_count, y_train, y_test = train_test_split(X_count, y, test_size=0.2, random_state=42)
X_train_tfidf, X_test_tfidf, _, _ = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Models and Hyperparameter Tuning with GridSearchCV
def evaluate_model(model, model_name, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

#Using Multiple Models
# Logistic Regression
log_reg = LogisticRegression()
param_grid_log = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']}
grid_log = GridSearchCV(log_reg, param_grid_log, cv=5, scoring='accuracy')
grid_log.fit(X_train_tfidf, y_train)

print(f"Best parameters for Logistic Regression: {grid_log.best_params_}")
evaluate_model(grid_log.best_estimator_, "Logistic Regression", X_test_tfidf, y_test)

# Random Forest
rf = RandomForestClassifier()
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_train_tfidf, y_train)

print(f"Best parameters for Random Forest: {grid_rf.best_params_}")
evaluate_model(grid_rf.best_estimator_, "Random Forest", X_test_tfidf, y_test)

# Naive Bayes
nb = MultinomialNB()
param_grid_nb = {'alpha': [0.5, 1.0, 1.5]}
grid_nb = GridSearchCV(nb, param_grid_nb, cv=5, scoring='accuracy')
grid_nb.fit(X_train_tfidf, y_train)

print(f"Best parameters for Naive Bayes: {grid_nb.best_params_}")
evaluate_model(grid_nb.best_estimator_, "Naive Bayes", X_test_tfidf, y_test)

# XGBoost
xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='accuracy')
grid_xgb.fit(X_train_tfidf, y_train)

print(f"Best parameters for XGBoost: {grid_xgb.best_params_}")
evaluate_model(grid_xgb.best_estimator_, "XGBoost", X_test_tfidf, y_test)

# Ensemble Method (Combining models based on Voting)
ensemble_model = VotingClassifier(
    estimators=[
        ('log_reg', grid_log.best_estimator_),
        ('rf', grid_rf.best_estimator_),
        ('nb', grid_nb.best_estimator_)
    ],
    voting='soft'
)

ensemble_model.fit(X_train_tfidf, y_train)

#Final Evaluation of Ensembled Models
evaluate_model(ensemble_model, "Ensemble Model", X_test_tfidf, y_test)

# Dumping the model
joblib.dump(ensemble_model, 'spam_classifier_model.joblib')

# Loading the model
loaded_model = joblib.load('spam_classifier_model.joblib')
print("Model loaded successfully!")

# Test run with a custom message
def test_run(message):
    processed_message = preprocess_text(message)
    message_vector = tfidf_vectorizer.transform([processed_message])
    prediction = loaded_model.predict(message_vector)
    return 'Spam' if prediction[0] == 1 else 'Ham'

test_message = "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize!"
print(f"Test Message: {test_message} - Prediction: {test_run(test_message)}")
