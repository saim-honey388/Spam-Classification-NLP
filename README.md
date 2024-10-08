# Spam Classification Project

This project focuses on building a **spam classification model** that distinguishes between **spam** and **ham** (legitimate) messages in SMS or email texts. The model leverages Natural Language Processing (NLP) techniques and machine learning algorithms for text preprocessing, feature extraction, and classification.

## Project Overview

The goal of this project is to create a model capable of identifying spam messages accurately. We utilize text preprocessing techniques like tokenization, stop word removal, and lemmatization, followed by feature extraction using **TF-IDF vectorization**. Several machine learning models are trained, and the best performing model is selected for spam detection. An ensemble model is also explored for boosting overall performance.

## Technologies Used

- **Python**: Main programming language for the project
- **Pandas**: For data manipulation and cleaning
- **NumPy**: For numerical operations
- **Scikit-learn**: For machine learning models and evaluation
- **NLTK**: For natural language preprocessing (stopwords, tokenization, etc.)
- **XGBoost**: For advanced gradient boosting models
- **Matplotlib & Seaborn**: For data visualization
- **Joblib**: For saving and loading trained models

## Key Features

- **Text Preprocessing**: Tokenization, stopword removal, and lemmatization to clean and prepare textual data.
- **Feature Extraction**: Implemented **TF-IDF Vectorization** to convert text into numerical features.
- **Model Training**: Trained multiple classifiers, including Logistic Regression, Naive Bayes, Random Forest, and XGBoost.
- **Ensemble Model**: Combined different models using **Voting Classifier** to boost accuracy.
- **Performance Evaluation**: Evaluated models using metrics like accuracy, confusion matrix, and classification reports.
  
## Model Accuracy

The project achieved competitive accuracy on the test dataset with the following results:

- **Logistic Regression**: ~97% accuracy
- **Random Forest**: ~96% accuracy
- **Naive Bayes**: ~95% accuracy
- **XGBoost**: ~96.5% accuracy
- **Voting Classifier**: ~97.2% accuracy

## How It Helped Me

- **Understanding NLP**: This project enhanced my understanding of text preprocessing techniques, feature extraction, and the complexities of dealing with textual data.
- **Experience with Machine Learning**: I gained experience working with various machine learning algorithms, hyperparameter tuning, and ensemble methods.
- **Real-World Applications**: It showcased the importance of accurate spam filtering for improving email and messaging system security.


To use the pre-trained model for spam detection, run the following:

1. Load the saved model and vectorizer:
    ```python
    import joblib
    model = joblib.load('models/spam_classifier_model.joblib')
    vectorizer = joblib.load('models/vectorizer.joblib')
    ```

2. Test with a custom message:
    ```python
    message = "Congratulations! You've won a $1,000 prize!"
    processed_message = vectorizer.transform([message])
    prediction = model.predict(processed_message)
    
    if prediction == 1:
        print("Spam")
    else:
        print("Ham")
    ```

## Dataset

The dataset used for this project comes from the SMS Spam Collection Dataset, which contains a collection of 5,572 messages labeled as **ham** or **spam**.

- [SMS Spam Collection Dataset on Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

## Future Improvements

- Implement **deep learning models** (e.g., LSTM, GRU) to further improve performance.
- Explore different feature extraction techniques like **Word2Vec** or **GloVe** for better text representation.
- Integrate the model into a **real-time spam detection system** using a web or mobile app interface.


#SpamDetection #MachineLearning #NLP #Python #DataScience #TextClassification #ArtificialIntelligence #GitHub
