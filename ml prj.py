import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Improved text preprocessing
def wordopt(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabet characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Function to train and save models
def main():
    try:
        # Load and preprocess dataset
        df = pd.read_csv('C:/Users/Shagun Singh/Desktop/PBLprj/news_dataset.csv')
        print("Dataset loaded successfully!")

        df = df.dropna()
        df['text'] = df['text'].apply(wordopt)
        X = df['text']
        y = df['label']

        print("\nLabel distribution:")
        print(y.value_counts())

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Define TF-IDF Vectorizer
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_df=0.7,
            ngram_range=(1, 2),
            min_df=5
        )

        # Define the models
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'PassiveAggressive': PassiveAggressiveClassifier(max_iter=1000),
            'MultinomialNB': MultinomialNB(),
            'RandomForest': RandomForestClassifier(class_weight='balanced'),
            'GradientBoosting': GradientBoostingClassifier()
        }

        trained_models = {}
        best_model = None
        best_score = 0

        # Train each model and evaluate
        for name, model in models.items():
            pipeline = Pipeline([
                ('tfidf', tfidf),
                ('clf', model)
            ])
            pipeline.fit(X_train, y_train)
            trained_models[name] = pipeline

            # Evaluate the model
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\n{name} Accuracy: {accuracy:.2%}")
            print(classification_report(y_test, y_pred))

            # Confusion Matrix (printed and plotted)
            cm = confusion_matrix(y_test, y_pred)
            print(f"{name} Confusion Matrix:")
            print(cm)
            plot_confusion_matrix(cm, f"{name} Confusion Matrix")

            # Track best model
            if accuracy > best_score:
                best_score = accuracy
                best_model = pipeline

        # Save all models and the best model
        for name, model in trained_models.items():
            joblib.dump(model, f'{name}_model.pkl')
        joblib.dump(best_model, 'best_model.pkl')

        print("\nAll models and the best model saved successfully!")
        return trained_models

    except FileNotFoundError:
        print("Error: 'news_dataset.csv' not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Output human-readable label (supports both numeric and string labels)
def output_label(n):
    if n in [0, 'FAKE', 'fake', 'Fake']:
        return "Fake News"
    else:
        return "Not A Fake News"

# Manual prediction with majority voting
def manual_testing(news, models):
    try:
        processed_text = wordopt(news)
        predictions = {}
        votes = []
        for name, model in models.items():
            pred = model.predict([processed_text])
            label = output_label(pred[0])
            predictions[name] = label
            votes.append(label)

        print("\n=== Prediction Results ===")
        for name, prediction in predictions.items():
            print(f"{name} Prediction: {prediction}")

        # Majority voting for final decision
        final_vote = max(set(votes), key=votes.count)
        print("\nFinal Decision based on Majority Voting:", final_vote)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")

# Function to load models from saved .pkl files (if needed later)
def load_models():
    model_names = ['LogisticRegression', 'PassiveAggressive', 'MultinomialNB', 'RandomForest', 'GradientBoosting']
    models = {}
    for name in model_names:
        try:
            models[name] = joblib.load(f'{name}_model.pkl')
        except Exception as e:
            print(f"Could not load {name}: {e}")
    return models

# Main Execution
if __name__ == "__main__":
    trained_models = main()

    if trained_models:
        while True:
            news = input("\nEnter the news article to check (or 'quit' to exit): ")
            if news.lower() == 'quit':
                break
            manual_testing(news, trained_models)
