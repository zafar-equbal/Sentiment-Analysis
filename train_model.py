# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib



# Sample dataset
data = {
    'text': [
        'I love this product!', 'This is bad', 'Amazing experience',
        'Worst service ever', 'So happy', 'Not good', 'Excellent!'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative
}
df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# Create model pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'sentiment_model.pkl')
print("Model Trained and Saved!")
