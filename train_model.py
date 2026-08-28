import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os
from utils import preprocess_text

def train_and_save_model():
    print("Training model...")

    # ── Training corpus ──────────────────────────────────────────────────────
    # Each list contains representative sentences for that label.
    # Mixed/ambiguous examples are labelled "Neutral" so the model learns that
    # texts containing both positive and negative signals sit in the middle ground.
    data = {
        'text': [
            # ── POSITIVE ──────────────────────────────────────────────────────
            "I absolutely loved this movie, it was fantastic and brilliant!",
            "The food was amazing and the service was excellent.",
            "Highly recommended! Will definitely buy again.",
            "Such a beautiful design and works flawlessly.",
            "I am so happy with this purchase. It exceeded all my expectations.",
            "Incredible quality and very fast delivery. Totally satisfied.",
            "Outstanding performance, I am thrilled with the results.",
            "Best experience ever. The staff was friendly and professional.",
            "I feel grateful and optimistic about the future.",
            "Wonderful product. I am excited to share this with my friends.",
            "Super helpful customer support. Everything was resolved quickly.",
            "Delighted with the outcome. Would strongly recommend to anyone.",
            "The app works perfectly. Really impressed by how smooth it is.",
            "Exceptional value for money. Very pleased overall.",
            "I appreciate the effort put into this. Truly hopeful about the next version.",

            # ── NEGATIVE ──────────────────────────────────────────────────────
            "Terrible experience, the worst product I have ever bought.",
            "I hate this, it broke after one use.",
            "Awful, horrible, do not buy this.",
            "I am extremely disappointed with the poor quality and frustrating service.",
            "Complete waste of money. The product stopped working after two days.",
            "Absolutely dreadful. I feel ignored and disrespected.",
            "I am so frustrated and irritated with this company.",
            "The service was rude, slow, and unhelpful. Very bad experience.",
            "I regret this purchase. Nothing works as advertised.",
            "Unacceptable quality. I would never recommend this to anyone.",
            "The worst customer support I have ever dealt with.",
            "I am angry and disgusted with how this was handled.",
            "Very disappointing. The product is defective and useless.",
            "Terrible. I feel cheated and let down completely.",
            "Nothing but problems since day one. Highly dissatisfied.",

            # ── NEUTRAL / MIXED ───────────────────────────────────────────────
            "It was okay, not great but not bad either. Just average.",
            "Meh, it does the job but nothing special.",
            "It is exactly what it says it is.",
            "The meeting was held at 10 AM and the report was submitted to the manager.",
            "The package arrived on time and the item was as described.",
            "I was initially excited but later became frustrated. Overall uncertain.",
            "Some things were good but other aspects were disappointing.",
            "Happy with the design but disappointed with the build quality.",
            "The staff was friendly but the product itself was subpar.",
            "I appreciate the effort but I am still uncertain if it was worth it.",
            "It has its pros and cons. I am not sure I would buy again.",
            "Good in some areas but lacking in others. Mixed feelings overall.",
            "The first half was great but the second half was a letdown.",
            "I liked the concept but the execution was poor.",
            "Satisfied with some parts, dissatisfied with others.",
            "I feel hopeful about the improvements but also frustrated by the delays.",
            "This is neither a great product nor a terrible one.",
            "The conference covered interesting topics but the organisation was poor.",
            "I got what I paid for, nothing more and nothing less.",
            "Some features are excellent while others feel incomplete.",
        ] * 20,  # Multiply so each class has enough samples (~300 per class)

        'label': [
            # 15 Positive
            "Positive", "Positive", "Positive", "Positive", "Positive",
            "Positive", "Positive", "Positive", "Positive", "Positive",
            "Positive", "Positive", "Positive", "Positive", "Positive",
            # 15 Negative
            "Negative", "Negative", "Negative", "Negative", "Negative",
            "Negative", "Negative", "Negative", "Negative", "Negative",
            "Negative", "Negative", "Negative", "Negative", "Negative",
            # 20 Neutral
            "Neutral", "Neutral", "Neutral", "Neutral", "Neutral",
            "Neutral", "Neutral", "Neutral", "Neutral", "Neutral",
            "Neutral", "Neutral", "Neutral", "Neutral", "Neutral",
            "Neutral", "Neutral", "Neutral", "Neutral", "Neutral",
        ] * 20
    }

    df = pd.DataFrame(data)

    # Preprocess text
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    X = df['cleaned_text']
    y = df['label']

    # ── Pipeline ──────────────────────────────────────────────────────────────
    # stop_words='english' prevents function words (the, and, was, is…)
    # from occupying feature slots and leaking into explanations.
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',   # ← KEY FIX: removes stopwords from features
            sublinear_tf=True,      # log(1+tf) dampens high-frequency terms
        )),
        ('clf', LogisticRegression(
            random_state=42,
            multi_class='ovr',
            C=1.0,
            max_iter=1000,
        ))
    ])

    pipeline.fit(X, y)

    model_path = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    accuracy = pipeline.score(X, y)
    print(f"Training Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    train_and_save_model()

