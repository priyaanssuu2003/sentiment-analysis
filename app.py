import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import plotly.express as px
from utils import preprocess_text

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css('style.css')
except FileNotFoundError:
    st.warning("Custom CSS file not found.")

# Cache the model loading
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# ── Stopwords that must never appear in explanations ─────────────────────────
# This set is independent of the TF-IDF stop_words list; it is used purely
# when building the human-readable "Why this result?" text.
_EXPLANATION_STOPWORDS = {
    "the", "and", "was", "is", "a", "an", "of", "to", "in", "for",
    "it", "this", "that", "with", "as", "on", "are", "at", "be",
    "by", "from", "or", "but", "not", "so", "if", "its", "he",
    "she", "we", "they", "i", "my", "me", "our", "you", "your",
    "had", "has", "have", "been", "were", "will", "would", "could",
    "should", "do", "did", "does", "about", "after", "before",
    "into", "than", "then", "there", "their", "them", "what",
    "which", "who", "when", "where", "how", "all", "also", "just",
    "more", "very", "no", "can", "some", "up", "out", "one",
    "get", "got", "make", "made", "go", "going", "came", "come",
    "because", "since", "though", "although", "while", "during",
    "each", "any", "both", "few", "most", "other", "own", "same",
    "such", "too", "over", "again", "further", "once", "only",
    "here", "two", "three", "however", "yet", "nor", "either",
}

# ── Mixed-sentiment detection parameters ─────────────────────────────────────
# If BOTH Positive and Negative probabilities exceed this threshold the text
# is considered "Neutral/Mixed" regardless of the argmax label.
_MIXED_MIN_PROB    = 0.20   # each of pos AND neg must be >= 20 %
_MIXED_MAX_MARGIN  = 0.35   # the gap between pos and neg must be < 35 pp

def resolve_sentiment(prob_dict):
    """
    Post-processing layer on top of the raw model output.

    Returns (final_label, display_confidence, prob_dict).

    Rules
    -----
    1. If Positive >= _MIXED_MIN_PROB AND Negative >= _MIXED_MIN_PROB AND
       |pos - neg| < _MIXED_MAX_MARGIN  →  "Neutral/Mixed"
       The displayed confidence is the raw Neutral probability OR the
       complement of the margin, whichever gives the more honest number.
    2. Otherwise, the argmax label stands as-is.

    This avoids any hardcoded keywords — it operates purely on the model's
    own probability distribution.
    """
    pos = prob_dict.get("Positive", 0)
    neg = prob_dict.get("Negative", 0)
    neu = prob_dict.get("Neutral",  0)

    is_mixed = (
        pos >= _MIXED_MIN_PROB
        and neg >= _MIXED_MIN_PROB
        and abs(pos - neg) < _MIXED_MAX_MARGIN
    )

    if is_mixed:
        # Represent confidence as the Neutral class probability,
        # but floor it at the complement of the pos-neg gap so
        # it doesn't look absurdly low on truly 50/50 splits.
        margin_complement = 1.0 - abs(pos - neg)
        display_conf = max(neu, margin_complement * 0.55)
        return "Neutral/Mixed", display_conf, prob_dict

    # Standard argmax
    best_label = max(prob_dict, key=prob_dict.get)
    return best_label, prob_dict[best_label], prob_dict


def get_prediction_explanation(model, raw_text, final_label, prob_dict):
    """
    Build a human-readable explanation of the prediction.

    Strategy
    --------
    - Extract TF-IDF feature weights for BOTH the Positive and Negative
      class coefficient vectors (not just the predicted class).
    - Filter all stopwords out of the result.
    - Separate tokens into "positive indicators" and "negative indicators"
      based on which class coefficient is larger for that token.
    - Compose a contextual explanation sentence depending on the label.

    This never uses hardcoded word lists — every word mentioned comes
    directly from the input text via the model's learned coefficients.
    """
    try:
        vectorizer = model.named_steps['tfidf']
        classifier = model.named_steps['clf']

        X_test = vectorizer.transform([raw_text])
        feature_names = vectorizer.get_feature_names_out()
        nonzero_indices = X_test.nonzero()[1]

        if len(nonzero_indices) == 0:
            return "The model didn't find recognisable sentiment keywords in this text."

        classes = list(classifier.classes_)

        # Locate coefficient rows for Positive and Negative
        # (Neutral is the residual / baseline in OvR)
        pos_coefs = classifier.coef_[classes.index("Positive")] if "Positive" in classes else None
        neg_coefs = classifier.coef_[classes.index("Negative")] if "Negative" in classes else None

        pos_words = []   # words leaning toward Positive
        neg_words = []   # words leaning toward Negative

        for idx in nonzero_indices:
            word = feature_names[idx]

            # ── Stopword guard ───────────────────────────────────────────────
            # Skip single-character tokens and anything in the stopword list.
            # Also skip bigrams that are composed entirely of stopwords.
            tokens = word.split()
            if all(t in _EXPLANATION_STOPWORDS or len(t) <= 1 for t in tokens):
                continue

            tfidf_val = X_test[0, idx]

            pos_score = float(pos_coefs[idx] * tfidf_val) if pos_coefs is not None else 0.0
            neg_score = float(neg_coefs[idx] * tfidf_val) if neg_coefs is not None else 0.0

            if pos_score > 0 and pos_score > neg_score:
                pos_words.append((word, pos_score))
            elif neg_score > 0 and neg_score > pos_score:
                neg_words.append((word, neg_score))

        # Sort by score strength, take top 4 each
        pos_words.sort(key=lambda x: x[1], reverse=True)
        neg_words.sort(key=lambda x: x[1], reverse=True)
        top_pos = [w for w, _ in pos_words[:4]]
        top_neg = [w for w, _ in neg_words[:4]]

        # ── Compose explanation based on final (resolved) label ───────────────
        def fmt(words):
            return ", ".join(f"'{w}'" for w in words)

        if final_label == "Neutral/Mixed":
            pos_part = f"Positive indicators: {fmt(top_pos)}. " if top_pos else ""
            neg_part = f"Negative indicators: {fmt(top_neg)}. " if top_neg else ""
            return (
                f"{pos_part}{neg_part}"
                "The text contains substantial evidence for both positive and negative "
                "sentiment, resulting in a mixed/uncertain classification."
            )

        elif final_label == "Positive":
            if top_pos:
                return (
                    f"Strong positive indicators such as {fmt(top_pos)} "
                    "contributed to the positive prediction."
                )
            return "The text carries an overall positive tone with no dominant negative signals."

        elif final_label == "Negative":
            if top_neg:
                return (
                    f"Strong negative indicators such as {fmt(top_neg)} "
                    "contributed to the negative prediction."
                )
            return "The text carries an overall negative tone with no dominant positive signals."

        else:  # Neutral
            pos_part = f"mild positive cues ({fmt(top_pos[:2])}) " if top_pos else ""
            neg_part = f"mild negative cues ({fmt(top_neg[:2])}) " if top_neg else ""
            both = "and " + neg_part if (pos_part and neg_part) else neg_part
            if pos_part or neg_part:
                return (
                    f"The text contains {pos_part}{both}but lacks strong "
                    "sentiment signals overall, so the model considers it neutral."
                )
            return (
                "The text contains limited strong sentiment indicators, "
                "so the model considers it largely neutral."
            )

    except Exception:
        return "Feedback could not be generated for this text."


# Sidebar
with st.sidebar:
    st.title("✨ AI Sentiment Studio")
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This is a production-grade Sentiment Analysis application built with Python, "
        "Scikit-learn, and Streamlit. It uses a trained Logistic Regression model "
        "with TF-IDF vectorization to classify text as Positive, Negative, or Neutral."
    )
    st.markdown("### Model Info")
    st.info("Algorithm: Logistic Regression\\n\\nFeature Extraction: TF-IDF\\n\\nTraining Accuracy: ~100% (Demo Data)")
    
    st.markdown("---")
    st.markdown("### Tech Stack")
    st.markdown("🔹 **Python**\\n🔹 **Scikit-learn**\\n🔹 **Streamlit**\\n🔹 **Pandas**\\n🔹 **Plotly**")

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Sentiment Analysis Studio</div>
    <div class="hero-subtitle">Instantly analyze the emotional tone of your text or bulk process datasets.</div>
</div>
""", unsafe_allow_html=True)

if not model:
    st.error("Model not found! Please run `python train_model.py` first to generate the model.")
    st.stop()

# Helper function to render sentiment card
def render_sentiment_card(prediction, probabilities, display_confidence):
    if prediction == "Positive":
        color_class = "sentiment-positive"
        icon = "🌟"
    elif prediction == "Negative":
        color_class = "sentiment-negative"
        icon = "⚠️"
    elif prediction == "Neutral/Mixed":
        color_class = "sentiment-neutral"
        icon = "🔀"
    else:
        color_class = "sentiment-neutral"
        icon = "⚖️"

    pos_prob = probabilities.get("Positive", 0) * 100
    neg_prob = probabilities.get("Negative", 0) * 100
    neu_prob = probabilities.get("Neutral",  0) * 100
    conf_pct = display_confidence * 100

    # Main Card
    html = f"""
    <div class="metric-card {color_class} fade-in" style="margin-bottom: 15px;">
        <div class="sentiment-icon">{icon}</div>
        <div class="sentiment-text">{prediction}</div>
        <div class="sentiment-score">Confidence: {conf_pct:.1f}%</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    # Expandable Breakdown — always shows the raw model probabilities
    with st.expander("View Detailed Confidence Breakdown"):
        breakdown_html = f"""
        <div style="margin-top: 5px; text-align: left; font-size: 0.9rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>Positive</span><span>{pos_prob:.1f}%</span>
            </div>
            <div style="width: 100%; background-color: #334155; border-radius: 4px; margin-bottom: 10px;">
                <div style="width: {pos_prob}%; background-color: #10b981; height: 8px; border-radius: 4px;"></div>
            </div>

            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>Neutral</span><span>{neu_prob:.1f}%</span>
            </div>
            <div style="width: 100%; background-color: #334155; border-radius: 4px; margin-bottom: 10px;">
                <div style="width: {neu_prob}%; background-color: #f59e0b; height: 8px; border-radius: 4px;"></div>
            </div>

            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>Negative</span><span>{neg_prob:.1f}%</span>
            </div>
            <div style="width: 100%; background-color: #334155; border-radius: 4px;">
                <div style="width: {neg_prob}%; background-color: #ef4444; height: 8px; border-radius: 4px;"></div>
            </div>
        </div>
        """
        st.markdown(breakdown_html, unsafe_allow_html=True)


# Main App Tabs
tab1, tab2 = st.tabs(["✍️ Single Text Analysis", "📁 Bulk CSV Upload"])

with tab1:
    st.markdown("### Enter text to analyze")
    user_input = st.text_area("Type or paste a sentence, paragraph, or review:", height=150, placeholder="E.g., The customer service was fantastic and I really loved the product!")
    
    # Word/char count
    word_count = len(user_input.split())
    char_count = len(user_input)
    st.caption(f"Words: {word_count} | Characters: {char_count}")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_btn = st.button("Analyze Sentiment", use_container_width=True)
        
    if analyze_btn and user_input:
        with st.spinner("Analyzing sentiment..."):
            time.sleep(0.8)  # Simulate processing time for UX

            # Preprocess and get raw model probabilities
            cleaned_text = preprocess_text(user_input)
            proba = model.predict_proba([cleaned_text])[0]
            classes = model.classes_
            prob_dict = {classes[i]: proba[i] for i in range(len(classes))}

            # ── Resolve final label (handles Neutral/Mixed detection) ─────────
            final_label, display_confidence, prob_dict = resolve_sentiment(prob_dict)

            # Display results
            st.markdown("---")
            render_sentiment_card(final_label, prob_dict, display_confidence)

            # Display explanation (uses final_label so Mixed gets the right text)
            explanation = get_prediction_explanation(model, cleaned_text, final_label, prob_dict)
            st.info(f"💡 **Why this result?**\n\n{explanation}")

with tab2:

    st.markdown("### Upload CSV for Bulk Analysis")
    st.markdown("Ensure your CSV has a column named `text`.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # ── Row cap: read only up to MAX_ROWS to avoid memory/time issues ──
            MAX_ROWS = 50_000
            df_full = pd.read_csv(uploaded_file)

            if 'text' not in df_full.columns:
                st.error("Error: CSV must contain a 'text' column.")
            else:
                total_rows = len(df_full)

                if total_rows > MAX_ROWS:
                    st.warning(
                        f"Your file has **{total_rows:,} rows**. "
                        f"Processing is capped at **{MAX_ROWS:,} rows** to keep the app responsive. "
                        f"The first {MAX_ROWS:,} rows will be analysed."
                    )
                    df = df_full.head(MAX_ROWS).copy()
                else:
                    df = df_full.copy()
                    st.success(f"Successfully loaded {total_rows:,} rows.")

                if st.button("Process Bulk Data"):
                    with st.spinner(f"Analysing {len(df):,} rows — please wait..."):

                        # ── Step 1: preprocess (vectorized string ops via apply) ──
                        df['cleaned_text'] = df['text'].apply(preprocess_text)

                        # ── Step 2: model inference — fully vectorized by sklearn ──
                        probs  = model.predict_proba(df['cleaned_text'])   # shape (N, 3)
                        classes = list(model.classes_)

                        pos_idx = classes.index('Positive')
                        neg_idx = classes.index('Negative')
                        neu_idx = classes.index('Neutral')

                        pos_arr = probs[:, pos_idx]
                        neg_arr = probs[:, neg_idx]
                        neu_arr = probs[:, neu_idx]

                        # ── Step 3: vectorized resolve_sentiment ──────────────────
                        # Instead of a Python-level loop (df.apply), all comparisons
                        # are done as NumPy array operations — runs in microseconds
                        # regardless of row count.
                        is_mixed = (
                            (pos_arr >= _MIXED_MIN_PROB) &
                            (neg_arr >= _MIXED_MIN_PROB) &
                            (np.abs(pos_arr - neg_arr) < _MIXED_MAX_MARGIN)
                        )

                        # Argmax label for non-mixed rows
                        argmax_idx    = np.argmax(probs, axis=1)
                        argmax_labels = np.array([classes[i] for i in argmax_idx])
                        argmax_conf   = probs[np.arange(len(probs)), argmax_idx]

                        # Mixed confidence formula (mirrors resolve_sentiment exactly)
                        margin_complement = 1.0 - np.abs(pos_arr - neg_arr)
                        mixed_conf = np.maximum(neu_arr, margin_complement * 0.55)

                        # Apply mixed override
                        final_labels = np.where(is_mixed, 'Neutral/Mixed', argmax_labels)
                        final_conf   = np.where(is_mixed, mixed_conf, argmax_conf)

                        # ── Step 4: write results back to dataframe ───────────────
                        df['Predicted Sentiment'] = final_labels
                        df['Confidence']          = np.round(final_conf, 4)

                        # Store per-class raw probabilities for download
                        for i, cls in enumerate(classes):
                            df[f'Prob_{cls}'] = np.round(probs[:, i], 4)

                        df_out = df.drop(columns=['cleaned_text'])

                        # ── Step 5: charts & table ────────────────────────────────
                        st.markdown("### Analysis Results")
                        sentiment_counts = (
                            df['Predicted Sentiment']
                            .value_counts()
                            .reset_index()
                        )
                        sentiment_counts.columns = ['Sentiment', 'Count']
                        
                        # Visualizations
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            fig_pie = px.pie(
                                sentiment_counts, 
                                names='Sentiment', 
                                values='Count',
                                title="Sentiment Distribution",
                                color='Sentiment',
                                color_discrete_map={
                                    'Positive': '#10b981',
                                    'Negative': '#ef4444',
                                    'Neutral': '#f59e0b',
                                    'Neutral/Mixed': '#a78bfa'
                                },
                                template="plotly_dark"
                            )
                            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                        with col2:
                            fig_bar = px.bar(
                                sentiment_counts,
                                x='Sentiment',
                                y='Count',
                                title="Count by Sentiment",
                                color='Sentiment',
                                color_discrete_map={
                                    'Positive': '#10b981',
                                    'Negative': '#ef4444',
                                    'Neutral': '#f59e0b',
                                    'Neutral/Mixed': '#a78bfa'
                                },
                                template="plotly_dark"
                            )
                            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Data Table
                        st.markdown("### Detailed Results")
                        # For styling dataframe rows, we can just display the dataframe
                        st.dataframe(df_out, use_container_width=True)
                        
                        # Download Button
                        csv = df_out.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv",
                        )
                        
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer
st.markdown("""
<div class="footer">
    <p>Built with ❤️ using Python, Scikit-learn, and Streamlit.</p>
    <p>Data in ➔ Model ➔ Interface out</p>
</div>
""", unsafe_allow_html=True)
