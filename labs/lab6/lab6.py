import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK imports (already in your code)
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
# Make sure you have downloaded necessary NLTK data like 'punkt' and 'wordnet'
# nltk.download('punkt')
# nltk.download('wordnet')

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder # Use this in pipelines later
from scipy.sparse import hstack, csr_matrix


# --- Function: load_data ---
def load_data(file_path):
    """
    Load the dataset from a JSON file.
    """
    try:
        df = pd.read_json(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found. Ensure the file exists at the specified path: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {e}")
        return None

# --- Function: process_metadata ---
def process_metadata(df):
    """
    Processes and displays metadata for 'project_phase' and 'author_role'.
    """
    if df is None:
        print("DataFrame is None. Cannot process metadata.")
        return

    print("\n--- Processing Document Metadata ---")
    for col in ['project_phase', 'author_role']:
        print(f"\n--- Processing '{col}' ---")
        if col in df.columns:
            unique_values = df[col].unique()
            print(f"Unique values in '{col}':")
            # Limit printing unique values if there are too many
            if len(unique_values) > 20:
                print(unique_values[:20], '...')
            else:
                for val in unique_values:
                    print(f"- {val}")

            value_counts = df[col].value_counts(dropna=False) # include NaNs in counts
            print(f"\nCounts of documents per '{col}':")
            print(value_counts)
            print(f"Missing values in '{col}': {df[col].isnull().sum()}")
        else:
            print(f"'{col}' column not found.")
    print("-" * 50)

# --- Function: perform_one_hot_encoding (Original demo version) ---
# Keeping this here for reference to your original code, but we'll use
# OneHotEncoder with a pipeline later for better integration.
def perform_one_hot_encoding_dictvectorizer(df, columns_to_encode):
    """
    Performs one-hot encoding on specified categorical columns using DictVectorizer.
    (Original demo version - prefer OneHotEncoder for pipelines)
    """
    if df is None:
        print("DataFrame is None. Cannot perform one-hot encoding.")
        return None

    valid_columns_to_encode = [col for col in columns_to_encode if col in df.columns]
    if not valid_columns_to_encode:
        print(f"Error: None of the specified columns to encode ({columns_to_encode}) are in the DataFrame.")
        return None
    if len(valid_columns_to_encode) < len(columns_to_encode):
        missing = set(columns_to_encode) - set(valid_columns_to_encode)
        print(f"Warning: Columns {missing} not found for DictVectorizer one-hot encoding demo.")

    print("\n--- One-Hot Encoding with DictVectorizer (Original Demo) ---")
    print(f"Columns to encode: {valid_columns_to_encode}")

    # Convert selected columns to string to handle potential mixed types or NaNs
    data_dict = df[valid_columns_to_encode].astype(str).to_dict(orient='records')

    vectorizer = DictVectorizer(sparse=False)
    try:
        encoded_data = vectorizer.fit_transform(data_dict)
        feature_names = vectorizer.get_feature_names_out()
        # Create a DataFrame with original index to easily merge later
        df_encoded = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)

        print("\nFirst 5 rows of one-hot encoded data (DictVectorizer):")
        print(df_encoded.head())
        print(f"\nShape of encoded DataFrame: {df_encoded.shape}")
        return df_encoded
    except Exception as e:
        print(f"Error during DictVectorizer one-hot encoding: {e}")
        return None
    print("-" * 50)


# --- Function: vectorize_text_count ---
def vectorize_text_count(df, text_column='content'):
    """
    Converts text data in a specified column to numerical features
    using Count Vectorization.
    """
    if df is None:
        print("DataFrame is None. Cannot perform text vectorization.")
        return None, None
    if text_column not in df.columns:
        print(f"Error: Text column '{text_column}' not found in DataFrame.")
        return None, None

    print(f"\n--- Performing Count Vectorization on '{text_column}' ---")

    # Fill missing values in the text column with empty strings
    # This prevents errors during vectorization
    text_data = df[text_column].fillna('').astype(str)

    # Initialize CountVectorizer
    # max_features: Build a vocabulary that only considers the top
    # features ordered by term frequency across the corpus.
    # stop_words='english': Remove common English stop words.
    vectorizer = CountVectorizer(max_features=1000, stop_words='english') # Example parameters

    try:
        # Fit and transform the text data
        text_features = vectorizer.fit_transform(text_data)

        print(f"\nShape of text features (CountVectorizer): {text_features.shape}")
        print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
        # print("First 5 rows of vectorized text data (sparse matrix representation):")
        # print(text_features[:5]) # Prints sparse representation

        return text_features, vectorizer

    except Exception as e:
        print(f"An error occurred during Count Vectorization: {e}")
        return None, None

# --- Function: visualize_doc_types_by_phase ---
def visualize_doc_types_by_phase(df):
    """
    Visualizes the distribution of document types across project phases using a bar plot.
    """
    if df is None:
        print("DataFrame is None. Cannot perform visualization.")
        return
    if not all(col in df.columns for col in ['project_phase', 'document_type']):
        print("Error: 'project_phase' or 'document_type' column missing for visualization.")
        return

    print("\n--- Visualizing Document Types Across Project Phases ---")
    df_viz = df.copy()
    # Fill NaN values in relevant columns for visualization purposes
    df_viz['project_phase'] = df_viz['project_phase'].fillna('Unknown_Phase')
    df_viz['document_type'] = df_viz['document_type'].fillna('Unknown_Type')


    cross_tab = pd.crosstab(df_viz['project_phase'], df_viz['document_type'])
    if cross_tab.empty:
        print("Cross-tabulation is empty. Cannot generate plot.")
        return

    print("\nCross-tabulation of Project Phase vs. Document Type:")
    print(cross_tab)

    sns.set_style("whitegrid")
    # Use stacked bar chart if it makes sense, otherwise separate bars
    cross_tab.plot(kind='bar', figsize=(14, 8), colormap='viridis', stacked=False)
    plt.title('Distribution of Document Types Across Project Phases', fontsize=16)
    plt.xlabel('Project Phase', fontsize=12)
    plt.ylabel('Number of Documents', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title='Document Type')
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()
    print("-" * 50)


# --- Main Execution Block ---
def main():

    file_path = 'construction_documents.json'
    df_original = load_data(file_path)
    if df_original is None:
        return

    df = df_original.copy()

    print("\n--- Metadata ---")
    process_metadata(df.copy())

    # Keeping the original DictVectorizer demo for comparison if needed
    # print("\n--- Demonstrating One-Hot Encoding with DictVectorizer ---")
    # columns_to_encode_demo = ['project_phase', 'author_role']
    # df_dictvectorizer_encoded_features = perform_one_hot_encoding_dictvectorizer(df.copy(), columns_to_encode_demo)
    visualize_doc_types_by_phase(df.copy())

    # --- New: Perform Count Vectorization ---
    text_features_count, count_vectorizer = vectorize_text_count(df.copy(), 'content')

    # You can now use text_features_count (a sparse matrix) for modeling
    # and count_vectorizer to inspect features if needed.

    # Example: Print top words by frequency (simple sum across documents)
    if text_features_count is not None and count_vectorizer is not None:
        print("\n--- Top words by frequency (CountVectorizer) ---")
        sum_words = text_features_count.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in count_vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        print("Top 10 words and their total counts:")
        for word, freq in words_freq[:10]:
            print(f"- {word}: {freq}")
        print("-" * 50)


if __name__ == '__main__':
    main()


from sklearn.naive_bayes import MultinomialNB # Moved import here as it was used below main()

# Step 1: Load dataset
data1 = load_data('construction_documents.json')
data = data1.copy()

# Step 2: Extract texts and labels
documents = data['content'].fillna('')
categories = data['document_type']

# Step 3: TF-IDF vectorization with unigrams and bigrams
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=1000
)
X_tfidf = vectorizer.fit_transform(documents)
terms = np.array(vectorizer.get_feature_names_out())

# Step 4: Split data and train Naive Bayes model
X_train, X_val, y_train, y_val = train_test_split(
    X_tfidf, categories,
    test_size=0.2,
    stratify=categories,
    random_state=42
)
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Step 5: Extract top 10 indicative terms per class
important_terms = set()
for idx, label in enumerate(clf.classes_):
    top_indices = np.argsort(clf.feature_log_prob_[idx])[-10:]
    important_terms.update(terms[top_indices])
important_terms = sorted(important_terms)
term_indices = [np.where(terms == word)[0][0] for word in important_terms]

# Step 6: Compute scaled Pearson correlations
correlation_table = pd.DataFrame(index=clf.classes_, columns=important_terms)

for label in clf.classes_:
    binary_label = (categories == label).astype(int).values
    for word in important_terms:
        col_idx = term_indices[important_terms.index(word)]
        word_scores = X_tfidf[:, col_idx].toarray().ravel()
        if word_scores.std() == 0:
            score = 0.0
        else:
            score = np.corrcoef(binary_label, word_scores)[0, 1]
        correlation_table.loc[label, word] = (score + 1) / 2

# Step 7: Display correlation matrix
print("\n[+] Scaled correlation (0–1) between classes and selected keywords:\n")
print(correlation_table)

# Step 8: Visualize with heatmap
plt.figure(figsize=(13, 6))
plt.imshow(correlation_table.astype(float), cmap='plasma', vmin=0, vmax=1)
plt.title('TF-IDF Term Correlation by Document Type')
plt.colorbar(label='Correlation Score')
plt.xticks(ticks=range(len(important_terms)), labels=important_terms, rotation=90)
plt.yticks(ticks=range(len(clf.classes_)), labels=clf.classes_)
plt.tight_layout()
plt.show()