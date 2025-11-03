import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import faiss
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 15432,
    'database': 'soilwise',
    'user': os.getenv('postgres_user', 'soilwise').strip("'"),
    'password': os.getenv('postgres_secret', '').strip("'")
}

# set configuration parameters
num_simil =100


def get_db_connection():
    """Create and return a database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Successfully connected to PostgreSQL database")
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        raise


def fetch_data(query):
    """Execute a query and return results as a pandas DataFrame."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def retrieve_newest_record():
    """Fetch most recent record from database."""
    query = """
            SELECT
                identifier,
                title,
                abstract,
                keywords
            FROM
                records
            ORDER BY
                date_creation DESC
            LIMIT 1;
    """
    query = """
            SELECT
                identifier,
                title,
                abstract,
                keywords
            FROM
                records
            LIMIT 1;
    """
    df = fetch_data(query)
    df['embedding_title'] = df.title
    df['embedding_abstract'] = df.abstract
    df['embedding_keywords'] = df.keywords
    return df


def retrieve_records_in_database():
    """Fetch records from database."""
    query = """
            SELECT
                identifier,
                title,
                abstract,
                keywords
            FROM
                records
            LIMIT 1000;
    """
    df = fetch_data(query)
    df['embedding_title'] = df.title
    df['embedding_abstract'] = df.abstract
    df['embedding_keywords'] = df.keywords
    return df


def generate_embeddings(df):
    """Generate embeddings using Sentence Transformers for better and faster embeddings."""

    # Load model optimized for semantic similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')  

    # Function to handle null values and convert to string
    def prepare_text(text):
        if pd.isnull(text):
            return ""
        return str(text)

    # Prepare text data
    titles = df['embedding_title'].apply(prepare_text).tolist()
    abstracts = df['embedding_abstract'].apply(prepare_text).tolist()
    keywords = df['embedding_keywords'].apply(prepare_text).tolist()

    # Batch processing for better performance
    print("Generating embeddings for titles...")
    title_embeddings = model.encode(titles, batch_size=32, show_progress_bar=True)

    print("Generating embeddings for abstracts...")
    abstracts_embeddings = model.encode(abstracts, batch_size=32, show_progress_bar=True)

    print("Generating embeddings for keywords...")
    keyword_embeddings = model.encode(keywords, batch_size=32, show_progress_bar=True)

    # Assign embeddings back to DataFrame
    df['embedding_title'] = list(title_embeddings)
    df['embedding_abstract'] = list(abstracts_embeddings)
    df['embedding_keywords'] = list(keyword_embeddings)
    return df


def create_faiss_index(embeddings):
    """Create a FAISS index for the given embeddings"""
    dimension = embeddings.shape[1]
    # Create FAISS index (IndexFlatIP for inner product = cosine similarity after normalization)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def calculate_similarity_new_vs_db(df_new, df_db, k=20):
    """Calculate similarity between new record(s) and database records using FAISS"""

    print("\nFinding similar records using FAISS ANN (Approximate Nearest Neighbors)...")

    # Extract embeddings from database records as numpy arrays (float32 for FAISS)
    db_title_embeddings = np.vstack(df_db['embedding_title'].values).astype('float32')
    db_abstract_embeddings = np.vstack(df_db['embedding_abstract'].values).astype('float32')
    db_keyword_embeddings = np.vstack(df_db['embedding_keywords'].values).astype('float32')

    # Extract embeddings from new record(s)
    new_title_embeddings = np.vstack(df_new['embedding_title'].values).astype('float32')
    new_abstract_embeddings = np.vstack(df_new['embedding_abstract'].values).astype('float32')
    new_keyword_embeddings = np.vstack(df_new['embedding_keywords'].values).astype('float32')

    # Normalize embeddings for cosine similarity (FAISS uses L2 distance)
    faiss.normalize_L2(db_title_embeddings)
    faiss.normalize_L2(db_abstract_embeddings)
    faiss.normalize_L2(db_keyword_embeddings)
    faiss.normalize_L2(new_title_embeddings)
    faiss.normalize_L2(new_abstract_embeddings)
    faiss.normalize_L2(new_keyword_embeddings)

    # Create FAISS indices for database records
    print("Creating FAISS index for titles...")
    title_index = create_faiss_index(db_title_embeddings)

    print("Creating FAISS index for abstracts...")
    abstract_index = create_faiss_index(db_abstract_embeddings)

    print("Creating FAISS index for keywords...")
    keyword_index = create_faiss_index(db_keyword_embeddings)

    # Number of similar records to find
    k = min(k, len(df_db))

    # Search for similar records
    print(f"Finding top {k} similar records...")
    title_similarities, title_indices = title_index.search(new_title_embeddings, k)
    abstract_similarities, abstract_indices = abstract_index.search(new_abstract_embeddings, k)
    keyword_similarities, keyword_indices = keyword_index.search(new_keyword_embeddings, k)

    # Create results DataFrame
    results = []

    # For each new record (typically just 1)
    for i in range(len(df_new)):
        # Collect all unique indices from top-k results across all fields
        all_indices = set()
        all_indices.update(title_indices[i])
        all_indices.update(abstract_indices[i])
        all_indices.update(keyword_indices[i])

        # For each unique database record in top-k
        for db_idx in all_indices:
            # Find similarity scores for this pair
            title_sim = None
            if db_idx in title_indices[i]:
                pos = np.where(title_indices[i] == db_idx)[0][0]
                title_sim = float(title_similarities[i][pos])

            abstract_sim = None
            if db_idx in abstract_indices[i]:
                pos = np.where(abstract_indices[i] == db_idx)[0][0]
                abstract_sim = float(abstract_similarities[i][pos])

            keyword_sim = None
            if db_idx in keyword_indices[i]:
                pos = np.where(keyword_indices[i] == db_idx)[0][0]
                keyword_sim = float(keyword_similarities[i][pos])

            # Handle None values in the original data
            if pd.isna(df_new.iloc[i]['title']) or pd.isna(df_db.iloc[db_idx]['title']):
                title_sim = 0.0

            if pd.isna(df_new.iloc[i]['abstract']) or pd.isna(df_db.iloc[db_idx]['abstract']):
                abstract_sim = 0.0

            if pd.isna(df_new.iloc[i]['keywords']) or pd.isna(df_db.iloc[db_idx]['keywords']):
                keyword_sim = 0.0

            # Calculate average similarity
            sim_scores = [s for s in [title_sim, abstract_sim, keyword_sim] if s is not None]
            if not sim_scores:
                continue

            avg_sim = sum(sim_scores) / 3

            results.append({
                'new_record_id': df_new.iloc[i]['identifier'],
                'db_record_id': df_db.iloc[db_idx]['identifier'],
                'new_record_title': df_new.iloc[i]['title'],
                'db_record_title': df_db.iloc[db_idx]['title'],
                'new_record_abstract': df_new.iloc[i]['abstract'][:100] + '...' if pd.notna(df_new.iloc[i]['abstract']) and len(str(df_new.iloc[i]['abstract'])) > 100 else df_new.iloc[i]['abstract'],
                'db_record_abstract': df_db.iloc[db_idx]['abstract'][:100] + '...' if pd.notna(df_db.iloc[db_idx]['abstract']) and len(str(df_db.iloc[db_idx]['abstract'])) > 100 else df_db.iloc[db_idx]['abstract'],
                'new_record_keywords': df_new.iloc[i]['keywords'],
                'db_record_keywords': df_db.iloc[db_idx]['keywords'],
                'title_similarity': title_sim if title_sim is not None else 0.0,
                'abstract_similarity': abstract_sim if abstract_sim is not None else 0.0,
                'keywords_similarity': keyword_sim if keyword_sim is not None else 0.0,
                'avg_similarity': avg_sim
            })

    similarity_df = pd.DataFrame(results)
    similarity_df = similarity_df.sort_values('avg_similarity', ascending=False)

    return similarity_df


def main():
    # Fetch most recent record from database
    df_new = retrieve_newest_record()
    print(f"Retrieved {len(df_new)} records:")
    print("\n" + "="*80 + "\n")
    print(df_new)
    print("\n" + "="*80 + "\n")
    print(f"\nDataFrame shape: {df_new.shape}")
    print(f"Columns: {list(df_new.columns)}")

    # Fetch other records from database
    df_db = retrieve_records_in_database()
    print(f"Retrieved {len(df_db)} records:")
    print("\n" + "="*80 + "\n")
    print(df_db)
    print("\n" + "="*80 + "\n")
    print(f"\nDataFrame shape: {df_db.shape}")
    print(f"Columns: {list(df_db.columns)}")

    # Generate embeddings title, abstract and keywords proporties of each record
    # TODO: now embeddings are calculated, but ideally they already exist in the database and can be extracted from there in first query
    df_new = generate_embeddings(df_new)
    df_db = generate_embeddings(df_db)

    # Calculate similarity between new record and database records
    similarity_df = calculate_similarity_new_vs_db(df_new, df_db, k=num_simil)

    print("\n" + "="*80)
    print(f"Found {len(similarity_df)} similar records")
    print("="*80 + "\n")

    # Display top  most similar records
    print("Top most similar records:")
    print(similarity_df[['new_record_id', 'db_record_id', 'new_record_title', 'db_record_title',
                         'title_similarity', 'abstract_similarity', 'keywords_similarity', 'avg_similarity']].head(10))

    print("\n" + "="*80 + "\n")

    # Save results to CSV
    output_file = 'similarity_results.csv'
    similarity_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    print("="*80 + "\n")



if __name__ == "__main__":
    main()


