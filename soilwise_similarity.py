import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import json

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
    """
    Fetch a random record from database and generate embeddings for it.
    In production, this would be the newly harvested record.
    Only selects records that don't have similar_records already populated.
    """
    query = """
            SELECT
                identifier,
                title,
                abstract,
                keywords
            FROM
                records
            WHERE
                similar_records IS NULL
            ORDER BY RANDOM()
            LIMIT 1;
    """
    df = fetch_data(query)

    # Generate embeddings for the new record
    print("Generating embeddings for the new record...")
    df = generate_embeddings(df)

    return df


def retrieve_records_in_database():
    """
    Fetch all records with pre-computed embeddings from database.
    Records with NULL embeddings are included (where original fields were empty).
    """
    query = """
            SELECT
                identifier,
                title,
                abstract,
                keywords,
                embedding_title,
                embedding_abstract,
                embedding_keywords
            FROM
                records;
    """
    df = fetch_data(query)

    print(f"Retrieved {len(df)} records from database")

    # Convert PostgreSQL arrays to numpy arrays (handle NULL values)
    def convert_embedding(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return np.array(x, dtype='float32')
        if isinstance(x, np.ndarray):
            return x.astype('float32')
        # If it's a pandas NA value
        try:
            if pd.isna(x):
                return None
        except (TypeError, ValueError):
            pass
        return np.array(x, dtype='float32')

    print("Converting embeddings to numpy arrays...")
    df['embedding_title'] = df['embedding_title'].apply(convert_embedding)
    df['embedding_abstract'] = df['embedding_abstract'].apply(convert_embedding)
    df['embedding_keywords'] = df['embedding_keywords'].apply(convert_embedding)

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

    # Prepare text data from original fields
    titles = df['title'].apply(prepare_text).tolist()
    abstracts = df['abstract'].apply(prepare_text).tolist()
    keywords = df['keywords'].apply(prepare_text).tolist()

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
    """
    Calculate similarity between new record(s) and database records using FAISS.
    Handles records with missing embeddings by calculating similarities independently for each field.
    """

    print("\nFinding similar records using FAISS ANN (Approximate Nearest Neighbors)...")

    # Initialize dictionaries to store indices and similarities for each field
    title_similarities_dict = {}
    abstract_similarities_dict = {}
    keyword_similarities_dict = {}

    # Process title embeddings
    title_valid_mask = df_db['embedding_title'].notna()
    if title_valid_mask.any():
        df_db_title = df_db[title_valid_mask].copy()
        db_title_embeddings = np.vstack(df_db_title['embedding_title'].values).astype('float32')
        new_title_embeddings = np.vstack(df_new['embedding_title'].values).astype('float32')

        faiss.normalize_L2(db_title_embeddings)
        faiss.normalize_L2(new_title_embeddings)

        print(f"Creating FAISS index for titles ({len(df_db_title)} records)...")
        title_index = create_faiss_index(db_title_embeddings)

        k_title = min(k, len(df_db_title))
        title_similarities, title_indices = title_index.search(new_title_embeddings, k_title)

        # Map back to original df_db indices
        original_indices = df_db_title.index.values
        for i in range(len(df_new)):
            for j, db_idx in enumerate(title_indices[i]):
                orig_idx = original_indices[db_idx]
                title_similarities_dict[(i, orig_idx)] = float(title_similarities[i][j])
    else:
        print("No valid title embeddings found in database")

    # Process abstract embeddings
    abstract_valid_mask = df_db['embedding_abstract'].notna()
    if abstract_valid_mask.any():
        df_db_abstract = df_db[abstract_valid_mask].copy()
        db_abstract_embeddings = np.vstack(df_db_abstract['embedding_abstract'].values).astype('float32')
        new_abstract_embeddings = np.vstack(df_new['embedding_abstract'].values).astype('float32')

        faiss.normalize_L2(db_abstract_embeddings)
        faiss.normalize_L2(new_abstract_embeddings)

        print(f"Creating FAISS index for abstracts ({len(df_db_abstract)} records)...")
        abstract_index = create_faiss_index(db_abstract_embeddings)

        k_abstract = min(k, len(df_db_abstract))
        abstract_similarities, abstract_indices = abstract_index.search(new_abstract_embeddings, k_abstract)

        # Map back to original df_db indices
        original_indices = df_db_abstract.index.values
        for i in range(len(df_new)):
            for j, db_idx in enumerate(abstract_indices[i]):
                orig_idx = original_indices[db_idx]
                abstract_similarities_dict[(i, orig_idx)] = float(abstract_similarities[i][j])
    else:
        print("No valid abstract embeddings found in database")

    # Process keyword embeddings
    keyword_valid_mask = df_db['embedding_keywords'].notna()
    if keyword_valid_mask.any():
        df_db_keywords = df_db[keyword_valid_mask].copy()
        db_keyword_embeddings = np.vstack(df_db_keywords['embedding_keywords'].values).astype('float32')
        new_keyword_embeddings = np.vstack(df_new['embedding_keywords'].values).astype('float32')

        faiss.normalize_L2(db_keyword_embeddings)
        faiss.normalize_L2(new_keyword_embeddings)

        print(f"Creating FAISS index for keywords ({len(df_db_keywords)} records)...")
        keyword_index = create_faiss_index(db_keyword_embeddings)

        k_keyword = min(k, len(df_db_keywords))
        keyword_similarities, keyword_indices = keyword_index.search(new_keyword_embeddings, k_keyword)

        # Map back to original df_db indices
        original_indices = df_db_keywords.index.values
        for i in range(len(df_new)):
            for j, db_idx in enumerate(keyword_indices[i]):
                orig_idx = original_indices[db_idx]
                keyword_similarities_dict[(i, orig_idx)] = float(keyword_similarities[i][j])
    else:
        print("No valid keyword embeddings found in database")

    # Create results DataFrame
    results = []
    print(f"Finding top {k} similar records...")

    # For each new record (typically just 1)
    for i in range(len(df_new)):
        # Collect all unique database indices that appeared in any field's top-k
        all_indices = set()
        for key in title_similarities_dict.keys():
            if key[0] == i:
                all_indices.add(key[1])
        for key in abstract_similarities_dict.keys():
            if key[0] == i:
                all_indices.add(key[1])
        for key in keyword_similarities_dict.keys():
            if key[0] == i:
                all_indices.add(key[1])

        # For each unique database record in top-k
        for db_idx in all_indices:
            # Get similarity scores for this pair (None if not in top-k for that field)
            title_sim = title_similarities_dict.get((i, db_idx), None)
            abstract_sim = abstract_similarities_dict.get((i, db_idx), None)
            keyword_sim = keyword_similarities_dict.get((i, db_idx), None)

            # If embedding was NULL in database, similarity should be None (not 0.0)
            # This preserves the distinction between "not similar" and "no data"

            # Calculate average similarity (only from available fields)
            sim_scores = [s for s in [title_sim, abstract_sim, keyword_sim] if s is not None]
            if not sim_scores:
                continue

            # Average over all fields
            avg_sim = sum(sim_scores) / 3 #len(sim_scores), latter is to average over only valid embeddings

            results.append({
                'new_record_id': df_new.iloc[i]['identifier'],
                'db_record_id': df_db.iloc[db_idx]['identifier'],
                'new_record_title': df_new.iloc[i]['title'],
                'db_record_title': df_db.iloc[db_idx]['title'],
                'new_record_abstract': df_new.iloc[i]['abstract'][:100] + '...' if pd.notna(df_new.iloc[i]['abstract']) and len(str(df_new.iloc[i]['abstract'])) > 100 else df_new.iloc[i]['abstract'],
                'db_record_abstract': df_db.iloc[db_idx]['abstract'][:100] + '...' if pd.notna(df_db.iloc[db_idx]['abstract']) and len(str(df_db.iloc[db_idx]['abstract'])) > 100 else df_db.iloc[db_idx]['abstract'],
                'new_record_keywords': df_new.iloc[i]['keywords'],
                'db_record_keywords': df_db.iloc[db_idx]['keywords'],
                'title_similarity': title_sim,
                'abstract_similarity': abstract_sim,
                'keywords_similarity': keyword_sim,
                'avg_similarity': avg_sim
            })

    similarity_df = pd.DataFrame(results)
    similarity_df = similarity_df.sort_values('avg_similarity', ascending=False)

    return similarity_df


def update_similar_records_in_database(identifier, similarity_df, threshold=0.5):
    """
    Update the similar_records field in the database for a given record.
    Only stores records with avg_similarity > threshold.
    Excludes the record itself from the similar records list.
    Also updates the reverse relationship - adds the new record to each similar record's list.
    """
    # Filter similarities above threshold and exclude self
    filtered_df = similarity_df[
        (similarity_df['avg_similarity'] > threshold) &
        (similarity_df['db_record_id'] != identifier)
    ].copy()

    if len(filtered_df) == 0:
        print(f"No similar records found above threshold {threshold}")
        return

    # Create list of similar records for the new record
    similar_records = []
    for _, row in filtered_df.iterrows():
        similar_records.append({
            'identifier': row['db_record_id'],
            'title_similarity': row['title_similarity'],
            'abstract_similarity': row['abstract_similarity'],
            'keywords_similarity': row['keywords_similarity'],
            'avg_similarity': row['avg_similarity']
        })

    # Convert to JSON
    similar_records_json = json.dumps(similar_records)

    # Update database with retry logic
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            conn = get_db_connection()
            try:
                with conn.cursor() as cur:
                    # Update the new record's similar_records field
                    cur.execute("""
                        UPDATE records
                        SET similar_records = %s::jsonb
                        WHERE identifier = %s;
                    """, (similar_records_json, identifier))
                    print(f"✓ Updated similar_records field for {identifier}")
                    print(f"  - Stored {len(similar_records)} similar records (avg_similarity > {threshold})")

                    # Update reverse relationships - add new record to each similar record's list
                    print(f"  - Updating {len(filtered_df)} reverse relationships...")
                    for idx, row in filtered_df.iterrows():
                        db_record_id = row['db_record_id']

                        # Create the reverse similarity entry
                        reverse_entry = {
                            'identifier': identifier,
                            'title_similarity': row['title_similarity'],
                            'abstract_similarity': row['abstract_similarity'],
                            'keywords_similarity': row['keywords_similarity'],
                            'avg_similarity': row['avg_similarity']
                        }
                        reverse_entry_json = json.dumps(reverse_entry)

                        # Append to existing similar_records or create new array if NULL
                        cur.execute("""
                            UPDATE records
                            SET similar_records =
                                CASE
                                    WHEN similar_records IS NULL THEN %s::jsonb
                                    ELSE similar_records || %s::jsonb
                                END
                            WHERE identifier = %s;
                        """, (f'[{reverse_entry_json}]', reverse_entry_json, db_record_id))

                        # Print progress every 10 records
                        if (idx + 1) % 10 == 0:
                            print(f"    - Progress: {idx + 1}/{len(filtered_df)} records updated")

                    conn.commit()
                    print(f"✓ Updated reverse relationships for {len(filtered_df)} records")
                    break  # Success, exit retry loop

            except Exception as e:
                conn.rollback()
                raise
            finally:
                conn.close()

        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"⚠ Database error (attempt {retry_count}/{max_retries}): {e}")
                print(f"  Retrying in 2 seconds...")
                import time
                time.sleep(2)
            else:
                print(f"✗ Error updating database after {max_retries} attempts: {e}")
                raise


def main():
    # Fetch a random record from database (in production, this would be a newly harvested record)
    df_new = retrieve_newest_record()
    new_record_id = df_new.iloc[0]['identifier']
    print(f"Retrieved new record: {new_record_id}")
    print("\n" + "="*80 + "\n")
    print(df_new[['identifier', 'title', 'abstract', 'keywords']])
    print("\n" + "="*80 + "\n")

    # Fetch all other records with pre-computed embeddings from database
    df_db = retrieve_records_in_database()
    print("\n" + "="*80 + "\n")
    print(f"DataFrame shape: {df_db.shape}")
    print(f"Columns: {list(df_db.columns)}")
    print("="*80 + "\n")

    # Calculate similarity between new record and database records
    similarity_df = calculate_similarity_new_vs_db(df_new, df_db, k=num_simil)

    print("\n" + "="*80)
    print(f"Found {len(similarity_df)} similar records")
    print("="*80 + "\n")

    # Display top most similar records
    print("Top most similar records:")
    print(similarity_df[['new_record_id', 'db_record_id', 'new_record_title', 'db_record_title',
                         'title_similarity', 'abstract_similarity', 'keywords_similarity', 'avg_similarity']].head(10))

    print("\n" + "="*80 + "\n")

    # Save results to CSV
    output_file = 'similarity_results.csv'
    similarity_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Update similar_records field in database
    print("\n" + "="*80)
    print("Updating database with similar records...")
    print("="*80 + "\n")
    update_similar_records_in_database(new_record_id, similarity_df, threshold=0.5)

    print("\n" + "="*80 + "\n")



if __name__ == "__main__":
    main()


