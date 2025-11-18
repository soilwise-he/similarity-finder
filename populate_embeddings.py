"""
Populate Embeddings in Database
================================
This script generates and stores embeddings for title, abstract, and keywords fields
in the records table. It processes records in batches to avoid memory issues.

Usage:
    python populate_embeddings.py
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_batch
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time

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

# Configuration
BATCH_SIZE = 5000
MODEL_NAME = 'all-MiniLM-L6-v2'


def get_db_connection():
    """Create and return a database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Successfully connected to PostgreSQL database")
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        raise


def get_total_records_without_embeddings():
    """Get count of records that need embeddings."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT COUNT(*)
                FROM records
                WHERE embedding_title IS NULL
                   OR embedding_abstract IS NULL
                   OR embedding_keywords IS NULL;
            """
            cur.execute(query)
            count = cur.fetchone()[0]
            return count
    finally:
        conn.close()


def fetch_batch_without_embeddings(batch_size=BATCH_SIZE):
    """
    Fetch a batch of records that don't have embeddings yet.
    Returns a DataFrame with identifier, title, abstract, and keywords.
    """
    conn = get_db_connection()
    try:
        query = """
            SELECT
                identifier,
                title,
                abstract,
                keywords
            FROM
                records
            WHERE
                embedding_title IS NULL
                OR embedding_abstract IS NULL
                OR embedding_keywords IS NULL
            LIMIT %s;
        """
        df = pd.read_sql_query(query, conn, params=(batch_size,))
        return df
    finally:
        conn.close()


def prepare_text(text):
    """Handle null values and convert to string."""
    if pd.isnull(text):
        return ""
    return str(text)


def generate_embeddings_for_batch(df, model):
    """
    Generate embeddings for a batch of records.
    Returns the DataFrame with embeddings added.
    """
    print(f"  Generating embeddings for {len(df)} records...")

    # Prepare text data
    titles = df['title'].apply(prepare_text).tolist()
    abstracts = df['abstract'].apply(prepare_text).tolist()
    keywords = df['keywords'].apply(prepare_text).tolist()

    # Generate embeddings
    print("    - Processing titles...")
    title_embeddings = model.encode(titles, batch_size=32, show_progress_bar=False)

    print("    - Processing abstracts...")
    abstracts_embeddings = model.encode(abstracts, batch_size=32, show_progress_bar=False)

    print("    - Processing keywords...")
    keyword_embeddings = model.encode(keywords, batch_size=32, show_progress_bar=False)

    # Add embeddings to DataFrame as lists (for PostgreSQL array insertion)
    df['embedding_title'] = [emb.tolist() for emb in title_embeddings]
    df['embedding_abstract'] = [emb.tolist() for emb in abstracts_embeddings]
    df['embedding_keywords'] = [emb.tolist() for emb in keyword_embeddings]

    return df


def update_embeddings_in_database(df):
    """
    Update the database with generated embeddings.
    Uses execute_batch for efficient bulk updates.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Prepare update query
            update_query = """
                UPDATE records
                SET
                    embedding_title = %s,
                    embedding_abstract = %s,
                    embedding_keywords = %s
                WHERE identifier = %s;
            """

            # Prepare data for batch update
            update_data = [
                (
                    row['embedding_title'],
                    row['embedding_abstract'],
                    row['embedding_keywords'],
                    row['identifier']
                )
                for _, row in df.iterrows()
            ]

            # Execute batch update
            print(f"  Updating {len(update_data)} records in database...")
            execute_batch(cur, update_query, update_data, page_size=100)
            conn.commit()
            print(f"  ✓ Successfully updated {len(update_data)} records")

    except Exception as e:
        conn.rollback()
        print(f"  ✗ Error updating database: {e}")
        raise
    finally:
        conn.close()


def main():
    """Main function to orchestrate the embedding population process."""
    print("="*80)
    print("Embedding Population Script")
    print("="*80)

    # Get total count
    total_records = get_total_records_without_embeddings()
    print(f"\nTotal records needing embeddings: {total_records}")

    if total_records == 0:
        print("No records need embeddings. Exiting.")
        return

    # Load the model once
    print(f"\nLoading SentenceTransformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("✓ Model loaded successfully")

    # Process in batches
    processed_count = 0
    batch_number = 0

    print(f"\nProcessing in batches of {BATCH_SIZE} records...")
    print("="*80)

    start_time = time.time()

    while processed_count < total_records:
        batch_number += 1
        print(f"\nBatch {batch_number}:")

        # Fetch batch
        print(f"  Fetching batch from database...")
        df_batch = fetch_batch_without_embeddings(BATCH_SIZE)

        if len(df_batch) == 0:
            print("  No more records to process.")
            break

        print(f"  Retrieved {len(df_batch)} records")

        # Generate embeddings
        df_batch = generate_embeddings_for_batch(df_batch, model)

        # Update database
        update_embeddings_in_database(df_batch)

        processed_count += len(df_batch)

        # Progress update
        progress_pct = (processed_count / total_records) * 100
        elapsed_time = time.time() - start_time
        avg_time_per_record = elapsed_time / processed_count
        estimated_remaining = avg_time_per_record * (total_records - processed_count)

        print(f"\n  Progress: {processed_count}/{total_records} ({progress_pct:.1f}%)")
        print(f"  Elapsed time: {elapsed_time:.1f}s")
        print(f"  Estimated remaining: {estimated_remaining:.1f}s")
        print("-"*80)

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Total records processed: {processed_count}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Average time per record: {total_time/processed_count:.3f}s")
    print("="*80)


if __name__ == "__main__":
    main()
