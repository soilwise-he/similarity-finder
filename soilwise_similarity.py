import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import json
import time
from datetime import timedelta
import gc

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
num_simil = 100
skip_reverse_updates = False  # Set to False to enable reverse relationship updates
batch_size = 1000  # Number of records to load per batch for extraction (safe limit based on diagnostics)
similarity_threshold = 0.5
use_batched_processing = True  # Set to True to process all records in batches
combine_before_faiss = True  # Set to True to combine all batches before FAISS (more accurate, more memory)


def get_db_connection():
    """Create and return a database connection with extended timeout settings."""
    try:
        # Add timeout settings to prevent premature disconnections
        config_with_timeout = DB_CONFIG.copy()
        config_with_timeout.update({
            'connect_timeout': 300,      # 5 minutes to establish connection
            'keepalives': 1,              # Enable TCP keepalives
            'keepalives_idle': 30,        # Start keepalives after 30 seconds idle
            'keepalives_interval': 10,    # Send keepalive every 10 seconds
            'keepalives_count': 5         # Max 5 failed keepalives before disconnect
        })

        conn = psycopg2.connect(**config_with_timeout)

        # Set statement timeout to 5 minutes (in milliseconds)
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = 300000;")  # 5 minutes
        conn.commit()

        print("Successfully connected to PostgreSQL database (with extended timeouts)")
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


def get_total_record_count():
    """Get the total number of records in the database."""
    query = "SELECT COUNT(*) FROM records;"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            count = cur.fetchone()[0]
        return count
    finally:
        conn.close()


def retrieve_records_in_database(limit=None, offset=0):
    """
    Fetch records with pre-computed embeddings from database.
    Records with NULL embeddings are included (where original fields were empty).

    Args:
        limit: Maximum number of records to retrieve (default: batch_size)
        offset: Number of records to skip (for pagination)
    """
    if limit is None:
        limit = batch_size

    query = f"""
            SELECT
                identifier,
                title,
                abstract,
                keywords,
                embedding_title,
                embedding_abstract,
                embedding_keywords
            FROM
                records
            LIMIT {limit}
            OFFSET {offset};
    """
    df = fetch_data(query)

    print(f"Retrieved {len(df)} records from database")

    # Force garbage collection after query to free memory immediately
    gc.collect()

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


def aggregate_top_similarities(all_results, top_k):
    """
    Aggregate similarity results across multiple batches and keep only top-k.

    Args:
        all_results: List of similarity DataFrames from different batches
        top_k: Number of top similar records to keep

    Returns:
        DataFrame with top-k most similar records across all batches
    """
    if not all_results:
        return pd.DataFrame()

    # Concatenate all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Remove duplicates (same db_record_id), keeping the one with highest avg_similarity
    combined_df = combined_df.sort_values('avg_similarity', ascending=False)
    combined_df = combined_df.drop_duplicates(subset=['db_record_id'], keep='first')

    # Return top-k
    return combined_df.head(top_k)


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
        new_record_id = df_new.iloc[i]['identifier']

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
            # Skip self-match
            db_record_id = df_db.iloc[db_idx]['identifier']
            if db_record_id == new_record_id:
                continue

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


def check_database_health(max_attempts=4, initial_delay=2):
    """
    Check if database is responsive before attempting updates.
    Uses exponential backoff to wait for database recovery.
    """
    for attempt in range(max_attempts):
        delay = initial_delay * (2 ** attempt)  # Exponential backoff: 2, 4, 8, 16, 32...
        if delay > 60:
            delay = 60  # Cap at 60 seconds

        try:
            print(f"  - Database health check (attempt {attempt + 1}/{max_attempts})...")
            conn = get_db_connection()
            with conn.cursor() as cur:
                # Simple query to verify database is responsive
                cur.execute("SELECT 1;")
                cur.fetchone()
            conn.close()
            print(f"  ✓ Database is responsive")
            return True
        except Exception as e:
            print(f"  ✗ Database not responsive: {e}")
            if attempt < max_attempts - 1:
                print(f"  - Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                print(f"  ✗ Database did not recover after {max_attempts} attempts")
                return False
    return False


def update_similar_records_in_database(identifier, similarity_df, threshold=similarity_threshold, skip_reverse_updates=False):
    """
    Update the similar_records field in the database for a given record.
    Only stores records with avg_similarity > threshold.
    Excludes the record itself from the similar records list.
    Also updates the reverse relationship - adds the new record to each similar record's list.
    Each database update is done in a separate transaction to avoid connection timeouts.

    Args:
        identifier: The record identifier to update
        similarity_df: DataFrame containing similarity results
        threshold: Minimum similarity threshold (default 0.5)
        skip_reverse_updates: If True, only updates the main record and skips reverse relationships (default False)
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
        # Convert NaN/None to None (which becomes null in JSON)
        def clean_value(val):
            if pd.isna(val):
                return None
            return float(val) if isinstance(val, (int, float, np.number)) else val

        similar_records.append({
            'identifier': row['db_record_id'],
            'title_similarity': clean_value(row['title_similarity']),
            'abstract_similarity': clean_value(row['abstract_similarity']),
            'keywords_similarity': clean_value(row['keywords_similarity']),
            'avg_similarity': clean_value(row['avg_similarity'])
        })

    # Convert to JSON (NaN values are now converted to null)
    similar_records_json = json.dumps(similar_records)

    # Wait and check database health before attempting updates
    print("\n" + "="*80)
    print("Checking database health before updates...")
    print("="*80)

    if not check_database_health(max_attempts=3, initial_delay=2):
        raise Exception("Database is not responsive. Cannot proceed with updates.")

    print("\n" + "="*80)
    print("Proceeding with database updates...")
    print("="*80 + "\n")

    # Update the main record's similar_records field
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE records
                    SET similar_records = %s::jsonb
                    WHERE identifier = %s;
                """, (similar_records_json, identifier))
                conn.commit()
                print(f"✓ Updated similar_records field for {identifier}")
                print(f"  - Stored {len(similar_records)} similar records (avg_similarity > {threshold})")
                break  # Success, exit retry loop

        except Exception as e:
            if conn:
                conn.rollback()
            retry_count += 1
            if retry_count < max_retries:
                print(f"⚠ Database error (attempt {retry_count}/{max_retries}): {e}")
                print(f"  Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"✗ Error updating main record after {max_retries} attempts: {e}")
                raise
        finally:
            if conn:
                conn.close()

    # Skip reverse updates if requested
    if skip_reverse_updates:
        print(f"  - Skipping reverse relationship updates (skip_reverse_updates=True)")
        print(f"  - Note: {len(filtered_df)} records were NOT updated with reverse relationships")
        return

    # Add a small delay to allow database to recover
    print(f"  - Waiting 3 seconds before updating reverse relationships...")
    time.sleep(3)

    # Update reverse relationships - use single connection for all updates
    print(f"  - Updating {len(filtered_df)} reverse relationships (using single connection)...")
    successful_updates = 0
    failed_updates = 0

    # Retry logic for establishing connection
    retry_count = 0
    while retry_count < max_retries:
        conn = None
        try:
            print(f"  - Attempting to connect to database (attempt {retry_count + 1}/{max_retries})...")
            conn = get_db_connection()
            print(f"  - Connection established successfully")

            print(f"  - Setting autocommit mode...")
            conn.autocommit = True  # Enable autocommit mode to avoid long transactions
            print(f"  - Autocommit enabled")

            print(f"  - Starting to process {len(filtered_df)} updates...")

            # Process all reverse relationships using the same connection
            for idx, row in filtered_df.iterrows():
                db_record_id = row['db_record_id']

                try:
                    with conn.cursor() as cur:
                        # Convert NaN/None to None (which becomes null in JSON)
                        def clean_value(val):
                            if pd.isna(val):
                                return None
                            return float(val) if isinstance(val, (int, float, np.number)) else val

                        # Create the reverse similarity entry
                        reverse_entry = {
                            'identifier': identifier,
                            'title_similarity': clean_value(row['title_similarity']),
                            'abstract_similarity': clean_value(row['abstract_similarity']),
                            'keywords_similarity': clean_value(row['keywords_similarity']),
                            'avg_similarity': clean_value(row['avg_similarity'])
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

                        successful_updates += 1

                        # Print progress every 10 records
                        if (idx + 1) % 10 == 0:
                            print(f"    - Progress: {idx + 1}/{len(filtered_df)} records updated")

                except Exception as e:
                    # Log individual record failure but continue processing
                    print(f"    ✗ Failed to update {db_record_id}: {e}")
                    failed_updates += 1

            # All updates completed successfully
            print(f"✓ Updated reverse relationships: {successful_updates} successful, {failed_updates} failed")
            break  # Exit retry loop

        except Exception as e:
            # Connection-level error - retry entire batch
            retry_count += 1
            if retry_count < max_retries:
                print(f"⚠ Database connection error (attempt {retry_count}/{max_retries}): {e}")
                print(f"  Retrying in 2 seconds...")
                time.sleep(2)
                # Reset counters for retry
                successful_updates = 0
                failed_updates = 0
            else:
                print(f"✗ Error connecting to database after {max_retries} attempts: {e}")
                print(f"  Managed to update {successful_updates} records before connection failed")
                raise
        finally:
            if conn:
                conn.close()


def main():
    # Start timing
    start_time = time.time()
    print("\n" + "="*80)
    print("STARTING SIMILARITY ANALYSIS")
    print("="*80 + "\n")

    # Fetch a random record from database (in production, this would be a newly harvested record)
    df_new = retrieve_newest_record()
    new_record_id = df_new.iloc[0]['identifier']
    print(f"Retrieved new record: {new_record_id}")
    print("\n" + "="*80 + "\n")
    print(df_new[['identifier', 'title', 'abstract', 'keywords']])
    print("\n" + "="*80 + "\n")

    if use_batched_processing:
        # BATCHED PROCESSING MODE - Process entire database in chunks
        print("="*80)
        print("BATCHED PROCESSING MODE ENABLED")
        print("="*80)
        print(f"Extracting database in batches of {batch_size} records")

        if combine_before_faiss:
            print(f"STRATEGY: Combine all batches → Single FAISS calculation (more accurate)")
        else:
            print(f"STRATEGY: FAISS per batch → Aggregate results (more memory-safe)")
        print()

        # Get total record count
        total_records = get_total_record_count()
        print(f"Total records in database: {total_records}")
        num_batches = (total_records + batch_size - 1) // batch_size  # Ceiling division
        print(f"Number of batches to extract: {num_batches}\n")

        if combine_before_faiss:
            # COMBINED APPROACH: Extract all batches, combine, then single FAISS
            print("="*80)
            print("PHASE 1: EXTRACTING ALL BATCHES")
            print("="*80)

            all_batches = []
            for batch_num in range(num_batches):
                offset = batch_num * batch_size
                print(f"\nExtracting batch {batch_num + 1}/{num_batches} (records {offset + 1} to {min(offset + batch_size, total_records)})")

                # Fetch batch of records
                df_db_batch = retrieve_records_in_database(limit=batch_size, offset=offset)

                if len(df_db_batch) == 0:
                    print("No more records to process. Stopping.")
                    break

                print(f"  ✓ Extracted {len(df_db_batch)} records")

                # Store the batch
                all_batches.append(df_db_batch)

                # Wait between batches to let database recover
                if batch_num < num_batches - 1:
                    print(f"  ⏱ Waiting 5 seconds before next extraction...")
                    time.sleep(5)

            # Combine all batches into single dataset
            print("\n" + "="*80)
            print("PHASE 2: COMBINING ALL BATCHES")
            print("="*80)
            print(f"Combining {len(all_batches)} batches into single dataset...")

            df_db = pd.concat(all_batches, ignore_index=True)
            print(f"✓ Combined dataset shape: {df_db.shape}")
            print(f"✓ Total records loaded: {len(df_db)}")

            # Free up batch memory
            del all_batches            
            gc.collect()
            print(f"✓ Memory cleanup after combining batches")

            # Calculate similarity on complete dataset
            print("\n" + "="*80)
            print("PHASE 3: CALCULATING SIMILARITIES (SINGLE FAISS)")
            print("="*80)
            print(f"Running FAISS on complete dataset of {len(df_db)} records...")

            similarity_df = calculate_similarity_new_vs_db(df_new, df_db, k=num_simil)
            print(f"✓ Found top {len(similarity_df)} most similar records across ALL {total_records} database records")

            # Free up combined dataset memory
            del df_db
            gc.collect()
            print(f"✓ Memory cleanup after FAISS calculation")

        else:
            # ORIGINAL APPROACH: Calculate FAISS per batch, then aggregate
            print("="*80)
            print("PROCESSING BATCHES WITH FAISS PER BATCH")
            print("="*80)

            all_results = []
            for batch_num in range(num_batches):
                offset = batch_num * batch_size
                print("="*80)
                print(f"PROCESSING BATCH {batch_num + 1}/{num_batches}")
                print(f"Records {offset + 1} to {min(offset + batch_size, total_records)}")
                print("="*80)

                # Fetch batch of records
                df_db_batch = retrieve_records_in_database(limit=batch_size, offset=offset)

                if len(df_db_batch) == 0:
                    print("No more records to process. Stopping.")
                    break

                print(f"Batch shape: {df_db_batch.shape}")

                # Calculate similarity for this batch
                batch_results = calculate_similarity_new_vs_db(df_new, df_db_batch, k=num_simil)
                print(f"Found {len(batch_results)} similar records in this batch")

                # Store results
                all_results.append(batch_results)

                # Free up memory after each batch
                del df_db_batch
                gc.collect()

                # Wait between batches to let database recover
                if batch_num < num_batches - 1:  # Don't wait after last batch
                    print(f"\n⏱ Waiting 5 seconds before next batch...")
                    time.sleep(5)

            # Aggregate results across all batches
            print("\n" + "="*80)
            print("AGGREGATING RESULTS ACROSS ALL BATCHES")
            print("="*80)
            print(f"Total batches processed: {len(all_results)}")

            similarity_df = aggregate_top_similarities(all_results, top_k=num_simil)
            print(f"Aggregated top {len(similarity_df)} most similar records across all {total_records} database records")

            # Clean up batch results
            del all_results
            gc.collect()

    else:
        # SINGLE-BATCH MODE - Load all records at once (original behavior)
        print("="*80)
        print("SINGLE-BATCH MODE (LEGACY)")
        print("="*80)
        print(f"Loading up to {batch_size} records from database\n")

        # Fetch records with pre-computed embeddings from database
        df_db = retrieve_records_in_database(limit=batch_size)
        print("\n" + "="*80 + "\n")
        print(f"DataFrame shape: {df_db.shape}")
        print(f"Columns: {list(df_db.columns)}")
        print("="*80 + "\n")

        # Calculate similarity between new record and database records
        similarity_df = calculate_similarity_new_vs_db(df_new, df_db, k=num_simil)

        # Free up memory
        del df_db       
        gc.collect()

    # Test database connection after all processing
    print("\n" + "="*80)
    print("Testing database connection after similarity calculations...")
    print("="*80)
    try:
        test_conn = get_db_connection()
        test_conn.close()
        print("✓ Database is still responsive after calculations\n")
    except Exception as e:
        print(f"✗ Database became unresponsive during calculations: {e}\n")

    print("\n" + "="*80)
    print(f"Found {len(similarity_df)} similar records")
    print("="*80 + "\n")

    # Display top most similar records
    print("Top 10 most similar records:")
    print(similarity_df[['new_record_id', 'db_record_id', 'new_record_title', 'db_record_title',
                         'title_similarity', 'abstract_similarity', 'keywords_similarity', 'avg_similarity']].head(10))

    print("\n" + "="*80 + "\n")

    # Save results to CSV
    output_file = 'similarity_results.csv'
    similarity_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Free up memory before database operations
    print("\nFreeing up memory before database updates...")
    del df_new  # Delete new record dataframe
    gc.collect()  # Force garbage collection
    print("Memory cleanup completed")

    # Give database extra time to fully recover
    print("\n⏱ Waiting 10 seconds for database to stabilize...")
    time.sleep(10)

    # Update similar_records field in database
    print("\n" + "="*80)
    print("Updating database with similar records...")
    print("="*80 + "\n")

    try:
        update_similar_records_in_database(new_record_id, similarity_df, threshold=similarity_threshold, skip_reverse_updates=skip_reverse_updates)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: Could not update database")
        print(f"Error: {e}")
        print(f"\n⚠ WORKAROUND: The similarity results have been saved to '{output_file}'")
        print(f"⚠ You can manually update the database later when it's stable.")
        print(f"\nRecord to update: {new_record_id}")
        print(f"Number of similar records found: {len(similarity_df[similarity_df['avg_similarity'] > similarity_threshold])}")

    # Calculate and display total execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_timedelta = timedelta(seconds=elapsed_time)

    print("\n" + "="*80)
    print("SIMILARITY ANALYSIS COMPLETED")
    print("="*80)
    print(f"Total execution time: {elapsed_timedelta}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print("="*80 + "\n")



if __name__ == "__main__":
    main()


