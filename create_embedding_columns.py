"""
Create Embedding Columns in Database
=====================================
This script adds the embedding columns to the records table using PostgreSQL arrays.

Usage:
    python create_embedding_columns.py
"""

import psycopg2
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


def get_db_connection():
    """Create and return a database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✓ Successfully connected to PostgreSQL database")
        return conn
    except psycopg2.Error as e:
        print(f"✗ Error connecting to PostgreSQL database: {e}")
        raise


def create_embedding_columns():
    """Add embedding columns to the records table."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            print("\nAdding embedding columns to records table...")

            # Add embedding_title column
            print("  - Adding embedding_title column...")
            cur.execute("""
                ALTER TABLE records
                ADD COLUMN IF NOT EXISTS embedding_title real[];
            """)

            # Add embedding_abstract column
            print("  - Adding embedding_abstract column...")
            cur.execute("""
                ALTER TABLE records
                ADD COLUMN IF NOT EXISTS embedding_abstract real[];
            """)

            # Add embedding_keywords column
            print("  - Adding embedding_keywords column...")
            cur.execute("""
                ALTER TABLE records
                ADD COLUMN IF NOT EXISTS embedding_keywords real[];
            """)

            # Add comments
            print("  - Adding column comments...")
            cur.execute("""
                COMMENT ON COLUMN records.embedding_title
                IS 'Vector embedding of the title field (384-dim array from all-MiniLM-L6-v2)';
            """)
            cur.execute("""
                COMMENT ON COLUMN records.embedding_abstract
                IS 'Vector embedding of the abstract field (384-dim array from all-MiniLM-L6-v2)';
            """)
            cur.execute("""
                COMMENT ON COLUMN records.embedding_keywords
                IS 'Vector embedding of the keywords field (384-dim array from all-MiniLM-L6-v2)';
            """)

            conn.commit()
            print("✓ Successfully added all embedding columns")

            # Verify the changes
            print("\nVerifying columns...")
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'records'
                AND column_name LIKE 'embedding%'
                ORDER BY column_name;
            """)

            columns = cur.fetchall()
            if columns:
                print("✓ Found embedding columns:")
                for col_name, col_type in columns:
                    print(f"    - {col_name}: {col_type}")
            else:
                print("⚠ Warning: No embedding columns found after creation")

            # Get total record count
            cur.execute("SELECT COUNT(*) FROM records;")
            total_records = cur.fetchone()[0]
            print(f"\n✓ Total records in table: {total_records}")

    except psycopg2.errors.DuplicateColumn as e:
        print("⚠ Columns may already exist. Checking...")
        conn.rollback()
    except Exception as e:
        conn.rollback()
        print(f"✗ Error creating columns: {e}")
        raise
    finally:
        conn.close()


def main():
    print("="*80)
    print("Create Embedding Columns Script")
    print("="*80)

    create_embedding_columns()

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nYou can now run: python populate_embeddings.py")
    print("="*80)


if __name__ == "__main__":
    main()
