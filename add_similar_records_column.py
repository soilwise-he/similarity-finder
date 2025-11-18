"""
Add Similar Records Column to Database
========================================
This script adds a JSONB column to store similar records for each record.

Usage:
    python add_similar_records_column.py
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


def add_similar_records_column():
    """Add similar_records column to the records table."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            print("\nAdding similar_records column to records table...")

            # Check if column exists first
            print("  - Checking if column exists...")
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'records'
                AND column_name = 'similar_records';
            """)
            column_exists = cur.fetchone() is not None

            if column_exists:
                print("  - Column already exists, skipping creation")
            else:
                # Add similar_records column as JSONB
                print("  - Adding similar_records column...")
                cur.execute("""
                    ALTER TABLE records
                    ADD COLUMN similar_records jsonb;
                """)
                print("✓ Successfully added similar_records column")

            # Add/update comment
            print("  - Adding column comment...")
            cur.execute("""
                COMMENT ON COLUMN records.similar_records
                IS 'Array of similar records with their similarity scores (identifier, title_similarity, abstract_similarity, keywords_similarity, avg_similarity)';
            """)

            conn.commit()

            # Verify the changes
            print("\nVerifying column...")
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'records'
                AND column_name = 'similar_records';
            """)

            columns = cur.fetchall()
            if columns:
                print("✓ Found similar_records column:")
                for col_name, col_type in columns:
                    print(f"    - {col_name}: {col_type}")
            else:
                print("⚠ Warning: Column not found after creation")

            # Get total record count
            cur.execute("SELECT COUNT(*) FROM records;")
            total_records = cur.fetchone()[0]
            print(f"\n✓ Total records in table: {total_records}")

    except Exception as e:
        conn.rollback()
        print(f"✗ Error creating column: {e}")
        raise
    finally:
        conn.close()


def main():
    print("="*80)
    print("Add Similar Records Column Script")
    print("="*80)

    add_similar_records_column()

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nYou can now run: python populate_similar_records.py")
    print("="*80)


if __name__ == "__main__":
    main()
