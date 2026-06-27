import os
import shutil
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/ai_translator")
DATA_DIR = os.getenv("DATA_DIR", "/data")

def clean_all():
    print("Connecting to database...")
    engine = create_engine(DATABASE_URL)
    
    # 1. Truncate tables
    with engine.connect() as conn:
        print("Truncating tables...")
        conn.execute(text("TRUNCATE datasets, dataset_versions, state_transitions, jobs, experiment_runs, model_registry CASCADE;"))
        conn.commit()
    print("Database truncated successfully.")

    # 2. Clean directories
    directories = ["raw", "merged", "batched", "processed"]
    for folder in directories:
        path = os.path.join(DATA_DIR, folder)
        if os.path.exists(path):
            print(f"Cleaning folder: {path}")
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception as e:
                    print(f"Failed to delete {item_path}: {e}")
                    
    print("All file directories cleaned successfully!")

if __name__ == "__main__":
    clean_all()
