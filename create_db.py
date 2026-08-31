import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

def create_database():
    conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    cur = conn.cursor()
    cur.execute("CREATE DATABASE nexus;")
    cur.close()
    conn.close()
    
    print("Database 'nexus' created successfully.")

if __name__ == "__main__":
    create_database()
    