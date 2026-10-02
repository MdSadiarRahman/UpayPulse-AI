import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "upaypulse.db")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def get_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    # Enable accessing columns by name
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the database schema for Agent, Transaction, and Merchant models.
    Also migrates existing CSV data into the SQLite database.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Insert existing synthetic data from CSVs
    try:
        if os.path.exists(os.path.join(DATA_DIR, "agents.csv")):
            agents_df = pd.read_csv(os.path.join(DATA_DIR, "agents.csv"))
            agents_df.to_sql("agents", conn, if_exists="replace", index=False)
            
            # Add AI columns
            try:
                cursor.execute("ALTER TABLE agents ADD COLUMN expected_shortage REAL DEFAULT 0")
                cursor.execute("ALTER TABLE agents ADD COLUMN risk_level TEXT DEFAULT 'SAFE'")
            except sqlite3.OperationalError:
                pass # Columns might already exist
            
        if os.path.exists(os.path.join(DATA_DIR, "merchants.csv")):
            merchants_df = pd.read_csv(os.path.join(DATA_DIR, "merchants.csv"))
            merchants_df.to_sql("merchants", conn, if_exists="replace", index=False)
            
        if os.path.exists(os.path.join(DATA_DIR, "transactions.csv")):
            txns_df = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
            txns_df.to_sql("transactions", conn, if_exists="replace", index=False)
            
        print("Successfully migrated CSV data into SQLite database models.")
    except Exception as e:
        print(f"Warning: Could not load CSVs - {e}")
        
    conn.close()


# --- Database Functions ---

def fetch_agent_data(agent_id=None):
    """
    Fetches agent data from the database. 
    If agent_id is provided, fetches a single agent. Otherwise, fetches all.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if agent_id:
        cursor.execute("SELECT * FROM agents WHERE agent_id = ?", (agent_id,))
        result = cursor.fetchone()
        conn.close()
        return dict(result) if result else None
    else:
        cursor.execute("SELECT * FROM agents")
        results = cursor.fetchall()
        conn.close()
        return [dict(row) for row in results]

def fetch_transactions(agent_id=None, limit=50):
    """
    Fetches transactions. Can be filtered by agent_id.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if agent_id:
        cursor.execute("SELECT * FROM transactions WHERE agent_id = ? ORDER BY timestamp DESC LIMIT ?", (agent_id, limit))
    else:
        cursor.execute("SELECT * FROM transactions ORDER BY timestamp DESC LIMIT ?", (limit,))
        
    results = cursor.fetchall()
    conn.close()
    return [dict(row) for row in results]

def update_prediction_result(agent_id: str, expected_shortage: float, risk_level: str):
    """
    Updates an agent's record with the latest AI/ML prediction results.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE agents 
        SET expected_shortage = ?, risk_level = ?
        WHERE agent_id = ?
    """, (expected_shortage, risk_level, agent_id))
    
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    
    return rows_affected > 0


if __name__ == "__main__":
    # Initialize DB and run tests when executed directly
    print("Initializing Database...")
    init_db()
    
    print("\n--- Testing Fetch All Agents (First 2) ---")
    agents = fetch_agent_data()
    print(agents[:2])
    
    if agents:
        test_agent = agents[0]['agent_id']
        print(f"\n--- Testing Update Prediction for {test_agent} ---")
        update_prediction_result(test_agent, expected_shortage=25000, risk_level="HIGH")
        
        updated_agent = fetch_agent_data(test_agent)
        print(updated_agent)
        
    print("\n--- Testing Fetch Transactions ---")
    txns = fetch_transactions(limit=2)
    print(txns)
