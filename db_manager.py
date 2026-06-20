import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "cloud_mock.db"))

def get_connection():
    """Returns a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)

def init_db(force=False):
    """Initializes the database and populates it with rich synthetic data if it doesn't exist or if force=True."""
    if os.path.exists(DB_PATH) and not force:
        return True

    if force and os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except Exception as e:
            print(f"Could not remove existing DB file: {e}")
            
    print(f"Initializing database at {DB_PATH}...")
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Create Tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        country TEXT,
        segment TEXT,
        created_at DATE
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT,
        price REAL,
        stock INTEGER
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER,
        order_date DATE,
        total_amount REAL,
        status TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS order_items (
        item_id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        unit_price REAL,
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS web_traffic (
        traffic_id INTEGER PRIMARY KEY AUTOINCREMENT,
        traffic_date DATE UNIQUE,
        page_views INTEGER,
        unique_visitors INTEGER,
        bounce_rate REAL,
        conversions INTEGER
    )
    """)
    
    # 2. Insert Synthetic Data
    np.random.seed(42)
    
    # Customers
    countries = ["China", "United States", "Japan", "Germany", "United Kingdom", "Canada"]
    segments = ["Enterprise", "SMB", "Individual"]
    names = [
        "张伟", "王芳", "李娜", "刘洋", "陈杰", "John Smith", "Emily Davis", "Kenji Sato", 
        "Hans Mueller", "Sarah Connor", "Michael Brown", "David Wilson", "Emma Thomas",
        "田中太郎", "林小明", "赵敏", "周杰伦", "Alice Johnson", "Robert Miller", "Sofia Garcia"
    ]
    
    customer_data = []
    start_date = datetime(2024, 1, 1)
    for i, name in enumerate(names):
        email = f"{name.lower().replace(' ', '')}@example.com"
        country = countries[i % len(countries)]
        segment = segments[i % len(segments)]
        created = (start_date + timedelta(days=int(np.random.randint(0, 365)))).strftime("%Y-%m-%d")
        customer_data.append((name, email, country, segment, created))
        
    cursor.executemany(
        "INSERT INTO customers (name, email, country, segment, created_at) VALUES (?, ?, ?, ?, ?)",
        customer_data
    )
    
    # Products
    products = [
        ("Enterprise Analytics Suite", "Software", 1200.00, 100),
        ("Developer License Single", "Software", 350.00, 500),
        ("AI Code Assistant Pro", "Software", 49.00, 1000),
        ("Cloud Storage 1TB Plan", "Cloud Services", 12.00, 2000),
        ("Virtual Private Server Base", "Cloud Services", 25.00, 800),
        ("Technical Support Hourly", "Services", 150.00, 9999),
        ("Agile Team Consultation Pack", "Services", 2500.00, 50),
        ("Data Pipeline Orchestrator", "Software", 600.00, 150)
    ]
    cursor.executemany(
        "INSERT INTO products (name, category, price, stock) VALUES (?, ?, ?, ?)",
        products
    )
    
    # Orders and Order Items
    # We will generate orders spanning from 2024-01-01 to 2025-05-31
    order_id_counter = 1
    orders_batch = []
    order_items_batch = []
    
    current_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 5, 31)
    
    while current_date <= end_date:
        # Determine number of orders for this day (0 to 3)
        num_orders = np.random.choice([0, 1, 2, 3], p=[0.2, 0.5, 0.2, 0.1])
        for _ in range(num_orders):
            cust_id = int(np.random.randint(1, len(names) + 1))
            status = np.random.choice(["Completed", "Completed", "Completed", "Pending", "Cancelled"], p=[0.8, 0.08, 0.08, 0.02, 0.02])
            
            # Select random products
            num_items = np.random.randint(1, 4)
            chosen_product_indices = np.random.choice(len(products), num_items, replace=False)
            
            total_amount = 0
            for idx in chosen_product_indices:
                prod_id = idx + 1
                prod_price = products[idx][2]
                quantity = int(np.random.choice([1, 2, 3, 5], p=[0.7, 0.15, 0.1, 0.05]))
                item_price = prod_price
                
                # Introduce occasional discount
                if np.random.rand() > 0.8:
                    item_price = round(prod_price * 0.9, 2)
                    
                total_amount += item_price * quantity
                order_items_batch.append((order_id_counter, prod_id, quantity, item_price))
                
            orders_batch.append((cust_id, current_date.strftime("%Y-%m-%d"), round(total_amount, 2), status))
            order_id_counter += 1
            
        current_date += timedelta(days=1)
        
    cursor.executemany(
        "INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES (?, ?, ?, ?)",
        orders_batch
    )
    cursor.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)",
        order_items_batch
    )
    
    # Web Traffic Data (180 days ending today)
    traffic_start_date = datetime.now() - timedelta(days=180)
    traffic_batch = []
    base_traffic = 500
    
    for day in range(181):
        t_date = traffic_start_date + timedelta(days=day)
        # Seasonal weekend dip
        day_of_week = t_date.weekday()
        weekend_factor = 0.6 if day_of_week >= 5 else 1.0
        
        # General upward trend + noise
        trend = day * 0.8
        visitors = int((base_traffic + trend + np.random.normal(0, 30)) * weekend_factor)
        visitors = max(50, visitors)
        page_views = int(visitors * np.random.uniform(2.2, 3.5))
        bounce_rate = round(float(np.random.uniform(0.35, 0.65)), 4)
        
        # Conversions scale with visitors, with a base rate of 2% + some noise
        conv_rate = np.random.uniform(0.015, 0.035)
        conversions = int(visitors * conv_rate)
        
        traffic_batch.append((t_date.strftime("%Y-%m-%d"), page_views, visitors, bounce_rate, conversions))
        
    cursor.executemany(
        "INSERT INTO web_traffic (traffic_date, page_views, unique_visitors, bounce_rate, conversions) VALUES (?, ?, ?, ?, ?)",
        traffic_batch
    )
    
    conn.commit()
    conn.close()
    print("Database initialization completed successfully.")
    return True

def get_schema_info():
    """Extracts schema definition of all tables in the database to guide LLM SQL generation."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get table list
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]
    
    schema_str = ""
    for table in tables:
        schema_str += f"Table: {table}\n"
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        # columns format: (cid, name, type, notnull, dflt_value, pk)
        cols_desc = []
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            is_pk = " PRIMARY KEY" if col[5] else ""
            cols_desc.append(f"  - {col_name} ({col_type}){is_pk}")
        schema_str += "\n".join(cols_desc) + "\n"
        
        # Show sample data (2 rows) for context
        cursor.execute(f"SELECT * FROM {table} LIMIT 2;")
        rows = cursor.fetchall()
        if rows:
            schema_str += "  Sample Rows:\n"
            for row in rows:
                schema_str += f"    {list(row)}\n"
        schema_str += "\n"
        
    conn.close()
    return schema_str

def execute_query(sql_query: str):
    """Executes a query safely. Blocks modifying statements (write actions)."""
    # Simple block for write commands
    forbidden = ["insert ", "update ", "delete ", "drop ", "create ", "alter ", "replace ", "truncate "]
    query_lower = sql_query.lower().strip()
    
    for word in forbidden:
        if query_lower.startswith(word) or f" {word}" in query_lower:
            raise PermissionError(f"Query contains forbidden write statement: '{word}'")
            
    conn = get_connection()
    try:
        df = pd.read_sql_query(sql_query, conn)
        # Convert nan/inf to None to prevent JSON serialization errors
        df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
        data = df.to_dict(orient="records")
        columns = list(df.columns)
        return {
            "success": True,
            "data": data,
            "columns": columns,
            "row_count": len(data),
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "data": [],
            "columns": [],
            "row_count": 0,
            "error": str(e)
        }
    finally:
        conn.close()

if __name__ == "__main__":
    init_db(force=True)
    print("Schema details:")
    print(get_schema_info())
