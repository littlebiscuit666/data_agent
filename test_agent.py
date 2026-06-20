import json
import os
import sys

# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db_manager import init_db, get_schema_info
from agent_graph import build_workflow

def run_tests():
    print("==================================================")
    print("🧪 DATA AGENT INTEGRATION & UNIT TEST SUITE")
    print("==================================================")
    
    # 1. Initialize DB
    print("\n[Test 1] Initializing mock cloud database...")
    db_ok = init_db(force=True)
    if db_ok:
        print("✅ Database initialized successfully.")
    else:
        print("❌ Database initialization failed.")
        return
        
    # Get schema
    schema_text = get_schema_info()
    print(f"✅ Extracted schema length: {len(schema_text)} characters.")
    
    # Build LangGraph
    print("\n[Test 2] Compiling LangGraph workflow graph...")
    graph = build_workflow()
    print("✅ LangGraph compiled successfully.")
    
    # 3. Test Standard NL-to-SQL Query
    print("\n[Test 3] Testing NL-to-SQL Query: 'Show me top 3 customers from China by total order amount'...")
    state = {
        "query": "Show me top 3 customers from China by total order amount",
        "intent": "sql",
        "sql_query": None,
        "data": None,
        "columns": None,
        "schema": schema_text,
        "chart_config": None,
        "ml_result": None,
        "conversational_response": None,
        "error": None,
        "retry_count": 0,
        "logs": []
    }
    
    res = graph.invoke(state)
    print(" -> Logs printed by agent:")
    for log in res["logs"]:
        print(f"    * {log}")
        
    if res.get("error"):
        print(f"❌ NL-to-SQL Test Failed. Error: {res['error']}")
    else:
        print(f"✅ NL-to-SQL Succeeded!")
        print(f"   Generated SQL: {res['sql_query']}")
        print(f"   Row Count: {len(res['data'] or [])}")
        if res["data"]:
            print(f"   Sample Row: {res['data'][0]}")
        print(f"   Chart Type Generated: {res['chart_config'].get('type') if res.get('chart_config') else 'None'}")
        
    # 4. Test ML Forecasting (Regression)
    print("\n[Test 4] Testing ML Forecast Query: 'Forecast unique visitors for the next 30 days'...")
    ml_state_forecast = {
        "query": "Forecast unique visitors for the next 30 days",
        "intent": "sql", # Router should overwrite this to ml_forecast
        "sql_query": None,
        "data": None,
        "columns": None,
        "schema": schema_text,
        "chart_config": None,
        "ml_result": None,
        "conversational_response": None,
        "error": None,
        "retry_count": 0,
        "logs": []
    }
    
    res_fore = graph.invoke(ml_state_forecast)
    print(" -> Logs printed by agent:")
    for log in res_fore["logs"]:
        print(f"    * {log}")
    if res_fore.get("error"):
        print(f"❌ ML Forecast Test Failed. Error: {res_fore['error']}")
    else:
        print(f"✅ ML Forecast Succeeded!")
        ml_res = res_fore.get("ml_result") or {}
        print(f"   Model Type: {ml_res.get('model_type')}")
        print(f"   R² Coefficient: {ml_res.get('metrics', {}).get('R_squared')}")
        print(f"   Trend Slope: {ml_res.get('metrics', {}).get('slope')}")
        print(f"   Forecast Data Points: {len(ml_res.get('forecast') or [])} days generated")
        print(f"   Chart Type Generated: {res_fore['chart_config'].get('type') if res_fore.get('chart_config') else 'None'}")
        
    # 5. Test ML Clustering (Segmentation)
    print("\n[Test 5] Testing ML Customer Segmentation: 'Group our customers using machine learning clustering'...")
    ml_state_seg = {
        "query": "Group our customers using machine learning clustering",
        "intent": "sql", # Router should overwrite to ml_segmentation
        "sql_query": None,
        "data": None,
        "columns": None,
        "schema": schema_text,
        "chart_config": None,
        "ml_result": None,
        "conversational_response": None,
        "error": None,
        "retry_count": 0,
        "logs": []
    }
    
    res_seg = graph.invoke(ml_state_seg)
    print(" -> Logs printed by agent:")
    for log in res_seg["logs"]:
        print(f"    * {log}")
    if res_seg.get("error"):
        print(f"❌ ML Customer Segmentation Test Failed. Error: {res_seg['error']}")
    else:
        print(f"✅ ML Customer Segmentation Succeeded!")
        ml_res = res_seg.get("ml_result") or {}
        print(f"   Model Type: {ml_res.get('model_type')}")
        print(f"   Number of Segments (Centroids): {len(ml_res.get('centroids') or [])}")
        for c in ml_res.get('centroids', []):
            print(f"     - Cluster {c['cluster_id']} ({c['label']}): Size={c['size']}, Avg Spent=${c['avg_spent']:.2f}")
        print(f"   Total customer labels: {len(ml_res.get('customers') or [])}")
        print(f"   Chart Type Generated: {res_seg['chart_config'].get('type') if res_seg.get('chart_config') else 'None'}")
        
    print("\n==================================================")
    print("🏁 ALL TESTS COMPLETED")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
