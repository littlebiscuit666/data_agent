import os
import sys
import asyncio

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db_manager import init_db, get_schema_info, execute_query
from agent_graph import build_workflow, create_initial_state, run_ml_analysis


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def run_sql_safety_tests():
    print("\n[Test 2] Testing read-only SQL validator and executor...")

    valid_queries = [
        "SELECT * FROM customers LIMIT 1",
        "SELECT COUNT(*) AS column_count FROM pragma_table_info('customers')",
        "WITH top_customers AS (SELECT customer_id, total_amount FROM orders LIMIT 3) SELECT * FROM top_customers",
    ]
    for query in valid_queries:
        result = execute_query(query)
        assert_true(result["success"], f"Expected valid query to succeed: {query}. Error: {result['error']}")

    invalid_queries = [
        "",
        "UPDATE customers SET name='x'",
        "DELETE FROM customers",
        "DROP TABLE customers",
        "SELECT * FROM customers; DROP TABLE customers",
        "PRAGMA table_info(customers)",
    ]
    for query in invalid_queries:
        result = execute_query(query)
        assert_true(not result["success"], f"Expected invalid query to fail: {query}")
        assert_true(bool(result["error"]), f"Expected invalid query to include an error: {query}")

    print("✅ SQL safety tests passed.")


def run_ml_unit_tests(schema_text):
    print("\n[Test 3] Testing deterministic ML segmentation parameters...")

    state_k5 = create_initial_state("Segment customers into 5 cohorts", schema_text)
    state_k5.update({"intent": "ml_segmentation", "ml_params": {"num_clusters": 5}})
    result_k5 = run_ml_analysis(state_k5)
    assert_true(not result_k5.get("error"), f"Expected K=5 segmentation to succeed: {result_k5.get('error')}")
    ml_result_k5 = result_k5.get("ml_result") or {}
    assert_true(len(ml_result_k5.get("centroids") or []) == 5, "Expected exactly 5 centroids.")
    assert_true("K=5" in ml_result_k5.get("model_type", ""), "Expected model_type to include K=5.")
    assert_true("5 behavioral segments" in ml_result_k5.get("summary", ""), "Expected summary to mention 5 segments.")

    state_default = create_initial_state("Segment customers with invalid k", schema_text)
    state_default.update({"intent": "ml_segmentation", "ml_params": {"num_clusters": "abc"}})
    result_default = run_ml_analysis(state_default)
    ml_result_default = result_default.get("ml_result") or {}
    assert_true(len(ml_result_default.get("centroids") or []) == 3, "Invalid K should default to 3 centroids.")

    state_clamped = create_initial_state("Segment customers with too many cohorts", schema_text)
    state_clamped.update({"intent": "ml_segmentation", "ml_params": {"num_clusters": "999"}})
    result_clamped = run_ml_analysis(state_clamped)
    ml_result_clamped = result_clamped.get("ml_result") or {}
    assert_true(len(ml_result_clamped.get("centroids") or []) == 6, "Out-of-range K should clamp to 6 centroids.")

    print("✅ ML segmentation unit tests passed.")


async def run_live_llm_tests(schema_text):
    # These tests require a working DeepSeek API key and network access.
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("\n[Skipped] Live LLM integration tests require DEEPSEEK_API_KEY.")
        return

    # Build LangGraph
    print("\n[Test 4] Compiling LangGraph workflow graph...")
    graph = build_workflow()
    print("✅ LangGraph compiled successfully.")

    # Test Standard NL-to-SQL Query
    print("\n[Test 5] Testing NL-to-SQL Query: 'Show me top 3 customers from China by total order amount'...")
    state = create_initial_state("Show me top 3 customers from China by total order amount", schema_text)

    res = await graph.ainvoke(state)
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

    # Test ML Forecasting (Regression)
    print("\n[Test 6] Testing ML Forecast Query: 'Forecast unique visitors for the next 30 days'...")
    ml_state_forecast = create_initial_state("Forecast unique visitors for the next 30 days", schema_text)

    res_fore = await graph.ainvoke(ml_state_forecast)
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

    # Test ML Clustering (Segmentation)
    print("\n[Test 7] Testing ML Customer Segmentation: 'Group our customers using machine learning clustering'...")
    ml_state_seg = create_initial_state("Group our customers using machine learning clustering", schema_text)

    res_seg = await graph.ainvoke(ml_state_seg)
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

    schema_text = get_schema_info()
    print(f"✅ Extracted schema length: {len(schema_text)} characters.")

    run_sql_safety_tests()
    run_ml_unit_tests(schema_text)
    asyncio.run(run_live_llm_tests(schema_text))

    print("\n==================================================")
    print("🏁 ALL TESTS COMPLETED")
    print("==================================================")


if __name__ == "__main__":
    run_tests()
