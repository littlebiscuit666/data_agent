import json
import re
import os
import asyncio
import pandas as pd
import numpy as np
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime, timedelta

from config import get_llm
from db_manager import execute_query, get_schema_info, get_connection

# Define Graph State
class AgentState(TypedDict):
    query: str
    intent: str                  # "sql", "ml_forecast", "ml_segmentation", "conversational"
    ml_params: Optional[Dict[str, Any]]
    sql_query: Optional[str]
    data: Optional[List[Dict[str, Any]]]
    columns: Optional[List[str]]
    schema: str
    chart_config: Optional[Dict[str, Any]]
    ml_result: Optional[Dict[str, Any]]
    conversational_response: Optional[str]
    error: Optional[str]
    retry_count: int
    logs: List[str]

VALID_INTENTS = {"sql", "ml_forecast", "ml_segmentation", "conversational"}


def create_initial_state(query: str, schema: Optional[str] = None) -> AgentState:
    """Creates a complete initial state for every LangGraph invocation."""
    return {
        "query": query,
        "intent": "sql",
        "ml_params": {},
        "sql_query": None,
        "data": None,
        "columns": None,
        "schema": schema if schema is not None else get_schema_info(),
        "chart_config": None,
        "ml_result": None,
        "conversational_response": None,
        "error": None,
        "retry_count": 0,
        "logs": []
    }


def parse_int_param(params: Dict[str, Any], key: str, default: int, min_value: int, max_value: int, logs: Optional[List[str]] = None) -> int:
    """Parses an integer ML parameter with safe defaults and clamping."""
    raw_value = params.get(key) if isinstance(params, dict) else None
    value = default

    try:
        if isinstance(raw_value, bool) or raw_value is None:
            raise ValueError
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
            if not re.fullmatch(r"[-+]?\d+", raw_value):
                raise ValueError
        value = int(raw_value)
    except (TypeError, ValueError):
        if logs is not None and raw_value not in (None, ""):
            logs.append(f"Invalid parameter {key}={raw_value!r}; using default {default}.")
        value = default

    clamped = max(min_value, min(max_value, value))
    if logs is not None and clamped != value:
        logs.append(f"Parameter {key}={value} was clamped to {clamped}.")
    return clamped

# Clean json parser helper
def parse_json_from_llm(content: str) -> Dict[str, Any]:
    """Cleans markdown syntax or outer text from LLM response and parses it to JSON."""
    content_clean = content.strip()
    
    # Check if there is a json code block
    match = re.search(r"```json\s*(.*?)\s*```", content_clean, re.DOTALL)
    if match:
        content_clean = match.group(1).strip()
    else:
        # Check if there is a generic code block
        match_code = re.search(r"```\s*(.*?)\s*```", content_clean, re.DOTALL)
        if match_code:
            content_clean = match_code.group(1).strip()
            
    # Sometimes it returns a bare json starting with {
    try:
        return json.loads(content_clean)
    except json.JSONDecodeError:
        # Try to search for the first '{' and last '}'
        start = content_clean.find("{")
        end = content_clean.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(content_clean[start:end+1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse LLM output as JSON: {content}")

# Node 1: Intent Router
async def route_intent(state: AgentState) -> AgentState:
    query = state["query"]
    logs = state.get("logs", [])
    logs.append(f"Analyzing user query intent: '{query}'")
    
    prompt = f"""
    You are an AI router. Your job is to classify the user's input query intent and extract parameters for a database agent.
    
    The user query: "{query}"
    
    Classify the intent into one of the following exact strings:
    - "ml_forecast": If the user explicitly asks to predict, forecast, project, or model future values/trends (e.g., "predict sales", "forecast visitors next 30 days").
    - "ml_segmentation": If the user asks to segment, cluster, group, or profile customers using ML (e.g., "cluster customers", "segment our buyers").
    - "conversational": If the user is just saying hello, asking who you are, or asking general support questions (e.g., "hi", "how are you", "what can you do").
    - "sql": For any standard database query, statistics, report, listing, aggregations, counts, sums, or filters (e.g., "show top 5 sales", "who is the best customer in China", "show orders in 2024"). If the query mentions database entities like orders, customers, total spent, or countries, but doesn't request forecasting or segmentation, it MUST be "sql".
    
    Also, extract parameters if applicable:
    - For "ml_forecast": extract "forecast_days" (integer, e.g. "next 30 days" -> 30, "60 days" -> 60, defaults to 30).
    - For "ml_segmentation": extract "num_clusters" (integer, e.g. "分成 5 类" -> 5, "cluster into 4 groups" -> 4, defaults to 3).
    
    Examples:
    - "Show me top 3 customers from China by total order amount" -> {{"intent": "sql", "ml_params": {{}}}}
    - "forecast unique visitors for 60 days" -> {{"intent": "ml_forecast", "ml_params": {{"forecast_days": 60}}}}
    - "使用机器学习K-Means算法对客户进行画像聚类分析，并且分成 5 类" -> {{"intent": "ml_segmentation", "ml_params": {{"num_clusters": 5}}}}
    
    Return ONLY a valid JSON object:
    {{
      "intent": "selected_category",
      "ml_params": {{
        "forecast_days": 30,
        "num_clusters": 3
      }}
    }}
    """
    
    try:
        response = await get_llm().ainvoke(prompt)
        res_json = parse_json_from_llm(response.content)
        intent = res_json.get("intent", "sql")
        ml_params = res_json.get("ml_params", {})
    except Exception as e:
        intent = "sql"
        ml_params = {}
        logs.append(f"Intent parsing failed: {e}. Defaulting to 'sql'")

    if intent not in VALID_INTENTS:
        logs.append(f"Unknown intent '{intent}', defaulting to 'sql'.")
        intent = "sql"
    if not isinstance(ml_params, dict):
        logs.append(f"Invalid ml_params type {type(ml_params).__name__}; using empty params.")
        ml_params = {}

    logs.append(f"Determined intent: {intent}, params: {ml_params}")
    return {
        **state,
        "intent": intent,
        "ml_params": ml_params,
        "logs": logs
    }

# Node 2: Text-to-SQL Generator (with self-correction logic)
async def generate_sql(state: AgentState) -> AgentState:
    # If we already have a success SQL and no new error, return
    if state.get("sql_query") and not state.get("error"):
        return state
        
    query = state["query"]
    schema = state["schema"]
    retry_count = state.get("retry_count", 0)
    error = state.get("error")
    logs = state.get("logs", [])
    
    logs.append(f"Generating SQL query (Attempt {retry_count + 1})...")
    
    retry_context = ""
    if error:
        retry_context = f"""
        WARNING: The previous attempt generated the SQL below, which failed with the error:
        Previous SQL: {state.get('sql_query')}
        Error Message: {error}
        Please analyze this error and correct the SQL query. Ensure you use valid table and column names from the schema.
        """
        
    prompt = f"""
    Given the SQLite database schema below:
    {schema}
    
    Translate the user's natural language request into a valid SQLite SELECT query.
    User request: {query}
    
    {retry_context}
    
    Rules for SQL generation:
    1. Only output standard, valid SQLite syntax.
    2. Write a single SELECT query. Do not use INSERT, UPDATE, DELETE, or DROP.
    3. Be careful with joins: use the correct foreign keys from the schema.
    4. For dates, SQLite supports text format YYYY-MM-DD. Use LIKE '2024%' or strftime('%Y-%m', order_date) to aggregate by month/year.
    5. Always return clean JSON containing 'sql' and a short 'explanation' key.
    
    Format:
    {{
      "sql": "SELECT ...",
      "explanation": "Summarize what columns were selected and why"
    }}
    """
    
    try:
        response = await get_llm().ainvoke(prompt)
        res_json = parse_json_from_llm(response.content)
        sql_query = res_json.get("sql")
        explanation = res_json.get("explanation", "")
        logs.append(f"Generated SQL: {sql_query}")
        logs.append(f"Explanation: {explanation}")
        
        # Strip trailing semicolon if LLM added it, though SQLite is fine with it
        sql_query = sql_query.strip().rstrip(";")
        
        return {
            **state,
            "sql_query": sql_query,
            "logs": logs
        }
    except Exception as e:
        logs.append(f"SQL generation crashed: {e}")
        return {
            **state,
            "error": str(e),
            "logs": logs
        }

# Node 3: Execute SQL Query
def execute_sql(state: AgentState) -> AgentState:
    sql_query = state.get("sql_query")
    logs = state.get("logs", [])
    
    if not sql_query:
        return {
            **state,
            "error": "No SQL query generated to execute.",
            "retry_count": state.get("retry_count", 0) + 1,
            "logs": logs
        }
        
    logs.append(f"Executing SQL on database: '{sql_query}'")
    
    res = execute_query(sql_query)
    
    if res["success"]:
        logs.append(f"Query succeeded, fetched {res['row_count']} rows.")
        return {
            **state,
            "data": res["data"],
            "columns": res["columns"],
            "error": None,
            "logs": logs
        }
    else:
        logs.append(f"Query failed with error: {res['error']}")
        return {
            **state,
            "error": res["error"],
            "retry_count": state.get("retry_count", 0) + 1,
            "logs": logs
        }

# Node 4: Generate Visualization (Chart.js Config)
async def generate_visualization(state: AgentState) -> AgentState:
    if state.get("error") or not state.get("data"):
        return state
        
    query = state["query"]
    columns = state["columns"]
    data = state["data"]
    logs = state.get("logs", [])
    
    logs.append("Generating Chart.js configuration...")
    
    sample_rows = data[:3]
    
    prompt = f"""
    Analyze the user's query and the data results schema to generate a beautiful, responsive Chart.js configuration.
    User prompt: {query}
    Columns: {columns}
    Sample Data (First 3 rows): {sample_rows}
    Total Rows: {len(data)}
    
    Rules:
    1. Select the most appropriate chart type. Supported: 'bar', 'line', 'pie', 'doughnut', 'polarArea'.
       - Use 'line' for chronological data, trends, or dates.
       - Use 'bar' for categories, comparison rankings.
       - Use 'pie' or 'doughnut' for percentage breakdown of parts of a whole (under 7 categories).
       - If a chart doesn't make sense (e.g. single value returned, or list of detailed records without aggregated metrics), return an empty JSON object {{}}.
    2. Format the output to be a valid JSON config that matches the Chart.js configuration structure (containing 'type', 'data', and 'options' keys).
    3. Colors: Make it look premium. Use modern theme color palettes (deep blues, teals, indigos, purples, oranges) with appropriate transparencies (RGBA).
    4. Options: Include responsive: true, maintainAspectRatio: false, grid line controls, and tooltips.
    5. The config must refer directly to the actual dataset fields in 'data'. Construct the labels array and datasets data array by extracting variables from the dataset.
    6. CRITICAL JSON COMPLIANCE: Do NOT output raw JavaScript functions (like `function(context) {{ ... }}` or arrow functions) directly inside the JSON values. Raw functions are not valid JSON and will crash the parser. If you need to include a Chart.js callback function (e.g., for tooltips), you MUST wrap the entire function definition in double quotes as a string (e.g., `"label": "function(context) {{ return context.label + ': ' + context.parsed + '%'; }}"`). The frontend will parse and execute it safely.
    
    Format example:
    {{
      "type": "bar",
      "data": {{
        "labels": ["Label1", "Label2"],
        "datasets": [
          {{
            "label": "Metric Name",
            "data": [120, 240],
            "backgroundColor": ["rgba(54, 162, 235, 0.6)"],
            "borderColor": ["rgba(54, 162, 235, 1)"],
            "borderWidth": 1
          }}
        ]
      }},
      "options": {{
        "responsive": true,
        "plugins": {{
          "legend": {{ "display": true }}
        }}
      }}
    }}
    
    Return ONLY the raw JSON block.
    """
    
    try:
        response = await get_llm().ainvoke(prompt)
        chart_config = parse_json_from_llm(response.content)
        logs.append(f"Chart.js configuration generated. Chart type: {chart_config.get('type', 'none')}")
        return {
            **state,
            "chart_config": chart_config if chart_config else None,
            "logs": logs
        }
    except Exception as e:
        logs.append(f"Visualization config generation failed: {e}")
        return {
            **state,
            "chart_config": None,
            "logs": logs
        }

# Node 5: Conversational Response Node
async def conversational_node(state: AgentState) -> AgentState:
    query = state["query"]
    logs = state.get("logs", [])
    logs.append("Routing to conversational response...")
    
    prompt = f"""
    The user is asking a conversational question or greeting.
    User request: {query}
    
    Respond in a professional and polite tone. Explain who you are:
    An Antigravity Database Agent that helps users query the cloud database using natural language (SQL extraction), visualize the results interactively, and run extensible machine learning tasks (like Sales Forecasting or Customer Segmentation).
    Keep it concise.
    """
    
    try:
        response = await get_llm().ainvoke(prompt)
        return {
            **state,
            "conversational_response": response.content,
            "logs": logs
        }
    except Exception as e:
        return {
            **state,
            "conversational_response": "Hello! I am your Antigravity DB Agent. How can I help you query the database or analyze your data today?",
            "logs": logs
        }

# Node 6: Extensible ML / Advanced Analysis Node
def run_ml_analysis(state: AgentState) -> AgentState:
    intent = state["intent"]
    query = state["query"]
    logs = state.get("logs", [])
    
    logs.append(f"Executing Machine Learning / Advanced Analysis: {intent}...")
    
    if intent == "ml_forecast":
        # Forecast Node logic
        # 1. Fetch web traffic data or order data depending on the query
        logs.append("Fetching data for forecasting regression model...")
        conn = get_connection()
        try:
            # Decide which data to forecast
            target_metric = "page_views"
            title = "Web Page Views Forecast"
            table = "web_traffic"
            date_col = "traffic_date"
            
            if "visitor" in query.lower() or "user" in query.lower():
                target_metric = "unique_visitors"
                title = "Unique Visitors Forecast"
            elif "conversion" in query.lower():
                target_metric = "conversions"
                title = "Conversions Forecast"
            elif "sale" in query.lower() or "order" in query.lower() or "revenue" in query.lower():
                table = "orders"
                date_col = "order_date"
                target_metric = "total_amount"
                title = "Daily Sales Revenue Forecast"
                
            if table == "web_traffic":
                df = pd.read_sql_query(f"SELECT {date_col}, {target_metric} FROM {table} ORDER BY {date_col}", conn)
            else: # Aggregate daily orders
                df = pd.read_sql_query(f"SELECT {date_col}, SUM({target_metric}) as {target_metric} FROM {table} GROUP BY {date_col} ORDER BY {date_col}", conn)
                
            if df.empty:
                raise ValueError("Insufficient data to perform ML forecasting.")
                
            # 2. Preprocess
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
            
            # Use past 90 days of history to fit and forecast
            df = df.tail(90).reset_index(drop=True)
            
            # Numeric X representing days from start
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[target_metric].values.astype(float)
            
            # Fit linear regression model: y = m*x + c
            # We can use numpy polyfit for simplicity or import scikit-learn LinearRegression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            r2 = float(model.score(X, y))
            slope = float(model.coef_[0])
            intercept = float(model.intercept_)
            
            # Get forecast days from params
            ml_params = state.get("ml_params") or {}
            forecast_days = parse_int_param(ml_params, "forecast_days", 30, 5, 180, logs)
            logs.append(f"Forecasting future {forecast_days} days...")
            
            # Predict future days
            X_future = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
            y_future = model.predict(X_future)
            # Ensure no negative predictions for physical quantities
            y_future = np.clip(y_future, 0, None)
            
            # Dates for future
            last_date = df[date_col].max()
            future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
            
            historical_list = [{"date": r[date_col].strftime("%Y-%m-%d"), "value": float(r[target_metric])} for _, r in df.iterrows()]
            forecast_list = [{"date": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in zip(future_dates, y_future)]
            
            # Create chart config
            labels = [h["date"] for h in historical_list] + [f["date"] for f in forecast_list]
            hist_values = [h["value"] for h in historical_list] + [None]*forecast_days
            fore_values = [None]*len(historical_list) + [f["value"] for f in forecast_list]
            
            chart_config = {
                "type": "line",
                "data": {
                    "labels": labels,
                    "datasets": [
                        {
                            "label": f"Historical {title}",
                            "data": hist_values,
                            "borderColor": "rgba(66, 133, 244, 1)",
                            "backgroundColor": "rgba(66, 133, 244, 0.1)",
                            "borderWidth": 2,
                            "fill": True,
                            "pointRadius": 2
                        },
                        {
                            "label": f"Linear Trend Forecast (Next {forecast_days} Days)",
                            "data": fore_values,
                            "borderColor": "rgba(234, 67, 53, 1)",
                            "borderWidth": 2,
                            "borderDash": [5, 5],
                            "fill": False,
                            "pointRadius": 3
                        }
                    ]
                },
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": f"{title} (R² = {r2:.4f})"
                        }
                    }
                }
            }
            
            ml_result = {
                "success": True,
                "model_type": "Linear Regression Time-Series Trend",
                "target_metric": target_metric,
                "metrics": {
                    "R_squared": round(r2, 4),
                    "slope": round(slope, 2),
                    "intercept": round(intercept, 2),
                    "direction": "upward" if slope > 0 else "downward"
                },
                "historical": historical_list[-15:], # Keep last 15 for table
                "forecast": forecast_list,
                "summary": f"Based on the last 90 days, {title} is showing a general { 'growth' if slope > 0 else 'declining' } trend (slope: {slope:.2f}/day). The model has a goodness-of-fit R² of {r2:.4f}."
            }
            
            logs.append(f"Forecasting complete. R2={r2:.4f}, slope={slope:.2f}")
            
            return {
                **state,
                "ml_result": ml_result,
                "chart_config": chart_config,
                "logs": logs
            }
        except Exception as e:
            logs.append(f"Forecasting model failed: {e}")
            return {
                **state,
                "error": f"ML Forecast Error: {str(e)}",
                "logs": logs
            }
        finally:
            conn.close()
            
    elif intent == "ml_segmentation":
        # Customer Segmentation using K-Means Clustering
        logs.append("Fetching customer metrics for segmentation clustering model...")
        conn = get_connection()
        try:
            # 1. Fetch SQL
            sql_cust = """
            SELECT c.customer_id, c.name, c.country, c.segment as business_tier,
                   COUNT(o.order_id) as total_orders,
                   COALESCE(SUM(o.total_amount), 0) as total_spent,
                   COALESCE(AVG(o.total_amount), 0) as avg_order_value
            FROM customers c
            LEFT JOIN orders o ON c.customer_id = o.customer_id
            GROUP BY c.customer_id
            """
            df = pd.read_sql_query(sql_cust, conn)
            
            if len(df) < 5:
                raise ValueError("Insufficient customer data to perform clustering (need at least 5 customers).")
                
            # 2. Run K-Means
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            features = ["total_orders", "total_spent", "avg_order_value"]
            X = df[features].values
            
            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit KMeans (dynamic clusters)
            ml_params = state.get("ml_params") or {}
            n_clusters = parse_int_param(ml_params, "num_clusters", 3, 2, 6, logs)
            n_clusters = min(n_clusters, len(df))
            logs.append(f"Fitting KMeans model with {n_clusters} clusters...")

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df["cluster"] = kmeans.fit_predict(X_scaled)
            
            # 3. Label Clusters dynamically based on mean total_spent
            cluster_means = df.groupby("cluster")["total_spent"].mean().sort_values()
            
            # Generate labels based on K
            cluster_map = {}
            if n_clusters == 3:
                labels_mapped = ["Standard/Low Value", "Loyal/Mid Value", "VIP/High Value"]
            elif n_clusters == 2:
                labels_mapped = ["Standard/Lower Value Cohort", "VIP/Higher Value Cohort"]
            else:
                labels_mapped = []
                for idx in range(n_clusters):
                    if idx == 0:
                        labels_mapped.append("Bronze/Lowest Value Cohort")
                    elif idx == n_clusters - 1:
                        labels_mapped.append("VIP/Highest Value Cohort")
                    else:
                        labels_mapped.append(f"Cohort {idx + 1}")
                        
            for idx, cluster_num in enumerate(cluster_means.index):
                cluster_map[cluster_num] = labels_mapped[idx]
                
            df["cluster_label"] = df["cluster"].map(cluster_map)
            
            # Compute centroid statistics
            centroids = []
            for c_num, label in cluster_map.items():
                c_df = df[df["cluster"] == c_num]
                centroids.append({
                    "cluster_id": int(c_num),
                    "label": label,
                    "size": int(len(c_df)),
                    "avg_orders": round(float(c_df["total_orders"].mean()), 2),
                    "avg_spent": round(float(c_df["total_spent"].mean()), 2),
                    "avg_ticket": round(float(c_df["avg_order_value"].mean()), 2)
                })
                
            # Prepare scatter plot config for frontend
            datasets = []
            colors = [
                {"background": "rgba(234, 67, 53, 0.6)", "border": "rgba(234, 67, 53, 1)"},    # Red
                {"background": "rgba(251, 188, 5, 0.6)", "border": "rgba(251, 188, 5, 1)"},    # Yellow
                {"background": "rgba(52, 168, 83, 0.6)", "border": "rgba(52, 168, 83, 1)"},    # Green
                {"background": "rgba(66, 133, 244, 0.6)", "border": "rgba(66, 133, 244, 1)"},  # Blue
                {"background": "rgba(161, 66, 244, 0.6)", "border": "rgba(161, 66, 244, 1)"},  # Purple
                {"background": "rgba(244, 143, 177, 0.6)", "border": "rgba(244, 143, 177, 1)"}  # Pink
            ]
            
            for i, centroid in enumerate(centroids):
                c_id = centroid["cluster_id"]
                cluster_data = df[df["cluster"] == c_id]
                points = [{"x": int(r["total_orders"]), "y": float(r["total_spent"]), "label": r["name"]} for _, r in cluster_data.iterrows()]
                
                datasets.append({
                    "label": centroid["label"],
                    "data": points,
                    "backgroundColor": colors[i % len(colors)]["background"],
                    "borderColor": colors[i % len(colors)]["border"],
                    "borderWidth": 1,
                    "pointRadius": 6,
                    "pointHoverRadius": 8
                })
                
            chart_config = {
                "type": "scatter",
                "data": {
                    "datasets": datasets
                },
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "scales": {
                        "x": {
                            "title": { "display": True, "text": "Total Number of Orders" }
                        },
                        "y": {
                            "title": { "display": True, "text": "Total Spent ($)" }
                        }
                    },
                    "plugins": {
                        "title": { "display": True, "text": f"Customer Segmentation (K-Means, K={n_clusters})" },
                        "tooltip": {
                            "callbacks": {
                                "label": "function(context) { return context.dataset.label + ': ' + context.raw.label + ' (Orders: ' + context.raw.x + ', Spent: $' + context.raw.y + ')'; }"
                            }
                        }
                    }
                }
            }
            
            customers_list = df[["customer_id", "name", "country", "total_orders", "total_spent", "cluster_label"]].to_dict(orient="records")
            # JSON cleanups
            for cust in customers_list:
                cust["total_spent"] = round(float(cust["total_spent"]), 2)
                
            ml_result = {
                "success": True,
                "model_type": f"K-Means Clustering Analysis (K={n_clusters})",
                "centroids": centroids,
                "customers": customers_list,
                "summary": f"Clustered customers into {n_clusters} behavioral segments: " +
                           ", ".join([f"{c['label']} ({c['size']} customers)" for c in centroids]) + "."
            }
            
            logs.append("Customer segmentation complete.")
            return {
                **state,
                "ml_result": ml_result,
                "chart_config": chart_config,
                "logs": logs
            }
        except Exception as e:
            logs.append(f"Clustering model failed: {e}")
            return {
                **state,
                "error": f"ML Segmentation Error: {str(e)}",
                "logs": logs
            }
        finally:
            conn.close()
    
    return state

# Build LangGraph Workflow
from langgraph.graph import StateGraph, START, END

def build_workflow():
    workflow = StateGraph(AgentState)
    
    # Define Nodes
    workflow.add_node("intent_router", route_intent)
    workflow.add_node("text_to_sql", generate_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("generate_visualization", generate_visualization)
    workflow.add_node("conversational_response", conversational_node)
    workflow.add_node("ml_analysis", run_ml_analysis)
    
    # Connect edges
    workflow.add_edge(START, "intent_router")
    
    # Conditional Router Edge from intent_router
    def route_decision(state: AgentState):
        intent = state.get("intent", "sql")
        if intent in ["ml_forecast", "ml_segmentation"]:
            return "ml_analysis"
        elif intent == "conversational":
            return "conversational_response"
        else:
            return "text_to_sql"
            
    workflow.add_conditional_edges(
        "intent_router",
        route_decision,
        {
            "ml_analysis": "ml_analysis",
            "conversational_response": "conversational_response",
            "text_to_sql": "text_to_sql"
        }
    )
    
    # Connection between Text-to-SQL and Execute SQL
    workflow.add_edge("text_to_sql", "execute_sql")
    
    # Conditional Edge for Self-Correction from execute_sql
    def check_execution_status(state: AgentState):
        error = state.get("error")
        retry_count = state.get("retry_count", 0)
        
        if error and retry_count < 3:
            return "retry"
        else:
            return "proceed"
            
    workflow.add_conditional_edges(
        "execute_sql",
        check_execution_status,
        {
            "retry": "text_to_sql",
            "proceed": "generate_visualization"
        }
    )
    
    # Endings
    workflow.add_edge("generate_visualization", END)
    workflow.add_edge("ml_analysis", END)
    workflow.add_edge("conversational_response", END)
    
    return workflow.compile()

if __name__ == "__main__":
    from db_manager import init_db
    init_db()
    
    # Simple Local Dry Run
    graph = build_workflow()
    
    # Let's run a test state
    initial_state = create_initial_state("Show me total page views in web_traffic table")
    
    print("Running graph invocation test...")
    res = asyncio.run(graph.ainvoke(initial_state))
    print("\nLogs:")
    for log in res["logs"]:
        print(f" -> {log}")
    print("\nSQL Output:", res.get("sql_query"))
    print("Data Row Count:", len(res.get("data") or []))
    print("Chart Config Output:", res.get("chart_config"))
