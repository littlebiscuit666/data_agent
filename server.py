import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from db_manager import init_db, get_schema_info, get_connection
from agent_graph import build_workflow, create_initial_state

# Initialize database on start
init_db(force=False)

app = FastAPI(
    title="Antigravity Extensible Data Agent API",
    description="Backend API for natural language database queries, visualization configuration, and machine learning models."
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compile LangGraph once
graph = build_workflow()

class QueryRequest(BaseModel):
    query: str

def get_structured_schema():
    """Extracts schema in a structured dict format for frontend UI representation."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_dict = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            # col format: (cid, name, type, notnull, dflt_value, pk)
            schema_dict[table] = [
                {"name": col[1], "type": col[2], "pk": bool(col[5])}
                for col in columns
            ]
        return schema_dict
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()

@app.get("/api/schema")
async def get_schema():
    """Returns database schema metadata for visualization in the frontend."""
    return {
        "structured": get_structured_schema(),
        "raw_text": get_schema_info()
    }

@app.post("/api/query")
async def execute_agent_query(request: QueryRequest):
    """Executes the LangGraph agent workflow for natural language queries."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    # Get current database schema raw description
    raw_schema = get_schema_info()
    
    # Setup initial state
    initial_state = create_initial_state(request.query, raw_schema)
    
    try:
        # Invoke LangGraph asynchronously so LLM I/O does not block the event loop.
        result_state = await graph.ainvoke(initial_state)
        
        # Format clean response
        response = {
            "query": request.query,
            "intent": result_state.get("intent"),
            "sql_query": result_state.get("sql_query"),
            "data": result_state.get("data"),
            "columns": result_state.get("columns"),
            "chart_config": result_state.get("chart_config"),
            "ml_result": result_state.get("ml_result"),
            "conversational_response": result_state.get("conversational_response"),
            "error": result_state.get("error"),
            "logs": result_state.get("logs", [])
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent workflow execution failed: {str(e)}")

@app.post("/api/reset")
async def reset_database():
    """Wipes and recreates the SQLite database with new synthetic records."""
    try:
        init_db(force=True)
        return {"success": True, "message": "Database reset and populated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")

# Serve Static Files (Frontend UI Dashboard)
workspace_dir = os.path.dirname(os.path.abspath(__file__))

# Serve main index.html at root
@app.get("/")
async def get_index():
    index_path = os.path.join(workspace_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "index.html not found in workspace."}

# Mount other static assets directly (app.js, style.css, etc.)
app.mount("/", StaticFiles(directory=workspace_dir), name="static")

if __name__ == "__main__":
    # HF Spaces sets PORT=7860; locally defaults to 8000.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
