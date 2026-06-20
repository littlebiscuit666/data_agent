# Antigravity Data Agent - Architecture Diagram

Here is the Mermaid code representing the agent's workflow architecture, along with a detailed explanation of each node and conditional routing rule.

## 📊 Workflow Flowchart (Mermaid)

```mermaid
graph TD
    User([User Prompt]) --> Frontend[Premium Web UI Dashboard]
    Frontend --> Backend[FastAPI Server]
    Backend --> Manager[LangGraph Workflow Manager]
    
    subgraph Agent Core (LangGraph Workflow)
        Manager --> Router{Intent Router}
        Router -- NL-to-SQL --> TextToSQL[SQL Generator Node]
        TextToSQL --> ExecuteSQL[SQL Executor Node]
        ExecuteSQL --> SQLSuccess{Success?}
        SQLSuccess -- Yes --> Visualize[Visualization Node]
        SQLSuccess -- No (Error) --> TextToSQL
        
        Router -- ML / Analysis --> MLAnalysis[ML & Advanced Analysis Node]
        Router -- Conversational --> ConvNode[Conversational Node]
        
        Visualize --> EndNode[Final State Synthesis]
        MLAnalysis --> EndNode
        ConvNode --> EndNode
    end
    
    ExecuteSQL --> CloudDB[(Cloud Database SQLite)]
    EndNode --> Backend
    Backend --> Frontend
```

---

## ⚙️ Node-by-Node Explanation

### 1. Intent Router (`route_intent` Node)
* **Goal**: Analyze the user's natural language input and classify it to select the correct execution path. It also extracts key parameters.
* **Classification categories**:
  * `sql`: For standard database queries (sorting, joining, grouping, filters).
  * `ml_forecast`: For predictive analysis (regression, time-series forecasting).
  * `ml_segmentation`: For customer clustering and profiling.
  * `conversational`: For general greetings or chat.
* **Parameter extraction**:
  * Extracts `num_clusters` (number of groups) for clustering.
  * Extracts `forecast_days` (future days) for forecasting.

### 2. Text-to-SQL Generator (`generate_sql` Node)
* **Goal**: Translate user natural language into valid SQLite code.
* **Schema Guidance**: Reads tables, columns, data types, and sample rows from the database schema and injects them into the prompt to guide the LLM.
* **Self-Correction**: If the previous query attempt failed with an execution error, this node is called again. It receives the failed SQL statement and the exact traceback error message to output a corrected query.

### 3. SQL Executor (`execute_sql` Node)
* **Goal**: Safely execute the generated SQL.
* **Security Rules**: Only allows `SELECT` statements. Blocks modify operations like `INSERT`, `UPDATE`, `DELETE`, `DROP`, or `ALTER`.
* **Flow Routing**:
  * If the query **succeeds**, the node populates the state with the raw records and redirects to the **Visualization Node**.
  * If the query **fails**, it increases the retry counter, records the error traceback in the state, and loops back to the **Text-to-SQL Generator** (up to 3 times).

### 4. Visualization Node (`generate_visualization` Node)
* **Goal**: Select the best chart type and prepare Chart.js configs.
* **Chart Selection**:
  * `line` for dates and time-series trends.
  * `bar` for comparing categories.
  * `pie` / `doughnut` for percentage breakdowns of categories.
  * `scatter` for multidimensional data like customer distributions.
* **Config Format**: Outputs a valid JSON configuration object containing the Chart.js properties (`type`, `data`, `options`). 

### 5. ML & Advanced Analysis Node (`run_ml_analysis` Node)
* **Goal**: Run local python machine learning algorithms using `scikit-learn` and `pandas` on the data.
* **Forecast regression**: Runs a `LinearRegression` model to calculate R-squared coefficients and forecast trends.
* **K-Means Clustering**: Runs a `KMeans` model, normalizes transaction metrics, and clusters customers into dynamic categories based on centroid values.

### 6. Conversational Node (`conversational_node` Node)
* **Goal**: Directly reply to greetings, system help requests, or out-of-scope questions without running database queries.

---

## 🔗 How to Render this Diagram
Many tools can render this Markdown file directly:
1. **GitHub**: GitHub natively renders `mermaid` code blocks inside `.md` files.
2. **VS Code**: Install the extension "Markdown Preview Mermaid Support" or "Markdown Preview Enhanced".
3. **Mermaid Live Editor**: Copy and paste the block above into [mermaid.live](https://mermaid.live).
