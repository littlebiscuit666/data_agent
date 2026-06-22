---
title: Data Agent
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Antigravity Extensible Data Agent

An agentic database analysis system that translates natural language queries into SQL, executes them against a cloud-like database, visualizes results, and performs machine learning tasks. Built with **LangGraph**, **FastAPI**, and **Chart.js**.

---

## 🌟 Key Features

* **NL-to-SQL Engine**: Ask questions in natural language and retrieve structured records instantly.
* **Agentic Self-Correction**: If a generated SQL statement fails to execute, the agent captures the traceback error and prompts the LLM to self-heal the query (up to 3 retries).
* **Interactive Charting**: Automatically infers database schemas and dynamically compiles modern Chart.js visual configs (supporting bar, line, pie, doughnut, and scatter graphs).
* **Dynamic Parameterized Machine Learning (Extensible)**:
  * *Sales/Traffic Forecasting*: Fits a `LinearRegression` model using scikit-learn to forecast trends over user-defined intervals (e.g., predict the next 60 days).
  * *Customer Segmentation*: Triggers K-Means clustering dynamically based on the requested cluster count (e.g., group customers into 5 cohorts) and visualizes them on a scatter plot.
* **Premium Frosted-Glass Dashboard**: Tech-aesthetic UI dashboard displaying responsive tables, code blocks, thinking logs, and ML centroids side-by-side.

---

## 📂 Project Structure

```text
data_agent/
├── agent_graph.py     # LangGraph workflow orchestration (Router, SQL Generator, ML Nodes)
├── db_manager.py      # SQLite connection manager & synthetic business data populator
├── server.py          # FastAPI application server & REST APIs
├── config.py          # LLM connection config (LangChain ChatOpenAI setup)
├── test_agent.py      # Automated integration and unit test suite
├── index.html         # Frosted-glass Web UI layout
├── style.css          # Premium tech-themed dark mode styles
├── app.js             # Frontend API integration & Chart.js dynamic rendering
├── architecture.md    # Detailed architecture documentation & Mermaid flowchart
├── requirements.txt   # Python library dependencies
└── .gitignore         # Keeps API keys (.env) and databases private
```

---

## 🛠️ Setup & Installation

### Prerequisites
* Python 3.10 to 3.14
* DeepSeek API Key

### Step 1: Clone and Navigate
Navigate into the application directory that contains `server.py`, `config.py`, and `requirements.txt`:
```bash
cd data-agent/data_agent
```

### Step 2: Set up Virtual Environment & Install Dependencies
Initialize a virtual environment and install the required packages:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Create a `.env` file in the `data_agent` folder next to `server.py` and `config.py`, then add your DeepSeek API key:
```env
DEEPSEEK_API_KEY=your_actual_deepseek_api_key
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

### SQL Safety Note
Generated SQL is executed through a read-only SQLite connection. The backend only accepts one read-only `SELECT` statement, or one read-only `WITH ... SELECT` statement, and rejects write or schema-changing commands.

---

## 🚀 Running the Application

### 1. Run Automated Test Suite
To verify database seeding, intent routing, SQL generation, and ML clustering/forecasting nodes:
```bash
python test_agent.py
```

### 2. Start the Backend Server
Launch the FastAPI uvicorn server:
```bash
python server.py
```

### 3. Open the UI Dashboard
Open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📈 Sample Prompts to Try
* *"Show me the top 5 customers from China by total order amount"* (Generates SQL joins & bar chart)
* *"Compare daily page views and conversions"* (Generates aggregated statistics & line chart)
* *"Predict unique visitors for the next 60 days"* (Runs regression trend ML model)
* *"Group our customers into 5 cohorts using machine learning"* (Runs K-Means clustering ML model)
