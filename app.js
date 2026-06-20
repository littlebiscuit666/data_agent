/**
 * Antigravity Extensible Data Agent Dashboard Logic
 * Handles interactive sidebar schema loading, example actions, tab routing,
 * Chart.js rendering, ML result display, and FastAPI communication.
 */

document.addEventListener('DOMContentLoaded', () => {
  
  // API URLs
  const API_SCHEMA = '/api/schema';
  const API_QUERY = '/api/query';
  const API_RESET = '/api/reset';
  
  // DOM Elements
  const schemaContainer = document.getElementById('schema-tree-container');
  const queryInput = document.getElementById('nl-query-input');
  const btnSubmit = document.getElementById('btn-submit-query');
  const btnResetDb = document.getElementById('btn-reset-db');
  
  const workspaceLoading = document.getElementById('workspace-loading');
  const loadingMessage = document.getElementById('loading-message');
  const workspaceEmpty = document.getElementById('workspace-empty');
  
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const tabBtnMl = document.getElementById('tab-btn-ml');
  
  const sqlCode = document.getElementById('generated-sql-code');
  const logsList = document.getElementById('agent-logs-list');
  
  const tableHeaders = document.getElementById('table-headers');
  const tableRows = document.getElementById('table-rows');
  const tableMetaInfo = document.getElementById('table-meta-info');
  
  const noChartAlert = document.getElementById('no-chart-alert');
  const chartCanvas = document.getElementById('agent-chart-canvas');
  
  // ML Elements
  const mlModelType = document.getElementById('ml-model-type');
  const mlMetricR2 = document.getElementById('ml-metric-r2');
  const mlMetricTrend = document.getElementById('ml-metric-trend');
  const mlModelDesc = document.getElementById('ml-model-desc');
  const mlTableTitle = document.getElementById('ml-table-title');
  const mlTableHeaders = document.getElementById('ml-table-headers');
  const mlTableRows = document.getElementById('ml-table-rows');
  const rowR2 = document.getElementById('row-r2');
  const rowTrend = document.getElementById('row-trend');
  
  let currentChartInstance = null;

  /* ==========================================================================
     1. Database Schema Loading & Rendering
     ========================================================================== */
  async function loadDatabaseSchema() {
    schemaContainer.innerHTML = '<div class="loading-spinner">读取架构中...</div>';
    try {
      const response = await fetch(API_SCHEMA);
      if (!response.ok) throw new Error("获取数据库架构失败。");
      const data = await response.json();
      renderSchemaTree(data.structured);
    } catch (error) {
      console.error(error);
      schemaContainer.innerHTML = `<div class="log-item error">架构读取失败: ${error.message}</div>`;
    }
  }

  function renderSchemaTree(schema) {
    if (!schema || Object.keys(schema).length === 0) {
      schemaContainer.innerHTML = '<div class="log-item system">当前无表结构。</div>';
      return;
    }
    
    schemaContainer.innerHTML = '';
    
    for (const tableName in schema) {
      const tableNode = document.createElement('div');
      tableNode.className = 'db-table-node';
      
      const header = document.createElement('div');
      header.className = 'table-node-header';
      header.innerHTML = `
        <div class="table-node-title">
          <span class="material-symbols-outlined">table</span>
          <span>${tableName}</span>
        </div>
        <span class="material-symbols-outlined table-node-toggle">chevron_right</span>
      `;
      
      const columnsList = document.createElement('div');
      columnsList.className = 'table-node-columns';
      
      schema[tableName].forEach(col => {
        const colItem = document.createElement('div');
        colItem.className = 'column-item';
        colItem.innerHTML = `
          <span class="col-name ${col.pk ? 'pk' : ''}">${col.name}</span>
          <span class="col-type">${col.type}</span>
        `;
        columnsList.appendChild(colItem);
      });
      
      tableNode.appendChild(header);
      tableNode.appendChild(columnsList);
      
      // Expand/Collapse click event
      header.addEventListener('click', () => {
        tableNode.classList.toggle('expanded');
      });
      
      schemaContainer.appendChild(tableNode);
    }
  }

  /* ==========================================================================
     2. Tab Switching Logic
     ========================================================================== */
  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const targetTabId = button.getAttribute('data-tab');
      
      tabButtons.forEach(btn => btn.classList.remove('active'));
      tabPanes.forEach(pane => pane.classList.remove('active'));
      
      button.classList.add('active');
      document.getElementById(targetTabId).classList.add('active');
    });
  });

  function switchTab(tabId) {
    const btn = Array.from(tabButtons).find(b => b.getAttribute('data-tab') === tabId);
    if (btn) btn.click();
  }

  /* ==========================================================================
     3. Submit natural language queries to Agent
     ========================================================================== */
  async function submitQuery(queryText) {
    if (!queryText.trim()) return;
    
    // UI state: loading
    workspaceEmpty.style.display = 'none';
    workspaceLoading.style.display = 'flex';
    loadingMessage.textContent = "智能体正在规划并转换自然语言...";
    
    // Hide ML tab initially
    tabBtnMl.style.display = 'none';
    
    try {
      const response = await fetch(API_QUERY, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "智能体运行出错。");
      }
      
      const result = await response.json();
      
      // Clear loading
      workspaceLoading.style.display = 'none';
      
      // Render components
      renderAgentLogs(result.logs, result.error);
      renderSqlResult(result.sql_query);
      renderTableResult(result.columns, result.data);
      renderChartResult(result.chart_config);
      
      // Handle ML Result
      if (result.intent === 'ml_forecast' || result.intent === 'ml_segmentation') {
        renderMlResult(result.intent, result.ml_result);
        tabBtnMl.style.display = 'inline-flex';
        // Auto-switch to ML result tab
        switchTab('tab-ml');
      } else {
        // Normal SQL query: auto-switch to chart or table
        if (result.chart_config && Object.keys(result.chart_config).length > 0) {
          switchTab('tab-chart');
        } else {
          switchTab('tab-table');
        }
      }
      
    } catch (error) {
      console.error(error);
      workspaceLoading.style.display = 'none';
      workspaceEmpty.style.display = 'flex';
      workspaceEmpty.innerHTML = `
        <span class="material-symbols-outlined empty-icon" style="color: var(--color-danger)">error</span>
        <h3 style="color: var(--color-danger)">智能体执行失败</h3>
        <p>${error.message}</p>
        <button class="btn btn-secondary" onclick="window.location.reload()" style="margin-top: 1rem;">刷新页面</button>
      `;
    }
  }

  // Render logs helper
  function renderAgentLogs(logs, error) {
    logsList.innerHTML = '';
    
    if (logs && logs.length > 0) {
      logs.forEach(log => {
        const logDiv = document.createElement('div');
        logDiv.className = 'log-item';
        
        if (log.includes('Analyzing user query')) {
          logDiv.className += ' router';
        } else if (log.includes('Query succeeded') || log.includes('complete')) {
          logDiv.className += ' success';
        } else if (log.includes('failed') || log.includes('Error')) {
          logDiv.className += ' error';
        }
        
        logDiv.textContent = log;
        logsList.appendChild(logDiv);
      });
    }
    
    if (error) {
      const errorDiv = document.createElement('div');
      errorDiv.className = 'log-item error';
      errorDiv.textContent = `CRITICAL ERROR: ${error}`;
      logsList.appendChild(errorDiv);
    }
  }

  // Render SQL text helper
  function renderSqlResult(sql) {
    if (sql) {
      sqlCode.textContent = sql;
    } else {
      sqlCode.textContent = '-- 意图未涉及 SQL 生成或未生成语句';
    }
  }

  // Render table helper
  function renderTableResult(columns, data) {
    tableHeaders.innerHTML = '';
    tableRows.innerHTML = '';
    
    if (!columns || columns.length === 0 || !data || data.length === 0) {
      tableMetaInfo.textContent = "共 0 行数据";
      tableRows.innerHTML = '<tr><td colspan="100" style="text-align: center; color: var(--text-muted);">无可用数据结果</td></tr>';
      return;
    }
    
    // Draw headers
    columns.forEach(col => {
      const th = document.createElement('th');
      th.textContent = col;
      tableHeaders.appendChild(th);
    });
    
    // Draw rows
    data.forEach(row => {
      const tr = document.createElement('tr');
      columns.forEach(col => {
        const td = document.createElement('td');
        const val = row[col];
        td.textContent = (val !== null && val !== undefined) ? val : '-';
        tr.appendChild(td);
      });
      tableRows.appendChild(tr);
    });
    
    tableMetaInfo.textContent = `共 ${data.length} 行数据`;
  }

  // Parse Chart.js stringified functions (e.g. tooltips)
  function parseChartOptions(options) {
    if (!options) return options;
    
    if (options.plugins && options.plugins.tooltip && options.plugins.tooltip.callbacks) {
      const callbacks = options.plugins.tooltip.callbacks;
      for (let key in callbacks) {
        if (typeof callbacks[key] === 'string' && callbacks[key].trim().startsWith('function')) {
          try {
            // Re-bind code string to native JS function
            callbacks[key] = new Function('return ' + callbacks[key])();
          } catch (e) {
            console.error("Failed to parse callback function:", e);
          }
        }
      }
    }
    return options;
  }

  // Render chart helper
  function renderChartResult(config) {
    // Destroy previous chart
    if (currentChartInstance) {
      currentChartInstance.destroy();
      currentChartInstance = null;
    }
    
    if (!config || Object.keys(config).length === 0) {
      chartCanvas.style.display = 'none';
      noChartAlert.style.display = 'flex';
      return;
    }
    
    chartCanvas.style.display = 'block';
    noChartAlert.style.display = 'none';
    
    try {
      // Parse potential tooltip functions
      config.options = parseChartOptions(config.options);
      
      // Enforce responsive sizing in dashboard
      if (config.options) {
        config.options.responsive = true;
        config.options.maintainAspectRatio = false;
      }
      
      currentChartInstance = new Chart(chartCanvas, config);
    } catch (e) {
      console.error("Chart.js creation crashed:", e);
      chartCanvas.style.display = 'none';
      noChartAlert.style.display = 'flex';
      noChartAlert.querySelector('p').textContent = `生成图表失败: ${e.message}`;
    }
  }

  // Render Machine Learning helper
  function renderMlResult(intent, mlResult) {
    mlTableHeaders.innerHTML = '';
    mlTableRows.innerHTML = '';
    
    if (!mlResult || !mlResult.success) {
      mlModelType.textContent = "模型执行错误";
      mlModelDesc.textContent = mlResult ? mlResult.error : "未知 ML 错误";
      return;
    }
    
    mlModelType.textContent = mlResult.model_type;
    mlModelDesc.textContent = mlResult.summary;
    
    if (intent === 'ml_forecast') {
      rowR2.style.display = 'flex';
      rowTrend.style.display = 'flex';
      
      mlMetricR2.textContent = mlResult.metrics.R_squared;
      mlMetricTrend.textContent = mlResult.metrics.direction === 'upward' ? '增长趋势 📈' : '下降趋势 📉';
      mlMetricTrend.className = 'metric-value ' + (mlResult.metrics.direction === 'upward' ? 'color-success' : 'color-danger');
      
      const forecastDays = Array.isArray(mlResult.forecast) ? mlResult.forecast.length : 0;
      mlTableTitle.textContent = `未来${forecastDays}天趋势预测数据`;
      
      // Draw Forecast headers
      ['日期/时间', '预测值'].forEach(headerText => {
        const th = document.createElement('th');
        th.textContent = headerText;
        mlTableHeaders.appendChild(th);
      });
      
      // Draw Forecast rows
      mlResult.forecast.forEach(item => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${item.date}</td>
          <td><strong>${item.value.toFixed(2)}</strong></td>
        `;
        mlTableRows.appendChild(tr);
      });
      
    } else if (intent === 'ml_segmentation') {
      rowR2.style.display = 'none';
      rowTrend.style.display = 'none';
      
      mlTableTitle.textContent = "客户聚类分群标签";
      
      // Draw Segment headers
      ['ID', '姓名', '国家', '订单数', '消费总额', '聚类分群标签'].forEach(headerText => {
        const th = document.createElement('th');
        th.textContent = headerText;
        mlTableHeaders.appendChild(th);
      });
      
      // Draw Segment rows
      mlResult.customers.forEach(cust => {
        const tr = document.createElement('tr');
        
        let labelColor = 'var(--text-secondary)';
        if (cust.cluster_label.includes('VIP')) labelColor = 'var(--color-success)';
        else if (cust.cluster_label.includes('Loyal')) labelColor = 'var(--color-warning)';
        
        tr.innerHTML = `
          <td>${cust.customer_id}</td>
          <td>${cust.name}</td>
          <td>${cust.country}</td>
          <td>${cust.total_orders}</td>
          <td>$${cust.total_spent.toFixed(2)}</td>
          <td style="color: ${labelColor}; font-weight: 600;">${cust.cluster_label}</td>
        `;
        mlTableRows.appendChild(tr);
      });
    }
  }

  /* ==========================================================================
     4. Button Event Handlers
     ========================================================================== */
  
  // Submit Query
  btnSubmit.addEventListener('click', () => {
    submitQuery(queryInput.value);
  });
  
  // Handle Enter key inside query input
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      submitQuery(queryInput.value);
    }
  });
  
  // Click on examples loads and runs them
  document.querySelectorAll('.example-item').forEach(item => {
    item.addEventListener('click', () => {
      const q = item.getAttribute('data-query');
      queryInput.value = q;
      submitQuery(q);
    });
  });
  
  // Reset database connection & records
  btnResetDb.addEventListener('click', async () => {
    if (!confirm("确定要重置模拟云端数据库吗？这会重新填充随机业务和访客数据。")) return;
    
    btnResetDb.disabled = true;
    btnResetDb.innerHTML = '<span class="material-symbols-outlined">sync</span><span>Resetting...</span>';
    
    try {
      const response = await fetch(API_RESET, { method: 'POST' });
      if (!response.ok) throw new Error("重置数据库失败。");
      const res = await response.json();
      
      alert(res.message);
      // Reload schema tree
      await loadDatabaseSchema();
    } catch (e) {
      alert("错误: " + e.message);
    } finally {
      btnResetDb.disabled = false;
      btnResetDb.innerHTML = '<span class="material-symbols-outlined">restart_alt</span><span>Reset Database</span>';
    }
  });

  /* ==========================================================================
     5. App Startup Initialization
     ========================================================================== */
  loadDatabaseSchema();
});
