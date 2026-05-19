# Maryland Traffic Real-Time Risk Pipeline (Databricks)

End-to-end Lakehouse pipeline that fuses **historical Maryland traffic incidents** with a **live weather stream** to surface real-time corridor risk scores and natural-language alerts. Built entirely on Databricks using Delta Lake, Spark Structured Streaming, Mosaic AI Functions, Genie, and AI/BI Dashboards.

---

# Architecture

```
new data1.csv (11,000 incidents)
        │
        ▼
   Delta Table (incidents_raw)
        │
        ▼  ai_classify + ai_extract  ← Mosaic AI Function (LLM-in-SQL)
        ▼
   incidents_enriched ──────► road_priors (per-corridor history)
                                       │
Open-Meteo API ──► JSON landing ──► Structured Streaming ──► weather_live
                                                                │
                                                                ▼
                                                  corridor_risk_view  ←── joins live + historical
                                                                │
                                                                ▼
                                                        ai_gen (alerts)
                                                                │
                                                                ▼
                                                Genie (NL Q&A) + AI/BI Dashboard
```

---

# Key results

- **11,000 incidents** automatically classified into 6 categories by an LLM directly from SQL — no Python ML pipeline, no model training.
- Discovered that **incident type predicts duration by an order of magnitude**: Disabled Vehicle ≈ 16 min vs. Roadwork ≈ 7 hours.
- **8 Maryland corridors** monitored in real time (I-95, I-695, I-70, I-270, US-50, MD-295, MD-100, MD-75).
- Real-time **risk score** per corridor (0–100) blending historical incident count with live weather severity.
- **Natural-language alerts** generated per corridor on every refresh.

# Incident classification breakdown (LLM-generated)

| Category | Count | Avg Duration |
|---|---|---|
| Disabled Vehicle | 6,422 | 16.2 min |
| Collision | 2,138 | 45.9 min |
| Debris | 1,155 | 42.8 min |
| Other | 565 | 155.9 min |
| Weather Event | 408 | 348.1 min |
| Roadwork | 301 | 439.3 min |

---

# Stack

`Databricks` · `Unity Catalog` · `Delta Lake` · `PySpark Structured Streaming` · `Mosaic AI Functions (ai_classify, ai_extract, ai_gen)` · `Genie` · `AI/BI Dashboards` · `SQL` · `Python`

---

# Screenshots

## LLM classifies free-text incident descriptions in SQL
![AI Classification](screenshots/ai_classification.png)

## Distribution across the 6 LLM-assigned categories
![Category Counts](screenshots/category_counts.png)

## Top Maryland roads by historical incident pressure
![Road Priors](screenshots/road_priors.png)

## Real-time risk view — live weather joined with historical priors
![Risk View](screenshots/risk_view.png)

## AI-generated alerts per corridor
![AI Alerts](screenshots/ai_alerts.png)

## Genie natural-language Q&A over the tables
![Genie](screenshots/genie.png)

## Published AI/BI Dashboard
![Dashboard](screenshots/dashboard.png)

---

# How to reproduce

1. Sign up for **Databricks Free Edition** (https://www.databricks.com/learn/free-edition) or any workspace with Mosaic AI Functions.
2. **Import** `databricks_realtime_agent.py` into your workspace (Workspace → Import → File). Databricks auto-detects the `# Databricks notebook source` header and converts it to a notebook.
3. **Upload** `new data1.csv` (Maryland 2016 incident data) to the Volume `workspace.traffic_agent.raw` created by Cell 0.
4. **Attach** the notebook to a Serverless compute (top-right Connect dropdown).
5. **Run** cells top-to-bottom. The streaming producer runs in a background thread and the Spark Streaming reader writes a Delta table.
6. **Build Genie + Dashboard** in the UI using `incidents_enriched`, `road_priors`, and `corridor_risk_view`.

---

# What this project demonstrates

| Skill | Where in this project |
|---|---|
| **SQL fluency** | Window functions, CTEs, regex joins, AI Functions, view creation |
| **Delta Lake / Lakehouse** | Bronze (raw) → Silver (enriched) → Gold (risk_view) progression |
| **Spark Structured Streaming** | JSON file source → Delta sink with checkpoints |
| **LLM integration** | `ai_classify`, `ai_extract`, `ai_gen` called from SQL |
| **Real-time joins** | Latest-row-per-key join between streaming and batch tables |
| **Data visualization** | AI/BI Dashboard with bar, donut, KPI, line widgets |
| **Natural-language interfaces** | Genie space configured for analyst Q&A |
| **Python (PySpark + threading)** | Background data producer, streaming reader |

---

