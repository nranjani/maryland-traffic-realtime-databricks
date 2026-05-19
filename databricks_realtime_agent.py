# Databricks notebook source
# MAGIC %md
# MAGIC # Real-Time Traffic Risk Pipeline on Databricks
# MAGIC ### Delta Lake + Structured Streaming + Mosaic AI Functions + Genie
# MAGIC
# MAGIC **Author:** Ranjani Narayanaswamy 
# MAGIC
# MAGIC This notebook builds an end-to-end real-time pipeline:
# MAGIC
# MAGIC 1. **Ingest** the historical Maryland incident CSV into a Delta table (Unity Catalog).
# MAGIC 2. **Enrich** the 11K free-text incident descriptions with Mosaic AI Functions (`ai_classify`, `ai_extract`) — directly in SQL.
# MAGIC 3. **Stream** live weather from Open-Meteo into a second Delta table using Spark Structured Streaming.
# MAGIC 4. **Join** live weather against the enriched history to produce a per-corridor real-time risk view.
# MAGIC 5. **Expose** the result through a Genie space (natural-language Q&A) and an AI/BI dashboard.
# MAGIC
# MAGIC Total runtime: ~10 minutes once the cluster is up.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuration
# MAGIC Set your catalog/schema names. If you're on a workspace with Unity Catalog disabled, use `hive_metastore` as the catalog.

# COMMAND ----------

CATALOG = "workspace"          # change to "hive_metastore" if UC not enabled
SCHEMA  = "traffic_agent"
RAW_TABLE       = f"{CATALOG}.{SCHEMA}.incidents_raw"
ENRICHED_TABLE  = f"{CATALOG}.{SCHEMA}.incidents_enriched"
WEATHER_TABLE   = f"{CATALOG}.{SCHEMA}.weather_live"
RISK_TABLE      = f"{CATALOG}.{SCHEMA}.corridor_risk"

# Path where you'll upload new data1.csv
RAW_CSV_PATH = "/Volumes/{}/{}/raw/new_data1.csv".format(CATALOG, SCHEMA)
# Alt path if you uploaded to DBFS instead of a Volume:
# RAW_CSV_PATH = "/FileStore/tables/new_data1.csv"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA  IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME  IF NOT EXISTS {CATALOG}.{SCHEMA}.raw")

print(f"Using {CATALOG}.{SCHEMA}")
print(f"Upload new data1.csv to:  {RAW_CSV_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Ingest the historical CSV into a Delta table
# MAGIC Upload `new data1.csv` to the Volume path printed above (Catalog Explorer -> traffic_agent -> raw -> Upload to this volume). Then run this cell.

# COMMAND ----------

# Just dump the CSV to Delta as raw strings - no conversion in PySpark
df_raw = (
    spark.read
    .option("header", True)
    .csv(RAW_CSV_PATH)
)

(df_raw.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(RAW_TABLE))

print(f"Wrote {df_raw.count():,} rows to {RAW_TABLE}")
display(spark.table(RAW_TABLE).limit(5))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Convert string timestamps and add derived columns using pure SQL
# MAGIC CREATE OR REPLACE TABLE workspace.traffic_agent.incidents_raw AS
# MAGIC SELECT
# MAGIC     incident_id, n_tmc, direction, road_inc, road_tmc,
# MAGIC     try_to_timestamp(start_time, 'M/d/yyyy H:mm') AS start_time,
# MAGIC     try_to_timestamp(end_time,   'M/d/yyyy H:mm') AS end_time,
# MAGIC     description, weather,
# MAGIC     (unix_timestamp(try_to_timestamp(end_time,   'M/d/yyyy H:mm'))
# MAGIC    - unix_timestamp(try_to_timestamp(start_time, 'M/d/yyyy H:mm'))) / 60.0 AS duration_min,
# MAGIC     hour(try_to_timestamp(start_time, 'M/d/yyyy H:mm'))      AS hour,
# MAGIC     dayofweek(try_to_timestamp(start_time, 'M/d/yyyy H:mm')) AS dow
# MAGIC FROM workspace.traffic_agent.incidents_raw;
# MAGIC
# MAGIC SELECT COUNT(*) AS rows_after_conversion FROM workspace.traffic_agent.incidents_raw;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. AI-enrich the incident descriptions with Mosaic AI Functions
# MAGIC The `ai_classify` function calls a Databricks-hosted foundation model directly from SQL. We classify each incident into 6 categories and extract structured fields from the free-text description.
# MAGIC
# MAGIC > **Requires:** Databricks runtime with Mosaic AI Foundation Model APIs (any serverless SQL Warehouse or DBR 14.1+ with serverless compute). If you don't have access, see the fallback cell below.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE workspace.traffic_agent.incidents_enriched AS
# MAGIC SELECT
# MAGIC   incident_id,
# MAGIC   road_inc,
# MAGIC   direction,
# MAGIC   start_time,
# MAGIC   end_time,
# MAGIC   duration_min,
# MAGIC   hour,
# MAGIC   description,
# MAGIC   weather AS weather_raw,
# MAGIC   ai_classify(
# MAGIC     description,
# MAGIC     ARRAY('Collision','Disabled Vehicle','Roadwork','Debris','Weather Event','Other')
# MAGIC   ) AS incident_category,
# MAGIC   ai_extract(
# MAGIC     description,
# MAGIC     ARRAY('road','direction','severity','involves_emergency_vehicles')
# MAGIC   ) AS extracted_fields
# MAGIC FROM workspace.traffic_agent.incidents_raw
# MAGIC WHERE description IS NOT NULL;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS total_rows FROM workspace.traffic_agent.incidents_enriched;
# MAGIC
# MAGIC SELECT incident_category, description
# MAGIC FROM workspace.traffic_agent.incidents_enriched
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT incident_category,
# MAGIC        COUNT(*) AS n,
# MAGIC        ROUND(AVG(duration_min), 1) AS avg_duration_min
# MAGIC FROM workspace.traffic_agent.incidents_enriched
# MAGIC GROUP BY incident_category
# MAGIC ORDER BY n DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fallback if AI Functions are not available in your workspace
# MAGIC Skip this cell if the SQL above ran successfully. Otherwise, this Python fallback uses simple keyword rules so the rest of the notebook still works.

# COMMAND ----------

# Run ONLY if ai_classify is not available in your workspace
RUN_FALLBACK = False

if RUN_FALLBACK:
    @F.udf("string")
    def keyword_category(desc):
        if desc is None: return "Other"
        d = desc.lower()
        if "collision" in d or "crash" in d: return "Collision"
        if "disabled" in d or "stalled" in d: return "Disabled Vehicle"
        if "roadwork" in d or "construction" in d: return "Roadwork"
        if "debris" in d: return "Debris"
        if "weather" in d or "ice" in d or "snow" in d or "fog" in d: return "Weather Event"
        return "Other"

    (spark.table(RAW_TABLE)
        .withColumn("incident_category", keyword_category("description"))
        .write.mode("overwrite").option("overwriteSchema","true")
        .saveAsTable(ENRICHED_TABLE))
    print("Fallback enrichment written.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Compute per-corridor historical priors
# MAGIC A small Delta table the streaming job will join against.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE workspace.traffic_agent.road_priors AS
# MAGIC SELECT
# MAGIC   UPPER(road_inc) AS road,
# MAGIC   COUNT(*) AS incident_count,
# MAGIC   ROUND(AVG(duration_min), 1) AS avg_duration_min,
# MAGIC   MODE(hour) AS busiest_hour,
# MAGIC   collect_set(incident_category) AS observed_categories
# MAGIC FROM workspace.traffic_agent.incidents_enriched
# MAGIC WHERE road_inc IS NOT NULL
# MAGIC GROUP BY UPPER(road_inc)
# MAGIC HAVING COUNT(*) >= 5
# MAGIC ORDER BY incident_count DESC;
# MAGIC
# MAGIC SELECT * FROM workspace.traffic_agent.road_priors LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Stream live weather data into a Delta table
# MAGIC We fetch Open-Meteo every 30s from a small Python loop, append to a JSON folder, and `readStream` that folder into Delta. This is the canonical "file source -> Structured Streaming -> Delta" pattern.

# COMMAND ----------

import threading, time, json, requests, os
from datetime import datetime

MD_CORRIDORS = {
    "I-95 (Baltimore)":   (39.2904, -76.6122),
    "I-695 (Beltway)":    (39.3850, -76.6413),
    "I-70 (Frederick)":   (39.4143, -77.4105),
    "I-270 (Rockville)":  (39.0840, -77.1528),
    "US-50 (Annapolis)":  (38.9784, -76.4922),
    "MD-295 (BW Pkwy)":   (39.1500, -76.7700),
    "MD-100":             (39.1730, -76.7280),
    "MD-75":              (39.4900, -77.2900),
}

LANDING_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/raw/weather_landing"
os.makedirs(LANDING_DIR, exist_ok=True)

def fetch_once():
    rows = []
    for name, (lat, lon) in MD_CORRIDORS.items():
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon,
                        "current": "temperature_2m,precipitation,wind_speed_10m,weather_code",
                        "timezone": "auto"},
                timeout=8,
            )
            cur = r.json()["current"]
            rows.append({
                "ts":         datetime.utcnow().isoformat(),
                "corridor":   name,
                "lat": lat, "lon": lon,
                "temp_c":     cur.get("temperature_2m"),
                "precip_mm":  cur.get("precipitation", 0) or 0,
                "wind_kmh":   cur.get("wind_speed_10m", 0) or 0,
                "weather_code": cur.get("weather_code", 0),
            })
        except Exception as e:
            print(f"  skip {name}: {e}")
    fname = f"{LANDING_DIR}/weather_{int(time.time())}.json"
    with open(fname, "w") as f:
        for row in rows: f.write(json.dumps(row) + "\n")
    return len(rows)

# Producer thread - drops a JSON file every 30s for ~10 minutes
def producer(stop_event, interval=30, max_iter=20):
    for i in range(max_iter):
        if stop_event.is_set(): break
        n = fetch_once()
        print(f"  producer tick {i+1}: wrote {n} rows")
        stop_event.wait(interval)

stop_event = threading.Event()
t = threading.Thread(target=producer, args=(stop_event,), daemon=True)
t.start()
print(f"Weather producer started. Landing dir: {LANDING_DIR}")

# COMMAND ----------

import os
files = sorted(os.listdir(LANDING_DIR))
print(f"{len(files)} weather files landed:")
for f in files[-5:]:
    print(" ", f)

# COMMAND ----------

# Structured Streaming reader - JSON -> Delta
weather_schema = (
    "ts STRING, corridor STRING, lat DOUBLE, lon DOUBLE, "
    "temp_c DOUBLE, precip_mm DOUBLE, wind_kmh DOUBLE, weather_code INT"
)

stream_df = (
    spark.readStream
    .schema(weather_schema)
    .json(LANDING_DIR)
)

checkpoint_path = f"/Volumes/{CATALOG}/{SCHEMA}/raw/_chk_weather"

query = (
    stream_df.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", checkpoint_path)
    .trigger(availableNow=True)
    .toTable("workspace.traffic_agent.weather_live")
)

print("Streaming query started. Stream ID:", query.id)
print("Let it run for ~2-3 minutes to accumulate a few batches, then move on.")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS rows_so_far FROM workspace.traffic_agent.weather_live;
# MAGIC
# MAGIC SELECT * FROM workspace.traffic_agent.weather_live ORDER BY ts DESC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS rows_so_far FROM workspace.traffic_agent.weather_live;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Build the real-time corridor risk view
# MAGIC One materialized view that joins the latest live weather with the historical priors and computes a risk score.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE VIEW workspace.traffic_agent.corridor_risk_view AS
# MAGIC WITH latest_weather AS (
# MAGIC   SELECT corridor, temp_c, precip_mm, wind_kmh, weather_code, ts,
# MAGIC          REGEXP_REPLACE(UPPER(corridor), '[^A-Z0-9]', '') AS corridor_norm,
# MAGIC          ROW_NUMBER() OVER (PARTITION BY corridor ORDER BY ts DESC) AS rn
# MAGIC   FROM workspace.traffic_agent.weather_live
# MAGIC ),
# MAGIC matches AS (
# MAGIC   SELECT w.corridor, w.ts, w.temp_c, w.precip_mm, w.wind_kmh, w.weather_code,
# MAGIC          p.road, p.incident_count, p.avg_duration_min,
# MAGIC          ROW_NUMBER() OVER (
# MAGIC            PARTITION BY w.corridor
# MAGIC            ORDER BY LENGTH(p.road) DESC, p.incident_count DESC
# MAGIC          ) AS match_rank
# MAGIC   FROM latest_weather w
# MAGIC   LEFT JOIN workspace.traffic_agent.road_priors p
# MAGIC     ON w.corridor_norm LIKE CONCAT('%', REGEXP_REPLACE(UPPER(p.road), '[^A-Z0-9]', ''), '%')
# MAGIC   WHERE w.rn = 1
# MAGIC     AND LENGTH(REGEXP_REPLACE(UPPER(p.road), '[^A-Z0-9]', '')) >= 3
# MAGIC )
# MAGIC SELECT
# MAGIC   corridor, road, ts,
# MAGIC   temp_c, precip_mm, wind_kmh, weather_code,
# MAGIC   incident_count, avg_duration_min,
# MAGIC   LEAST(100,
# MAGIC     CAST(LEAST(40, COALESCE(incident_count, 0)/50.0) AS INT)
# MAGIC     + CASE WHEN weather_code IN (95,96,99)         THEN 40
# MAGIC            WHEN weather_code IN (65,82,86,75)      THEN 30
# MAGIC            WHEN weather_code IN (45,48)            THEN 25
# MAGIC            WHEN weather_code IN (63,73,81,85)      THEN 20
# MAGIC            WHEN weather_code IN (51,53,55,61,71,80) THEN 10
# MAGIC            ELSE 0 END
# MAGIC     + CASE WHEN wind_kmh  > 40 THEN 10 ELSE 0 END
# MAGIC     + CASE WHEN precip_mm > 5  THEN 10 ELSE 0 END
# MAGIC     + CASE WHEN temp_c   < 0  THEN 10 ELSE 0 END
# MAGIC   ) AS risk_score
# MAGIC FROM matches
# MAGIC WHERE match_rank = 1;
# MAGIC
# MAGIC SELECT corridor, road, risk_score, weather_code, temp_c, precip_mm, wind_kmh, incident_count
# MAGIC FROM workspace.traffic_agent.corridor_risk_view
# MAGIC ORDER BY risk_score DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT corridor, risk_score, weather_code, temp_c, precip_mm, wind_kmh, incident_count
# MAGIC FROM ${CATALOG}.${SCHEMA}.corridor_risk_view
# MAGIC ORDER BY risk_score DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. AI-generated alerts using `ai_query` (Mosaic AI)
# MAGIC One row per corridor with a natural-language alert written by a Databricks-hosted Claude / Llama endpoint. This is the "AI agent" step done entirely in SQL.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   corridor,
# MAGIC   road,
# MAGIC   risk_score,
# MAGIC   ai_gen(
# MAGIC     CONCAT(
# MAGIC       'You are a Maryland traffic operations analyst. In one sentence, ',
# MAGIC       'explain the risk for ', corridor,
# MAGIC       ' given risk score ', risk_score, '/100, ',
# MAGIC       precip_mm, 'mm precip, ', wind_kmh, 'km/h wind, weather code ', weather_code,
# MAGIC       ', and ', COALESCE(incident_count, 0), ' historical incidents on this corridor.'
# MAGIC     )
# MAGIC   ) AS ai_alert
# MAGIC FROM workspace.traffic_agent.corridor_risk_view
# MAGIC ORDER BY risk_score DESC
# MAGIC LIMIT 8;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Stop the streaming job
# MAGIC Run this when you're done collecting data so the cluster doesn't keep spinning.

# COMMAND ----------

# Stop the producer thread if it's still in memory
try:
    stop_event.set()
    print("Producer thread signaled to stop.")
except NameError:
    print("Producer variable lost (session reset) - that's fine, it'll time out on its own.")

# Stop any active streaming queries
active = spark.streams.active
print(f"Found {len(active)} active streaming query(ies).")
for q in active:
    print(f"  stopping: {q.id}")
    q.stop()
print("All streams stopped.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Next steps (do these in the Databricks UI, not in this notebook)
# MAGIC
# MAGIC **A. Build a Genie space (natural-language Q&A over your data)**
# MAGIC 1. Left sidebar -> **Genie** -> **New**.
# MAGIC 2. Add tables: `incidents_enriched`, `road_priors`, `corridor_risk_view`.
# MAGIC 3. Add a few sample questions: "Which roads had the most collisions?", "What is the current highest-risk corridor?", "How does precipitation correlate with incident duration?".
# MAGIC 4. Share the space and grab the URL for your portfolio.
# MAGIC
# MAGIC **B. Build an AI/BI Dashboard**
# MAGIC 1. Left sidebar -> **Dashboards** -> **Create dashboard**.
# MAGIC 2. Data source: `corridor_risk_view` and `incidents_enriched`.
# MAGIC 3. Add charts: risk score bar chart by corridor, incident category pie, hour-of-day histogram, line chart of live `weather_live.precip_mm` over time.
# MAGIC 4. Publish. The dashboard auto-refreshes against the streaming Delta table.
# MAGIC
# MAGIC **C. Take screenshots** of (1) this notebook's executed cells, (2) the Genie chat, (3) the AI/BI dashboard. Put them in your portfolio / GitHub README.

