# =========================
# app.py (Streamlit + Spark + Folium)
# =========================

import streamlit as st
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, desc


# -------------------------
# 1) Streamlit Page Config
# -------------------------
st.set_page_config(page_title="LA Crime + Income Spatial App", layout="wide")
st.title("LA Crime + Income Query App")


# -------------------------
# 2) Spark Session (cache_resource)
# -------------------------
@st.cache_resource
def get_spark():
    spark = (
        SparkSession.builder
        .appName("CS226 LA Crime Spatial App")
        .config("spark.sql.shuffle.partitions", "80")
        # If you hit heap OOM later, uncomment and tune:
        # .config("spark.driver.memory", "6g")
        # .config("spark.executor.memory", "6g")
        .getOrCreate()
    )
    return spark


spark = get_spark()


# -------------------------
# 3) Load Data (cache_resource)
#    - returns Spark DF
#    - do light cleaning here
# -------------------------
@st.cache_resource
def load_df(parquet_path: str):
    df = spark.read.parquet(parquet_path)

    # Create numeric income if needed
    if "med_hh_income" in df.columns and "med_hh_income_num" not in df.columns:
        df = df.withColumn(
            "med_hh_income_num",
            F.regexp_replace(F.regexp_replace(col("med_hh_income"), "[$,]", ""), " ", "").cast("double"),
        )

    # Create LAT_d / LON_d and geo_valid
    if "LAT" in df.columns and "LON" in df.columns:
        df = (
            df.withColumn("LAT_d", col("LAT").cast("double"))
              .withColumn("LON_d", col("LON").cast("double"))
              .withColumn(
                  "geo_valid",
                  F.when(
                      col("LAT_d").between(33.5, 34.5) & col("LON_d").between(-119.0, -117.0),
                      1,
                  ).otherwise(0),
              )
        )

    return df


# -------------------------
# 4) Distinct values helper (cache_data)
# IMPORTANT:
#   Do NOT pass Spark DF into cache_data (unhashable).
#   Cache by (parquet_path, colname) instead.
# -------------------------
@st.cache_data
def distinct_values(parquet_path: str, colname: str, limit: int = 5000):
    d = (
        spark.read.parquet(parquet_path)
        .select(colname)
        .where(col(colname).isNotNull())
        .distinct()
        .limit(limit)
    )
    vals = d.collect()
    return sorted([r[colname] for r in vals])


# -------------------------
# 5) Sidebar: dataset path + filters
# -------------------------
default_path = "CS226_project/LA_Crime_Income_Merged_Final_parquet"
parquet_path = st.sidebar.text_input("Parquet folder path", value=default_path)

try:
    df = load_df(parquet_path)
except Exception as e:
    st.error(f"Could not load parquet from:\n{parquet_path}\n\nError:\n{e}")
    st.stop()

with st.sidebar:
    st.header("Filters")

    spa_vals = distinct_values(parquet_path, "spa") if "spa" in df.columns else []
    year_vals = distinct_values(parquet_path, "Occured_Year") if "Occured_Year" in df.columns else []
    month_vals = distinct_values(parquet_path, "Occured_Month") if "Occured_Month" in df.columns else []

    spa_selected = st.multiselect("SPA", spa_vals, default=[])
    year_selected = st.multiselect("Year", year_vals, default=year_vals[-3:] if len(year_vals) >= 3 else year_vals)
    month_selected = st.multiselect("Month", month_vals, default=month_vals)

    st.divider()
    st.caption("Leave a filter empty to include all values.")


def apply_filters(df_in):
    d = df_in
    if spa_selected:
        d = d.filter(col("spa").isin(spa_selected))
    if year_selected:
        d = d.filter(col("Occured_Year").isin(year_selected))
    if month_selected:
        d = d.filter(col("Occured_Month").isin(month_selected))
    return d


df_f = apply_filters(df)

# Avoid df_f.count() on every rerun (Spark job). Compute on-demand instead.
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Columns", len(df.columns))
with c2:
    st.metric("SPAs selected", len(spa_selected) if spa_selected else len(spa_vals))
with c3:
    if st.button("Compute filtered row count"):
        st.metric("Filtered rows", int(df_f.count()))
    else:
        st.metric("Filtered rows", "click button")


# -------------------------
# 6) Haversine distance (Spark Column)
# -------------------------
def haversine_km(lat0, lon0, lat_col, lon_col):
    r = 6371.0
    return 2 * r * F.asin(
        F.sqrt(
            F.pow(F.sin((F.radians(lat_col) - F.radians(F.lit(lat0))) / 2), 2)
            + F.cos(F.radians(F.lit(lat0)))
            * F.cos(F.radians(lat_col))
            * F.pow(F.sin((F.radians(lon_col) - F.radians(F.lit(lon0))) / 2), 2)
        )
    )


# -------------------------
# 7) Tabs
# -------------------------
tab_map, tab_normal = st.tabs(["Spatial Queries (Folium)", "Normal Queries (Spark)"])


# ==========================================================
# TAB A: Spatial Queries
# ==========================================================
with tab_map:
    st.subheader("Spatial Queries")

    if "geo_valid" not in df.columns:
        st.error("LAT/LON columns not found or geo_valid not created. Check your schema.")
        st.stop()

    df_geo = df_f.filter(col("geo_valid") == 1)

    mode = st.selectbox(
        "Choose spatial query",
        ["Hotspots HeatMap (grid)", "Bounding Box Query", "Radius Query (km)"],
    )

    tiles = st.selectbox("Map tiles", ["CartoDB positron", "OpenStreetMap"])
    m = folium.Map(location=[34.05, -118.25], zoom_start=10, tiles=tiles)

    # ---------
    # A1) Hotspots HeatMap (grid)
    # ---------
    if mode == "Hotspots HeatMap (grid)":
        colA, colB, colC = st.columns(3)
        with colA:
            grid_scale = st.selectbox("Grid scale (higher=finer)", [200, 500, 1000], index=1)
        with colB:
            K = st.slider("Top-K grid cells", 500, 20000, 8000, step=500)
        with colC:
            use_log = st.checkbox("Use log(1+count) weights", value=True)

        df_grid = (
            df_geo.withColumn("lat_grid", F.floor(col("LAT_d") * grid_scale) / grid_scale)
                  .withColumn("lon_grid", F.floor(col("LON_d") * grid_scale) / grid_scale)
        )

        hotspots = (
            df_grid.groupBy("lat_grid", "lon_grid")
            .count()
            .orderBy(desc("count"))
            .limit(K)
        )

        pdf = hotspots.toPandas()
        if pdf.empty:
            st.warning("No points for current filters.")
        else:
            pdf["w"] = np.log1p(pdf["count"]) if use_log else pdf["count"].astype(float)
            heat_data = pdf[["lat_grid", "lon_grid", "w"]].values.tolist()

            HeatMap(heat_data, radius=14, blur=18, max_zoom=12).add_to(m)

            st_folium(m, width=1100, height=650)

            with st.expander("Show hotspot table (Top cells)"):
                st.dataframe(pdf.sort_values("count", ascending=False), use_container_width=True)

    # ---------
    # A2) Bounding Box Query
    # ---------
    elif mode == "Bounding Box Query":
        st.write("Filter crimes inside a latitude/longitude rectangle and map sampled points.")

        c1, c2 = st.columns(2)
        with c1:
            lat_min = st.number_input("lat_min", value=34.00, format="%.6f")
            lat_max = st.number_input("lat_max", value=34.08, format="%.6f")
        with c2:
            lon_min = st.number_input("lon_min", value=-118.30, format="%.6f")
            lon_max = st.number_input("lon_max", value=-118.20, format="%.6f")

        sample_n = st.slider("Max points to plot", 1000, 50000, 15000, step=1000)

        df_box = df_geo.filter(
            col("LAT_d").between(lat_min, lat_max) & col("LON_d").between(lon_min, lon_max)
        )

        # Draw rectangle
        folium.Rectangle([[lat_min, lon_min], [lat_max, lon_max]], color="blue", fill=False).add_to(m)

        # Pull only a capped number of points to browser
        cols = ["LAT_d", "LON_d"]
        if "Crm_Cd_Desc" in df.columns:
            cols.append("Crm_Cd_Desc")

        pdf_pts = df_box.select(*cols).limit(sample_n).toPandas()

        for _, r in pdf_pts.iterrows():
            popup = str(r.get("Crm_Cd_Desc", ""))[:80]
            folium.CircleMarker(
                location=[r["LAT_d"], r["LON_d"]],
                radius=2,
                fill=True,
                fill_opacity=0.5,
                opacity=0.2,
                popup=popup,
            ).add_to(m)

        st_folium(m, width=1100, height=650)

        if "Crm_Cd_Desc" in df.columns:
            topN = st.slider("Top N crime types (box)", 5, 30, 10)
            top_crimes = (
                df_box.groupBy("Crm_Cd_Desc")
                .count()
                .orderBy(desc("count"))
                .limit(topN)
                .toPandas()
            )
            st.subheader("Top crimes inside bounding box")
            st.dataframe(top_crimes, use_container_width=True)

    # ---------
    # A3) Radius Query (km)
    # ---------
    else:
        st.write("Crimes within **R km** of a chosen center point (plots sampled points).")

        c1, c2, c3 = st.columns(3)
        with c1:
            center_lat = st.number_input("Center latitude", value=34.052200, format="%.6f")
        with c2:
            center_lon = st.number_input("Center longitude", value=-118.243700, format="%.6f")
        with c3:
            R = st.slider("Radius (km)", 0.5, 10.0, 2.0, step=0.5)

        sample_n = st.slider("Max points to plot (radius)", 1000, 50000, 15000, step=1000)

        df_radius = df_geo.withColumn("dist_km", haversine_km(center_lat, center_lon, col("LAT_d"), col("LON_d")))
        df_near = df_radius.filter(col("dist_km") <= float(R))

        folium.Marker([center_lat, center_lon], popup="Center").add_to(m)
        folium.Circle([center_lat, center_lon], radius=float(R) * 1000, color="red", fill=False).add_to(m)

        cols = ["LAT_d", "LON_d", "dist_km"]
        if "Crm_Cd_Desc" in df.columns:
            cols.append("Crm_Cd_Desc")

        pdf_pts = (
            df_near.select(*cols)
            .orderBy(col("dist_km").asc())
            .limit(sample_n)
            .toPandas()
        )

        for _, r in pdf_pts.iterrows():
            popup = f"{str(r.get('Crm_Cd_Desc',''))[:60]} | {r['dist_km']:.2f} km"
            folium.CircleMarker(
                location=[r["LAT_d"], r["LON_d"]],
                radius=2,
                fill=True,
                fill_opacity=0.5,
                opacity=0.2,
                popup=popup,
            ).add_to(m)

        st_folium(m, width=1100, height=650)

        if "Crm_Cd_Desc" in df.columns:
            topN = st.slider("Top N crime types (radius)", 5, 30, 10)
            top_near = (
                df_near.groupBy("Crm_Cd_Desc")
                .count()
                .orderBy(desc("count"))
                .limit(topN)
                .toPandas()
            )
            st.subheader(f"Top crimes within {R} km")
            st.dataframe(top_near, use_container_width=True)


# ==========================================================
# TAB B: Normal Queries
# ==========================================================
with tab_normal:
    st.subheader("Normal Queries (Spark Aggregations)")

    q = st.selectbox(
        "Choose query",
        [
            "Top crime descriptions",
            "Crime count by SPA",
            "Weapon vs non-weapon crimes",
            "Crime by victim descent",
            "Average victim age by crime type",
            "Crime vs income by SPA",
        ],
    )

    topN = st.slider("Top N", 5, 50, 10)

    if q == "Top crime descriptions":
        if "Crm_Cd_Desc" not in df.columns:
            st.warning("Crm_Cd_Desc not found.")
        else:
            out = (
                df_f.groupBy("Crm_Cd_Desc")
                .count()
                .orderBy(desc("count"))
                .limit(topN)
                .toPandas()
            )
            st.dataframe(out, use_container_width=True)

    elif q == "Crime count by SPA":
        if "spa" not in df.columns:
            st.warning("spa not found.")
        else:
            out = df_f.groupBy("spa").count().orderBy(desc("count")).toPandas()
            st.dataframe(out, use_container_width=True)

    elif q == "Weapon vs non-weapon crimes":
        if "Weapon_Used_Cd" not in df.columns:
            st.warning("Weapon_Used_Cd not found.")
        else:
            df_weapon = df_f.withColumn("Has_Weapon", F.when(col("Weapon_Used_Cd").isNotNull(), "Yes").otherwise("No"))
            out = df_weapon.groupBy("Has_Weapon").count().orderBy(desc("count")).toPandas()
            st.dataframe(out, use_container_width=True)

            if "spa" in df.columns:
                out2 = (
                    df_weapon.groupBy("spa", "Has_Weapon")
                    .count()
                    .orderBy(desc("count"))
                    .limit(200)
                    .toPandas()
                )
                st.caption("Weapon split by SPA (top rows)")
                st.dataframe(out2, use_container_width=True)

    elif q == "Crime by victim descent":
        if "Vict_Descent" not in df.columns:
            st.warning("Vict_Descent not found.")
        else:
            out = df_f.groupBy("Vict_Descent").count().orderBy(desc("count")).toPandas()
            st.dataframe(out, use_container_width=True)

    elif q == "Average victim age by crime type":
        if "Vict_Age" not in df.columns or "Crm_Cd_Desc" not in df.columns:
            st.warning("Vict_Age or Crm_Cd_Desc not found.")
        else:
            out = (
                df_f.groupBy("Crm_Cd_Desc")
                .agg(F.avg("Vict_Age").alias("avg_age"))
                .orderBy(desc("avg_age"))
                .limit(topN)
                .toPandas()
            )
            st.dataframe(out, use_container_width=True)

    else:  # Crime vs income by SPA
        if "spa" not in df.columns or "med_hh_income_num" not in df.columns:
            st.warning("Need spa and med_hh_income_num (derived from med_hh_income).")
        else:
            out = (
                df_f.groupBy("spa")
                .agg(F.count("*").alias("crime_count"), F.avg("med_hh_income_num").alias("avg_income"))
                .orderBy(desc("crime_count"))
                .toPandas()
            )
            st.dataframe(out, use_container_width=True)