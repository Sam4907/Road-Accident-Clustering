import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pydeck as pdk
from scipy.spatial import cKDTree

st.set_page_config(
    page_title="Road Accident Clustering & Analytics",
    layout="wide"
)

FILE = "cleaned_accidents.parquet"
MAX_MAP = 40
MAX_POINTS = 5000

# Complete UI Makeover CSS
st.markdown("""
<style>
/* Global light theme styling */
.stApp {
    background: #F8FAFC;
    color: #1E293B;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

/* Custom Tabs Styling (fixes white blending issues) */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #E2E8F0;
    padding: 6px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: 8px;
    color: #475569;
    font-weight: 600;
    background-color: transparent;
    border: none;
    padding: 0 16px;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #2563EB;
    background-color: rgba(255, 255, 255, 0.5);
}
.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #2563EB !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
}
div[data-testid="stMetric"] label {
    color: #64748B !important;
    font-weight: 500;
}

/* Insight Banners */
.insight-blue {
    background: #EFF6FF;
    border-left: 4px solid #2563EB;
    color: #1E40AF;
    padding: 14px;
    margin: 10px 0;
    border-radius: 6px;
    font-weight: 500;
}
.insight-purple {
    background: #FAF5FF;
    border-left: 4px solid #7C3AED;
    color: #5B21B6;
    padding: 14px;
    margin: 10px 0;
    border-radius: 6px;
    font-weight: 500;
}

/* Headers */
h1, h2, h3 {
    color: #0F172A;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("Road Accident Intelligence")
st.caption("County-level accident patterns, temporal trends and hotspot analysis.")


@st.cache_data(show_spinner="Loading prepared accident data...")
def load_data(path, modified):
    cols = [
        "ID", "Severity", "Start_Time", "End_Time",
        "Start_Lat", "Start_Lng", "Street", "County",
        "State", "Duration_Mins"
    ]
    d = pd.read_parquet(path, columns=cols)
    d["Start_Time"] = pd.to_datetime(d["Start_Time"], errors="coerce")
    d = d.dropna(subset=[
        "County", "Street", "Start_Time", "Start_Lat", "Start_Lng"
    ])
    return d


@st.cache_data(show_spinner=False)
def street_summary(d):
    summary = (
        d.groupby("Street", observed=True)
        .agg(
            Incidents=("ID", "size"),
            Lat=("Start_Lat", "mean"),
            Lng=("Start_Lng", "mean")
        )
        .nlargest(MAX_MAP, "Incidents")
        .reset_index()
    )
    
    # Assign color based on number of incidents (Blue -> Purple -> Red gradient)
    if not summary.empty:
        min_inc = summary["Incidents"].min()
        max_inc = summary["Incidents"].max()
        if max_inc == min_inc:
            norm_inc = np.zeros(len(summary))
        else:
            norm_inc = (summary["Incidents"] - min_inc) / (max_inc - min_inc)
            
        def get_color(val):
            if val <= 0.5:
                t = val * 2
                r = int(59 + t * (139 - 59))
                g = int(130 + t * (92 - 130))
                b = int(246 + t * (246 - 246))
            else:
                t = (val - 0.5) * 2
                r = int(139 + t * (239 - 139))
                g = int(92 + t * (68 - 92))
                b = int(246 + t * (68 - 246))
            return [r, g, b, 210]
            
        summary["color"] = [get_color(v) for v in norm_inc]
    else:
        summary["color"] = []
        
    return summary


@st.cache_data(show_spinner=False)
def hour_summary(d):
    return (
        d.assign(Hour=d["Start_Time"].dt.hour)
        .groupby("Hour").size()
        .reindex(range(24), fill_value=0)
        .rename("Incidents").reset_index()
    )


@st.cache_data(show_spinner=False)
def severity_summary(d):
    return (
        d.groupby("Severity", observed=True).size()
        .rename("Incidents").reset_index()
    )


@st.cache_data(show_spinner=False)
def spatial_analysis(coords):
    lat = coords["Start_Lat"].to_numpy(dtype=float)
    lon = coords["Start_Lng"].to_numpy(dtype=float)
    total = len(lat)

    if total < 10:
        return None, total, 0.0, False

    mean_lat = lat.mean()
    all_x = lon * 69.0 * np.cos(np.radians(mean_lat))
    all_y = lat * 69.0

    xmin, xmax = all_x.min(), all_x.max()
    ymin, ymax = all_y.min(), all_y.max()
    area = max((xmax - xmin) * (ymax - ymin), 0.1)

    sampled = total > MAX_POINTS
    if sampled:
        rng = np.random.default_rng(42)
        idx = rng.choice(total, MAX_POINTS, replace=False)
        x, y = all_x[idx], all_y[idx]
    else:
        x, y = all_x, all_y

    points = np.column_stack([x, y])
    n = len(points)
    edge = np.minimum.reduce([
        x - xmin, xmax - x, y - ymin, ymax - y
    ])
    tree = cKDTree(points)
    radii = np.linspace(0.1, 2.0, 20)
    kvals = []

    for r in radii:
        inner = edge >= r
        ni = int(inner.sum())
        if ni == 0:
            kvals.append(np.nan)
            continue
        neighbors = tree.query_ball_point(
            points[inner], r, return_length=True
        ) - 1
        kvals.append(area * neighbors.sum() / (n * ni))

    kvals = np.asarray(kvals)
    results = pd.DataFrame({
        "Radius (miles)": radii,
        "Observed K": kvals,
        "CSR K": np.pi * radii**2,
        "Besag L": np.sqrt(np.maximum(kvals, 0) / np.pi) - radii
    })
    return results, total, area, sampled


if not os.path.exists(FILE):
    st.error("cleaned_accidents.parquet not found. Restart the setup cell.")
    st.stop()

df = load_data(FILE, os.path.getmtime(FILE))

if df.empty:
    st.error("No usable records found.")
    st.stop()

st.sidebar.header("Control Center")
st.sidebar.success(f"Records loaded")

counties = sorted(df["County"].dropna().unique().tolist())
if not counties:
    st.error("No counties found in the dataset.")
    st.stop()

default = "Los Angeles" if "Los Angeles" in counties else counties[0]
county = st.sidebar.selectbox(
    "Select County", counties, index=counties.index(default)
)

d = df.loc[df["County"] == county]
if d.empty:
    st.warning("No records for this county.")
    st.stop()

t1, t2, t3, t4 = st.tabs([
    "Overview & Hotspots", 
    "Time Patterns",
    "Advanced Spatial Analysis", 
    "Raw Data"
])

with t1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Recorded Incidents", f"{len(d):,}")
    col2.metric("Median Duration", f"{d['Duration_Mins'].median():.0f} min")
    col3.metric("Mean Severity", f"{d['Severity'].mean():.2f}")

    st.markdown("### Key Discoveries")
    hours = d["Start_Time"].dt.hour.value_counts()
    if not hours.empty:
        peak = int(hours.idxmax())
        label = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=peak)
        st.markdown(
            f'<div class="insight-blue">Busiest start hour: <b>'
            f'{label.strftime("%I %p").lstrip("0")}</b> '
            f'({int(hours.max()):,} incidents).</div>',
            unsafe_allow_html=True
        )

    common = d["Severity"].mode()
    if not common.empty:
        st.markdown(
            f'<div class="insight-purple">Most common severity level: '
            f'<b>{int(common.iloc[0])}</b>.</div>',
            unsafe_allow_html=True
        )

    st.markdown("### Accident Hotspots by Street")
    st.caption("Dots scale in size and color based on incident frequency (Blue → Purple → Red).")
    
    streets = street_summary(d)
    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(
            latitude=float(d["Start_Lat"].mean()),
            longitude=float(d["Start_Lng"].mean()),
            zoom=9
        ),
        layers=[pdk.Layer(
            "ScatterplotLayer",
            data=streets,
            get_position="[Lng, Lat]",
            get_fill_color="color",
            get_radius="Incidents",
            radius_scale=8,
            radius_min_pixels=5,
            radius_max_pixels=20,
            pickable=True
        )],
        tooltip={"html": "<b>Street:</b> {Street}<br/>Incidents: {Incidents}"}
    )
    st.pydeck_chart(deck, width="stretch")
    st.dataframe(streets.drop(columns=["color"]), width="stretch", hide_index=True)

with t2:
    st.markdown("### Incidents by Hour of Day")
    h = hour_summary(d)
    fig = px.bar(
        h, x="Hour", y="Incidents", template="plotly_white",
        color="Incidents", color_continuous_scale=["#3b82f6", "#8b5cf6", "#ef4444"]
    )
    fig.update_layout(xaxis=dict(dtick=1), coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Incidents by Severity Level")
    s = severity_summary(d)
    fig2 = px.bar(
        s, x="Severity", y="Incidents", template="plotly_white",
        color="Incidents", color_continuous_scale=["#3b82f6", "#8b5cf6", "#ef4444"]
    )
    fig2.update_layout(coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, width="stretch")

with t3:
    st.markdown(f"### Spatial Clustering Analysis: {county} County")
    st.write(
        "Spatial estimates use up to 5,000 sampled points for large "
        "counties. Sampling makes results approximate."
    )

    if st.button("Run Spatial Analysis", type="primary"):
        with st.spinner("Computing spatial statistics..."):
            result, n, area, sampled = spatial_analysis(
                d[["Start_Lat", "Start_Lng"]]
            )
        st.session_state["spatial"] = {
            "county": county, "result": result,
            "n": n, "area": area, "sampled": sampled
        }

    saved = st.session_state.get("spatial")
    if saved and saved["county"] == county:
        result = saved["result"]
        if result is None:
            st.warning("At least 10 incidents are needed.")
        else:
            colA, colB = st.columns(2)
            colA.metric("Total Incidents Analyzed", f"{saved['n']:,}")
            colB.metric("Bounding Rectangle Area", f"{saved['area']:,.1f} sq mi")

            if saved["sampled"]:
                st.info("Results are based on a fixed random sample of 5,000 points.")

            st.markdown("#### Ripley's K: Observed vs. CSR")
            long_k = result.melt(
                id_vars=["Radius (miles)"],
                value_vars=["Observed K", "CSR K"],
                var_name="Function", value_name="K"
            )
            fig_k = px.line(long_k, x="Radius (miles)", y="K",
                            color="Function", markers=True,
                            template="plotly_white",
                            color_discrete_sequence=["#2563EB", "#EF4444"])
            fig_k.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_k, width="stretch")

            st.markdown("#### Besag's L Function")
            fl = px.line(
                result, x="Radius (miles)", y="Besag L",
                markers=True, template="plotly_white",
                color_discrete_sequence=["#7C3AED"]
            )
            fl.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
            fl.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fl, width="stretch")
    elif saved:
        st.info("Click 'Run Spatial Analysis' for the selected county above.")

with t4:
    st.markdown(f"### Raw Records for {county}")
    st.caption(f"Showing preview of up to 1,000 records.")
    st.dataframe(d.head(1000), width="stretch", hide_index=True)

    if st.button("Prepare CSV download (first 100,000 rows)"):
        csv = d.head(100_000).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name=f"{county.replace(' ', '_')}_accidents.csv",
            mime="text/csv"
        )
