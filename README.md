# Road Accident Intelligence

### **UNIT – II: Getting Started with Pandas & File Formats**

* **Data Loading, Storage & File Formats:** Optimized memory-efficient CSV reading with `pd.read_csv(engine="pyarrow", usecols=..., dtype=...)` and persisted the preprocessed dataset in columnar format using `to_parquet(compression="snappy")` and `pd.read_parquet()`.
* **Series, DataFrame & Index Manipulation:** Realigned time-series bins with `.reindex(range(24), fill_value=0)` and reset row indices with `.reset_index()`.
* **Indexing, Selection & Filtering:** Applied geospatial bounding-box filtering (`df["Start_Lat"].between(24, 50)`) and dynamic user-selected filtering (`df.loc[df["County"] == county]`).
* **Descriptive Statistics:** Extracted key central tendencies across incident severity and duration using `.mean()`, `.median()`, `.mode()`, `.min()`, and `.max()`.

---

### **UNIT – III: Data Cleaning, Preparation & Wrangling**

* **Filtering Missing Data:** Handled nulls with `df.dropna(subset=['Start_Lat', 'Start_Lng', 'Start_Time', 'End_Time', 'Street', 'County'])`.
* **Transforming Data & Outlier Removal:** Filtered duration anomalies with `df["Duration_Mins"].between(0, 1440, inclusive="neither")` and normalized street name strings using `.str.strip().str.title()`.
* **Categorical Extension Type:** Reduced memory overhead on repetitive high-cardinality text columns (`Street`, `County`, `State`) by converting them to `category` dtypes.
* **Discretization & Mapping:** Normalized continuous incident densities to compute dynamic RGB color gradient arrays (`[r, g, b, 210]`) across low-, medium-, and high-risk hotspots.

---

### **UNIT – IV: Data Aggregation, Group Operations & Time Series**

* **Data Aggregation & Group Operations:** Grouped records by corridor using `d.groupby("Street", observed=True).agg(...)` to extract incident counts and cluster centroids (`Lat=("Start_Lat", "mean")`, `Lng=("Start_Lng", "mean")`), ranked with `.nlargest()`.
* **Date and Time Data Types & Tools:** Converted timestamps using `pd.to_datetime(errors="coerce")`.
* **Time Series Basics & Duration Extraction:** Extracted diurnal accident peaks via `.dt.hour` and computed crash clearing durations by taking timestamp deltas (`(End_Time - Start_Time).dt.total_seconds() / 60`).

---

### **UNIT – V: Plotting and Visualization**

* **Visualisation:** Used `matplotlib.pyplot` and `seaborn` to extract useful visualisations such as Accidents by Hour and Top 10 Streets by Incident Count.
* **Geospatial Hotspot Mapping:** Integrated PyDeck (`pdk.Deck`, `pdk.ViewState`, `pdk.Layer("ScatterplotLayer")`) over custom Carto vector basemaps to dynamically scale pin radius and color gradients based on incident density.

---

### **Major Components Beyond the Core Syllabus**

* **Spatial Point Pattern Analysis (Ripley’s $K$ & Besag’s $L$ Functions):**
* * **Interactive Charting:** Used Plotly Express (`plotly.express`) to render interactive temporal line charts, incident severity bar charts, and Ripley's $K$ function plots.
* Rather than standard distance metrics, the project evaluates spatial point processes against theoretical **Complete Spatial Randomness (CSR)** ($K_{\text{CSR}}(r) = \pi r^2$) across radius ranges from 0.1 to 2.0 miles.
* Computed **Besag’s $L$-function** ($L(r) = \sqrt{\frac{K(r)}{\pi}} - r$) to detect whether accidents cluster significantly above random Poisson distributions.
* Implemented **border/edge effect correction** using `np.minimum.reduce()` to eliminate bias at the boundaries of geographic regions.


* **High-Performance Spatial Indexing (`scipy.spatial.cKDTree`):**
* Used a k-d tree data structure to execute radius neighborhood searches (`tree.query_ball_point(...)`) in O(k \log N)$ time instead of an exhaustive O(N^2) pairwise distance matrix.


* **Interactive Web Application & State Architecture (Streamlit):**
* Built a multi-tab web dashboard (`st.tabs`) featuring dynamic metrics (`st.metric`), county selectboxes (`st.selectbox`), and sidebar controls (`st.sidebar`).
* Used Streamlit caching decorators (`@st.cache_data`) with timestamp monitoring (`os.path.getmtime`) to prevent redundant parquet reloads and calculations on rerun.


* **Automated Data Pipeline & Dataset Retrieval:**
* Integrated the `kagglehub` API to automate fetching the multi-gigabyte US Accidents dataset directly from Kaggle.
* Executed headless environment bootstrapping via `subprocess.run([sys.executable, "-m", "pip", "install", ...])` within the notebook execution flow.

Devs- Shrika reballi, Sameeha Yasmin , Sreeya Rao
