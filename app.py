import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import requests
from io import StringIO

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Distribución Global de Sismos",
    page_icon="🌍",
    layout="wide",
)

# ── Style ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; }
    .sub-title  { font-size: 1rem; color: #555; margin-bottom: 1.5rem; }
    .metric-card {
        background: #f8f9fa; border-radius: 12px; padding: 1rem 1.5rem;
        border-left: 4px solid #e63946;
    }
    .stMetric label { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌍 Distribución Global de Sismos</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Fuente: USGS Earthquake Hazards Program · edgarmunoz.co</div>', unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filtros")

    year_range = st.slider(
        "Rango de años",
        min_value=2000,
        max_value=datetime.now().year,
        value=(2015, datetime.now().year),
        step=1,
    )

    min_mag = st.slider(
        "Magnitud mínima",
        min_value=4.0,
        max_value=9.0,
        value=6.0,
        step=0.5,
        help="Sismos por debajo de este valor no se incluyen"
    )

    mag_categories = {
        "Moderado (4–5.9)":   (4.0, 5.9),
        "Fuerte (6–6.9)":     (6.0, 6.9),
        "Mayor (7–7.9)":      (7.0, 7.9),
        "Gran terremoto (8+)":(8.0, 10.0),
    }

    st.markdown("---")
    st.markdown("**Referencia de magnitudes**")
    for k in mag_categories:
        st.markdown(f"- {k}")

    st.markdown("---")
    st.caption("Datos: earthquake.usgs.gov/fdsnws/event/1/")

# ── Data fetch ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_usgs(start_year: int, end_year: int, min_magnitude: float) -> pd.DataFrame:
    frames = []
    progress = st.progress(0, text="Descargando datos USGS…")
    n_years = end_year - start_year + 1

    for i, year in enumerate(range(start_year, end_year + 1)):
        url = (
            f"https://earthquake.usgs.gov/fdsnws/event/1/query.csv"
            f"?format=csv&starttime={year}-01-01&endtime={year}-12-31"
            f"&minmagnitude={min_magnitude}&orderby=time-asc&limit=20000"
        )
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df_year = pd.read_csv(StringIO(r.text))
            df_year["year"] = year
            frames.append(df_year)
        except Exception as e:
            st.warning(f"No se pudo cargar {year}: {e}")

        progress.progress((i + 1) / n_years, text=f"Cargando {year}…")

    progress.empty()

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["month_name"] = df["time"].dt.strftime("%b")
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["depth"]     = pd.to_numeric(df["depth"],     errors="coerce")

    # Categoría de magnitud
    df["categoria"] = pd.cut(
        df["mag"],
        bins=[0, 4.9, 5.9, 6.9, 7.9, 10],
        labels=["< 5", "5–5.9", "6–6.9", "7–7.9", "≥ 8"],
        right=True
    )
    return df

# ── Load ─────────────────────────────────────────────────────────────────────────
with st.spinner("Consultando USGS…"):
    df = fetch_usgs(year_range[0], year_range[1], min_mag)

if df.empty:
    st.error("No se obtuvieron datos. Intenta ampliar el rango o bajar la magnitud mínima.")
    st.stop()

# ── KPIs ─────────────────────────────────────────────────────────────────────────
total = len(df)
años  = year_range[1] - year_range[0] + 1
prom_anual = total / años
max_mag = df["mag"].max()
max_ev  = df.loc[df["mag"].idxmax(), "place"] if "place" in df.columns else "—"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de sismos", f"{total:,}")
col2.metric("Promedio anual",  f"{prom_anual:,.0f}")
col3.metric("Magnitud máxima", f"{max_mag:.1f}")
col4.metric("Lugar del mayor", max_ev[:35] + "…" if len(str(max_ev)) > 35 else max_ev)

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📅 Por año", "🌐 Mapa global", "📊 Distribución", "📈 Tendencias"])

# ── TAB 1: Por año ────────────────────────────────────────────────────────────
with tab1:
    yearly = (
        df.groupby(["year", "categoria"])
        .size()
        .reset_index(name="count")
    )
    yearly_total = df.groupby("year").size().reset_index(name="total")

    col_a, col_b = st.columns([2, 1])

    with col_a:
        fig_bar = px.bar(
            yearly, x="year", y="count", color="categoria",
            title=f"Sismos por año (M ≥ {min_mag})",
            labels={"year": "Año", "count": "Número de sismos", "categoria": "Magnitud"},
            color_discrete_sequence=px.colors.sequential.Reds_r,
            barmode="stack",
        )
        fig_bar.update_layout(
            plot_bgcolor="white",
            xaxis=dict(tickmode="linear", dtick=1),
            legend_title_text="Categoría",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.markdown("**Sismos por año**")
        table = yearly_total.rename(columns={"year": "Año", "total": "Sismos"})
        table["Año"] = table["Año"].astype(str)
        st.dataframe(table, hide_index=True, use_container_width=True)

# ── TAB 2: Mapa global ────────────────────────────────────────────────────────
with tab2:
    st.markdown("Tamaño del círculo proporcional a la magnitud · Color = profundidad (km)")
    df_map = df.dropna(subset=["latitude", "longitude", "mag"])

    fig_map = px.scatter_geo(
        df_map,
        lat="latitude", lon="longitude",
        size="mag",
        color="depth",
        color_continuous_scale="Viridis_r",
        hover_name="place" if "place" in df_map.columns else None,
        hover_data={"mag": True, "depth": True, "time": True, "latitude": False, "longitude": False},
        projection="natural earth",
        title="Epicentros de sismos a nivel mundial",
        size_max=20,
        opacity=0.7,
    )
    fig_map.update_layout(
        geo=dict(showland=True, landcolor="#e8e8e8", showocean=True, oceancolor="#cce5ff"),
        coloraxis_colorbar_title="Profundidad (km)",
        height=550,
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ── TAB 3: Distribución ───────────────────────────────────────────────────────
with tab3:
    col_c, col_d = st.columns(2)

    with col_c:
        fig_hist = px.histogram(
            df, x="mag", nbins=40,
            title="Distribución de magnitudes",
            labels={"mag": "Magnitud", "count": "Frecuencia"},
            color_discrete_sequence=["#e63946"],
        )
        fig_hist.update_layout(plot_bgcolor="white", bargap=0.05)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_d:
        fig_depth = px.histogram(
            df.dropna(subset=["depth"]), x="depth", nbins=50,
            title="Distribución de profundidad",
            labels={"depth": "Profundidad (km)", "count": "Frecuencia"},
            color_discrete_sequence=["#457b9d"],
        )
        fig_depth.update_layout(plot_bgcolor="white", bargap=0.05)
        st.plotly_chart(fig_depth, use_container_width=True)

    # Scatter mag vs depth
    fig_scatter = px.scatter(
        df.dropna(subset=["mag", "depth"]).sample(min(5000, len(df))),
        x="depth", y="mag",
        color="categoria",
        opacity=0.4,
        title="Magnitud vs. Profundidad",
        labels={"depth": "Profundidad (km)", "mag": "Magnitud"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_scatter.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── TAB 4: Tendencias ─────────────────────────────────────────────────────────
with tab4:
    col_e, col_f = st.columns(2)

    with col_e:
        # Promedio de magnitud por año
        mag_year = df.groupby("year")["mag"].agg(["mean", "max", "min"]).reset_index()
        fig_mag = go.Figure()
        fig_mag.add_trace(go.Scatter(x=mag_year["year"], y=mag_year["mean"],
                                     mode="lines+markers", name="Promedio",
                                     line=dict(color="#e63946", width=2)))
        fig_mag.add_trace(go.Scatter(x=mag_year["year"], y=mag_year["max"],
                                     mode="lines", name="Máximo",
                                     line=dict(color="#457b9d", dash="dash")))
        fig_mag.update_layout(
            title="Magnitud promedio y máxima por año",
            xaxis_title="Año", yaxis_title="Magnitud",
            plot_bgcolor="white", legend_title_text="",
            xaxis=dict(tickmode="linear", dtick=1),
        )
        st.plotly_chart(fig_mag, use_container_width=True)

    with col_f:
        # Estacionalidad mensual
        monthly = df.groupby("month").size().reset_index(name="count")
        month_names = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
        monthly["mes"] = monthly["month"].apply(lambda x: month_names[x-1])
        fig_month = px.bar(
            monthly, x="mes", y="count",
            title="Estacionalidad mensual (todos los años)",
            labels={"mes": "Mes", "count": "Número de sismos"},
            color="count",
            color_continuous_scale="Reds",
            category_orders={"mes": month_names},
        )
        fig_month.update_layout(plot_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(fig_month, use_container_width=True)

    # Top 10 sismos más grandes
    st.markdown("### 🔴 Top 10 sismos de mayor magnitud en el período")
    top10 = (
        df.nlargest(10, "mag")[["time", "place", "mag", "depth", "latitude", "longitude"]]
        .reset_index(drop=True)
    )
    top10.index += 1
    top10["time"] = top10["time"].dt.strftime("%Y-%m-%d %H:%M UTC")
    top10.columns = ["Fecha/Hora UTC", "Lugar", "Magnitud", "Profundidad (km)", "Lat", "Lon"]
    st.dataframe(top10, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🌐 edgarmunoz.co · Datos: USGS Earthquake Hazards Program (earthquake.usgs.gov) · Actualización cada hora")
