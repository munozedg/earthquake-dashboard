import pickle
import sqlite3
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

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

CACHE_DB_PATH = Path(__file__).resolve().parent / "earthquake_cache.sqlite"
CACHE_TTL_SECONDS = 15 * 60


def init_cache_db() -> None:
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS earthquake_cache (
            request_key TEXT PRIMARY KEY,
            start_year INTEGER NOT NULL,
            end_year INTEGER NOT NULL,
            min_magnitude REAL NOT NULL,
            data_payload BLOB NOT NULL,
            last_event_time TEXT,
            last_checked_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["month_name"] = df["time"].dt.strftime("%b")
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df["categoria"] = pd.cut(
        df["mag"],
        bins=[0, 4.9, 5.9, 6.9, 7.9, 10],
        labels=["< 5", "5–5.9", "6–6.9", "7–7.9", "≥ 8"],
        right=True,
    )
    return df


def normalize_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_cache_entry(conn, request_key: str, start_year: int, end_year: int, min_magnitude: float, df: pd.DataFrame, now: datetime) -> None:
    payload = pickle.dumps(df)
    latest_event = df["time"].dropna().max() if "time" in df.columns and not df.empty else None
    last_event_time = normalize_timestamp(latest_event)
    updated_at = normalize_timestamp(now)
    last_checked_at = updated_at
    conn.execute(
        """
        INSERT OR REPLACE INTO earthquake_cache (
            request_key, start_year, end_year, min_magnitude, data_payload,
            last_event_time, last_checked_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            request_key,
            start_year,
            end_year,
            min_magnitude,
            sqlite3.Binary(payload),
            last_event_time,
            last_checked_at,
            updated_at,
        ),
    )
    conn.commit()


def touch_cache_entry(conn, request_key: str, now: datetime) -> None:
    updated_at = normalize_timestamp(now)
    conn.execute(
        "UPDATE earthquake_cache SET last_checked_at = ? , updated_at = ? WHERE request_key = ?",
        (updated_at, updated_at, request_key),
    )
    conn.commit()


def fetch_usgs_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def fetch_full_dataset(start_year: int, end_year: int, min_magnitude: float) -> pd.DataFrame:
    frames = []
    progress = st.progress(0, text="Descargando datos USGS…")
    n_years = end_year - start_year + 1

    for i, year in enumerate(range(start_year, end_year + 1)):
        url = (
            f"https://earthquake.usgs.gov/fdsnws/event/1/query.csv"
            f"?format=csv&starttime={year}-01-01&endtime={year}-12-31"
            f"&minmagnitude={min_magnitude}&orderby=time-asc&limit=200000"
        )
        try:
            df_year = fetch_usgs_csv(url)
            if not df_year.empty:
                df_year["year"] = year
                frames.append(df_year)
        except Exception as e:
            st.warning(f"No se pudo cargar {year}: {e}")

        progress.progress((i + 1) / n_years, text=f"Cargando {year}…")

    progress.empty()

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    return prepare_dataframe(df)


def fetch_incremental_dataset(start_year: int, end_year: int, min_magnitude: float, since: str) -> pd.DataFrame:
    start_time = since
    end_time = datetime(end_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"
        f"?format=csv&starttime={start_time}&endtime={end_time}"
        f"&minmagnitude={min_magnitude}&orderby=time-asc&limit=200000"
    )
    try:
        df = fetch_usgs_csv(url)
    except Exception as e:
        st.warning(f"No se pudieron sincronizar los nuevos datos: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    return prepare_dataframe(df)


def fetch_usgs(start_year: int, end_year: int, min_magnitude: float) -> pd.DataFrame:
    init_cache_db()
    request_key = f"{start_year}:{end_year}:{min_magnitude:.1f}"
    now = datetime.now(timezone.utc)

    conn = sqlite3.connect(CACHE_DB_PATH)
    row = conn.execute(
        "SELECT data_payload, last_event_time, last_checked_at, updated_at FROM earthquake_cache WHERE request_key = ?",
        (request_key,),
    ).fetchone()

    if row is not None:
        cached_df = pickle.loads(row[0])
        last_checked_at = datetime.fromisoformat(row[2].replace("Z", "+00:00"))
        if (now - last_checked_at).total_seconds() < CACHE_TTL_SECONDS:
            conn.close()
            return cached_df

        new_events = pd.DataFrame()
        if row[1]:
            new_events = fetch_incremental_dataset(start_year, end_year, min_magnitude, row[1])

        if not new_events.empty:
            merged_df = pd.concat([cached_df, new_events], ignore_index=True)
            merged_df = merged_df.drop_duplicates(subset=["id"], keep="last")
            merged_df = prepare_dataframe(merged_df)
            save_cache_entry(conn, request_key, start_year, end_year, min_magnitude, merged_df, now)
            conn.close()
            return merged_df

        touch_cache_entry(conn, request_key, now)
        conn.close()
        return cached_df

    df = fetch_full_dataset(start_year, end_year, min_magnitude)
    if not df.empty:
        save_cache_entry(conn, request_key, start_year, end_year, min_magnitude, df, now)
    conn.close()
    return df


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
# ── Load ─────────────────────────────────────────────────────────────────────────
with st.spinner("Consultando USGS…"):
    df = fetch_usgs(year_range[0], year_range[1], min_mag)

if df.empty:
    st.error("No se obtuvieron datos. Intenta ampliar el rango o bajar la magnitud mínima.")
    st.stop()

st.caption("Cache local en earthquake_cache.sqlite · se actualiza automáticamente cuando USGS reporta nuevos sismos y se refresca cada 15 minutos.")

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
        month_names = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
        # Garantizar los 12 meses aunque alguno tenga 0 sismos
        monthly_counts = df.groupby("month").size()
        monthly = (
            pd.Series(0, index=range(1, 13))
            .add(monthly_counts, fill_value=0)
            .reset_index()
        )
        monthly.columns = ["month", "count"]
        monthly["mes"] = monthly["month"].apply(lambda x: month_names[x-1])
        # Ordenar explícitamente por número de mes (no alfabético)
        monthly = monthly.sort_values("month").reset_index(drop=True)
        fig_month = px.bar(
            monthly, x="mes", y="count",
            title="Estacionalidad mensual (todos los años)",
            labels={"mes": "Mes", "count": "Número de sismos"},
            category_orders={"mes": month_names},
        )
        fig_month.update_traces(marker_color="#e63946")
        fig_month.update_layout(plot_bgcolor="white")
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
