import io
import os
import re
import json
import ast
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import requests
import streamlit as st


# ——— Layout amplo e ajustes de espaçamento
st.set_page_config(page_title="MODELO RNA — Clusters SP", page_icon="🧠", layout="wide")
st.markdown("""
<style>
/* aumenta a largura útil e reduz paddings */
.block-container {max-width: 95vw; padding-top: .5rem; padding-bottom: .75rem;}
/* sidebar um pouco mais larga para controles */
[data-testid="stSidebar"] {min-width: 340px;}
</style>
""", unsafe_allow_html=True)


# ==========================
# GITHUB I/O HELPERS
# ==========================
API_BASE = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"


def _secret(path, default=None):
    cur = st.secrets
    try:
        for p in path:
            cur = cur[p]
        return cur
    except Exception:
        return default


def _gh_headers():
    token = _secret(["github", "token"], None)
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def download_df(df: pd.DataFrame, base_name: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Baixar CSV", csv, file_name=f"{base_name}.csv", mime="text/csv")

def download_plotly_png(fig, base_name: str, width_px: int = 2400, height_px: int = 1400):
    """Gera PNG em alta resolução (requer 'kaleido'). 2400x1400 ~ 8x4.7" @300dpi."""
    try:
        import plotly.io as pio
        png = pio.to_image(fig, format="png", width=width_px, height=height_px, scale=1)
        st.download_button("🖼️ Baixar PNG 300 DPI", png, file_name=f"{base_name}.png", mime="image/png")
    except Exception:
        st.info("Para exportar gráficos em PNG 300 DPI instale a dependência 'kaleido'.")
        html = fig.to_html(include_plotlyjs="cdn")
        st.download_button("💾 Baixar HTML interativo", html, file_name=f"{base_name}.html", mime="text/html")

def normalize_repo(owner_repo: str) -> str:
    s = (owner_repo or "").strip()
    s = s.replace("https://github.com/", "").replace("http://github.com/", "")
    s = s.strip("/")
    parts = [p for p in s.split("/") if p]
    if len(parts) < 2:
        raise RuntimeError(
            "Informe o repositório no formato 'owner/repo'. Ex.: 'emiliobneto/UrbanTechCluster'"
        )
    return f"{parts[0]}/{parts[1]}"


@st.cache_data(show_spinner=True)
def github_repo_info(owner_repo: str):
    owner_repo = normalize_repo(owner_repo)
    url = f"{API_BASE}/repos/{owner_repo}"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Falha lendo repo {owner_repo}: {r.status_code} {r.text}")
    return r.json()

def resolve_branch(owner_repo: str, user_branch: str | None):
    owner_repo = normalize_repo(owner_repo)
    b = (user_branch or "").strip()
    if b:
        url = f"{API_BASE}/repos/{owner_repo}/branches/{b}"
        r = requests.get(url, headers=_gh_headers(), timeout=60)
        if r.status_code == 200:
            return b
    info = github_repo_info(owner_repo)
    return info.get("default_branch", "main")


def build_raw_url(owner_repo: str, path: str, branch: str) -> str:
    owner_repo = normalize_repo(owner_repo).strip("/")
    path = path.lstrip("/")
    return f"{RAW_BASE}/{owner_repo}/{branch}/{path}"


@st.cache_data(show_spinner=False)
def github_listdir(owner_repo: str, path: str, branch: str):
    owner_repo = normalize_repo(owner_repo)
    url = f"{API_BASE}/repos/{owner_repo}/contents/{path}?ref={branch}"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    if r.status_code != 200:
        return []
    return r.json()


@st.cache_data(show_spinner=True)
def github_get_contents(owner_repo: str, path: str, branch: str):
    owner_repo = normalize_repo(owner_repo)
    url = f"{API_BASE}/repos/{owner_repo}/contents/{path}?ref={branch}"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Falha listando {path}: {r.status_code} {r.text}")
    return r.json()


@st.cache_data(show_spinner=True)
def github_fetch_bytes(owner_repo: str, path: str, branch: str) -> bytes:
    meta = github_get_contents(owner_repo, path, branch)
    download_url = meta.get("download_url") or build_raw_url(owner_repo, path, branch)
    r = requests.get(download_url, headers=_gh_headers(), timeout=180)
    if r.status_code != 200:
        ct = r.headers.get("Content-Type", "")
        raise RuntimeError(
            f"Download falhou ({r.status_code}, Content-Type={ct}). Verifique token/privacidade."
        )
    data = r.content
    # Ponteiro Git LFS?
    if data.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            "Arquivo está em LFS (ponteiro). Defina token em st.secrets['github']['token']."
        )
    # HTML/JSON?
    head = data[:200].strip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise RuntimeError("Recebi HTML em vez do arquivo. Provável rate limit/privado. Defina token.")
    return data


@st.cache_data(show_spinner=True)
def load_gpkg(owner_repo: str, path: str, branch: str, layer: str | None = None):
    try:
        import geopandas as gpd
    except Exception as e:
        raise RuntimeError("geopandas/pyogrio são necessários para ler GPKG.") from e
    blob = github_fetch_bytes(owner_repo, path, branch)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
        tmp.write(blob)
        tmp.flush()
        tmp_path = tmp.name
    try:
        return gpd.read_file(tmp_path, layer=layer, engine="pyogrio")
    except Exception:
        return gpd.read_file(tmp_path, layer=layer)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@st.cache_data(show_spinner=True)
def load_parquet(owner_repo: str, path: str, branch: str) -> pd.DataFrame:
    blob = github_fetch_bytes(owner_repo, path, branch)
    return pd.read_parquet(io.BytesIO(blob), engine="pyarrow")


@st.cache_data(show_spinner=True)
def load_csv(owner_repo: str, path: str, branch: str) -> pd.DataFrame:
    blob = github_fetch_bytes(owner_repo, path, branch)
    return pd.read_csv(io.BytesIO(blob))


def list_files(owner_repo: str, path: str, branch: str, exts=(".parquet", ".csv", ".gpkg")):
    items = github_listdir(owner_repo, path, branch)
    out = []
    for it in items:
        if isinstance(it, dict) and it.get("type") == "file":
            nm = it["name"]
            if any(nm.lower().endswith(e) for e in exts):
                out.append({"path": f"{path.rstrip('/')}/{nm}", "name": nm})
    return out


@st.cache_data(show_spinner=True)
def github_branch_info(owner_repo: str, branch: str):
    owner_repo = normalize_repo(owner_repo)
    url = f"{API_BASE}/repos/{owner_repo}/branches/{branch}"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Falha lendo branch {branch}: {r.status_code} {r.text}")
    return r.json()


@st.cache_data(show_spinner=True)
def github_tree_paths(owner_repo: str, branch: str):
    info = github_branch_info(owner_repo, branch)
    tree_sha = info["commit"]["commit"]["tree"]["sha"]
    url = f"{API_BASE}/repos/{normalize_repo(owner_repo)}/git/trees/{tree_sha}?recursive=1"
    r = requests.get(url, headers=_gh_headers(), timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Falha lendo tree: {r.status_code} {r.text}")
    tree = r.json().get("tree", [])
    return [ent["path"] for ent in tree if ent.get("type") == "blob"]


def pick_existing_dir(owner_repo: str, branch: str, candidates: list[str]) -> str:
    """Tenta encontrar diretório existente (case-insensitive / alternativas)."""
    for cand in candidates:
        items = github_listdir(owner_repo, cand, branch)
        if items:
            return cand
    all_paths = github_tree_paths(owner_repo, branch)
    for cand in candidates:
        key = cand.strip("/").lower()
        for p in all_paths:
            if p.lower().startswith(key):
                parts = p.split("/")
                return "/".join(parts[: len(key.split("/"))])
    return candidates[0]


# ==========================
# CORES / CLASSIF / MAPAS / LEGENDAS
# ==========================

def hex_to_rgba(hex_color: str):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return [r, g, b, 180]


SEQUENTIAL = {
    4: ["#fee8d8", "#fdbb84", "#fc8d59", "#d7301f"],
    5: ["#feedde", "#fdbe85", "#fd8d3c", "#e6550d", "#a63603"],
    6: ["#feedde", "#fdd0a2", "#fdae6b", "#fd8d3c", "#e6550d", "#a63603"],
    7: ["#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", "#e6550d", "#a63603"],
    8: ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"],
}
CATEGORICAL = [
    "#7c3aed",
    "#d946ef",
    "#fb7185",
    "#f97316",
    "#f59e0b",
    "#facc15",
    "#fde047",
    "#a16207",
    "#9a3412",
    "#b91c1c",
    "#ea580c",
    "#be185d",
    "#9333ea",
    "#6b21a8",
    "#a21caf",
    "#c026d3",
    "#db2777",
    "#e11d48",
    "#eab308",
    "#f43f5e",
]


def pick_sequential(n: int):
    n = max(4, min(8, n))
    return SEQUENTIAL.get(n, SEQUENTIAL[6])


def pick_categorical(k: int):
    if k <= len(CATEGORICAL):
        return CATEGORICAL[:k]
    reps = (k // len(CATEGORICAL)) + 1
    return (CATEGORICAL * reps)[:k]


def is_categorical(series: pd.Series) -> bool:
    if series.dtype.kind in ("O", "b", "M", "m", "U", "S"):
        return True
    return series.dropna().nunique() <= 12


def ensure_wgs84(gdf):
    try:
        if hasattr(gdf, "crs") and gdf.crs and str(gdf.crs).lower() not in ("epsg:4326", "wgs84"):
            return gdf.to_crs(4326)
    except Exception:
        pass
    return gdf


def make_geojson(gdf):
    gdf = ensure_wgs84(gdf)
    return json.loads(gdf.to_json())


def render_geojson_layer(geojson_obj, name="Polygons"):
    return pdk.Layer(
        "GeoJsonLayer",
        geojson_obj,
        pickable=True,
        stroked=False,
        filled=True,
        extruded=False,
        get_fill_color="properties.fill_color",
        get_line_color=[100, 100, 100],
        get_line_width=0.5,
        auto_highlight=True,
    )


def render_line_layer(geojson_obj, name="Lines"):
    return pdk.Layer(
        "GeoJsonLayer",
        geojson_obj,
        pickable=True,
        stroked=True,
        filled=False,
        get_line_color=[30, 30, 30],
        get_line_width=2,
    )


def render_point_layer(geojson_obj, name="Points"):
    return pdk.Layer(
        "GeoJsonLayer",
        geojson_obj,
        pickable=True,
        point_type="circle",
        get_fill_color=[60, 60, 60, 220],
        get_radius=60,
    )


def deck(layers, satellite=False, initial_view_state=None):
    token = st.secrets.get("mapbox", {}).get("token", None)
    map_style = "mapbox://styles/mapbox/light-v11"
    if satellite:
        map_style = "mapbox://styles/mapbox/satellite-streets-v12"
    r = pdk.Deck(
        layers=layers,
        initial_view_state=initial_view_state
        or pdk.ViewState(latitude=-23.55, longitude=-46.63, zoom=10),
        map_style=map_style,
        api_keys={"mapbox": token} if token else None,
        tooltip={"text": "{name}\n{value}"},
    )
    st.pydeck_chart(r, use_container_width=True)


def osm_basemap_deck(layers, initial_view_state=None):
    tile = pdk.Layer("TileLayer", data="https://a.tile.openstreetmap.org/{z}/{x}/{y}.png")
    r = pdk.Deck(
        layers=[tile] + layers,
        initial_view_state=initial_view_state
        or pdk.ViewState(latitude=-23.55, longitude=-46.63, zoom=10),
        map_style=None,
    )
    st.pydeck_chart(r, use_container_width=True)


# ---------- LEGENDAS ----------

def _legend_row(hex_color: str, label: str):
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
           <div style="width:14px;height:14px;border-radius:3px;border:1px solid #00000022;background:{hex_color};"></div>
           <div style="font-size:0.9rem;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_legend_categorical(cmap: dict, title="Legenda"):
    st.markdown(f"**{title}**")
    for k in sorted(cmap.keys(), key=lambda x: str(x)):
        _legend_row(cmap[k], str(k))


def _fmt_num(x):
    try:
        if x == -float("inf"):
            return "-∞"
        if x == float("inf"):
            return "+∞"
        return f"{float(x):.3g}"
    except Exception:
        return str(x)


def render_legend_numeric(bins, palette, title="Legenda"):
    st.markdown(f"**{title}**")
    k = len(palette)
    for i in range(k):
        left = bins[i]
        right = bins[i + 1] if i + 1 < len(bins) else float("inf")
        if left == -float("inf"):
            label = f"≤ {_fmt_num(right)}"
        elif right == float("inf"):
            label = f"> {_fmt_num(left)}"
        else:
            label = f"({_fmt_num(left)} – {_fmt_num(right)}]"
        _legend_row(palette[i], label)


# ==========================
# FUNÇÕES AUXILIARES DE BUSCA/EXIBIÇÃO (tabelas prontas)
# ==========================

def find_files_by_patterns(owner_repo, branch, base_dirs, patterns=(), exts=(".csv", ".parquet")):
    """Procura arquivos dentro de múltiplos diretórios candidatos filtrando por padrões (substring/regex simples)."""
    found = []
    for base in base_dirs:
        base_dir = pick_existing_dir(owner_repo, branch, [base])
        for f in list_files(owner_repo, base_dir, branch, exts):
            name_low = f["name"].lower()
            ok = True if not patterns else any(re.search(p, name_low) for p in patterns)
            if ok:
                found.append({"path": f["path"], "name": f["name"], "base": base_dir})
    return found


def load_tabular(owner_repo, path, branch):
    if path.lower().endswith(".parquet"):
        return load_parquet(owner_repo, path, branch)
    return load_csv(owner_repo, path, branch)


def pairs_to_matrix(df_pairs, i_col, j_col, val_col, sym_max=True):
    m = df_pairs.pivot(index=i_col, columns=j_col, values=val_col)
    if sym_max:
        m = m.combine_first(m.T)
        m = pd.DataFrame(np.maximum(m.values, m.T.values), index=m.index, columns=m.columns)
    return m


# ==========================
# PCA — helpers e renderização (aba 4)
# ==========================

def _safe_literal_list(x):
    """
    Converte strings tipo "[0.41, 0.22, ...]" em lista de floats.
    Se já for lista, retorna como está. Se falhar, retorna [].
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    if pd.isna(x):
        return []
    s = str(x).strip()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple, np.ndarray)):
            return list(val)
    except Exception:
        pass
    try:
        s2 = s.strip("[]()")
        parts = [p.strip() for p in s2.split(",")]
        vals = []
        for p in parts:
            if p:
                vals.append(float(p))
        return vals
    except Exception:
        return []


def _render_variancia_file(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    col_group = cols.get("grupo") or cols.get("grupos") or None
    col_evr = (
        cols.get("variancia_explicada")
        or cols.get("var_exp")
        or next((c for c in df.columns if "variancia" in c.lower() and "explic" in c.lower()), None)
        or next((c for c in df.columns if "var_exp" in c.lower() and "acumul" not in c.lower()), None)
    )
    col_evr_cum = (
        cols.get("variancia_acumulada")
        or cols.get("var_exp_acumulada")
        or next((c for c in df.columns if "variancia" in c.lower() and "acumul" in c.lower()), None)
        or next((c for c in df.columns if "var_exp" in c.lower() and "acumul" in c.lower()), None)
    )
    if not col_evr:
        st.warning("Não identifiquei a coluna de variância explicada neste arquivo.")
        st.dataframe(df.head(), use_container_width=True)
        return

    df_use = df.copy()
    if col_group and col_group in df_use.columns:
        grupos = df_use[col_group].dropna().astype(str).unique().tolist()
        if grupos:
            g_sel = st.selectbox("Grupo (quando aplicável)", grupos, index=0, key="pca_group")
            df_use = df_use[df_use[col_group].astype(str) == g_sel]
    if len(df_use) > 1 and (col_evr in df_use.columns):
        df_use = df_use.head(1)

    evr_list = _safe_literal_list(df_use.iloc[0][col_evr])
    if col_evr_cum and col_evr_cum in df_use.columns:
        evr_cum_list = _safe_literal_list(df_use.iloc[0][col_evr_cum])
    else:
        total = 0.0
        evr_cum_list = []
        for v in evr_list:
            total += float(v)
            evr_cum_list.append(total)

    df_plot = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(evr_list))],
            "explained_variance_ratio": evr_list,
            "cumulative": evr_cum_list,
        }
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            df_plot,
            x="component",
            y="explained_variance_ratio",
            title="Scree — Variância explicada por componente",
        )
        st.plotly_chart(fig, use_container_width=True)
        download_plotly_png(fig, "pca_scree")
    with c2:
        fig2 = px.line(
            df_plot,
            x="component",
            y="cumulative",
            markers=True,
            title="Variância explicada acumulada",
        )
        st.plotly_chart(fig2, use_container_width=True)
        download_plotly_png(fig2, "pca_variancia_acumulada")

    st.subheader("Tabela — Variância")
    st.dataframe(df_plot, use_container_width=True)
    download_df(df_plot, "pca_variancia_tabela")


def _render_pipeline_file(df: pd.DataFrame):
    first_col = df.columns[0]
    if df[first_col].astype(str).str.lower().head(5).isin(["pca", "imputer", "scaler", "cols", "k"]).any():
        df2 = df.set_index(first_col)
    else:
        df2 = df.copy()

    st.subheader("Tabela — Pipeline PCA por grupo")
    st.dataframe(df2, use_container_width=True)
    try:
        k_row = df2.loc[[c for c in df2.index if str(c).lower() == "k"][0]]
        st.caption("Componentes (k) por grupo:")
        st.write(k_row.to_frame("k").T)
    except Exception:
        pass


def _find_pca_base_dir(repo, branch, pick_existing_dir):
    return pick_existing_dir(repo, branch, ["Data/analises/PCA", "Data/Analises/PCA", "data/analises/PCA"])


def _classify_pca_file(df: pd.DataFrame):
    cols = [c.lower() for c in df.columns]
    if any(("explained" in c and "ratio" in c) for c in cols) or "explained_variance_ratio" in cols:
        return "evr"
    if ("component" in cols and ("loading" in cols or "valor" in cols or "carga" in cols)):
        return "loadings_long"
    pc_like = [c for c in cols if c.startswith("pc") or c.startswith("component")]
    if len(pc_like) >= 2:
        return "loadings_wide"
    id_like = any(c in cols for c in ["sq", "id", "codigo", "code"])
    has_pcs = any(c.startswith("pc") for c in cols)
    if has_pcs:
        return "scores" if id_like else "scores_no_id"
    return "unknown"


def _list_candidate_files(repo, branch, base_dir, list_files, load_parquet, load_csv):
    files_all = list_files(repo, base_dir, branch, (".parquet", ".csv"))
    candidates = {"evr": [], "loadings": [], "scores": [], "unknown": []}
    for f in files_all:
        try:
            df = (
                load_parquet(repo, f["path"], branch)
                if f["name"].endswith(".parquet")
                else load_csv(repo, f["path"], branch)
            )
            kind = _classify_pca_file(df)
        except Exception:
            kind = "unknown"
        if kind == "evr":
            candidates["evr"].append((f, "evr"))
        elif kind in ("loadings_long", "loadings_wide"):
            candidates["loadings"].append((f, kind))
        elif kind in ("scores", "scores_no_id"):
            candidates["scores"].append((f, kind))
        else:
            candidates["unknown"].append((f, "unknown"))
    return candidates


def _tidy_loadings(df: pd.DataFrame):
    cols_lower = {c: c.lower() for c in df.columns}
    if "component" in cols_lower.values() and any(x in cols_lower.values() for x in ["loading", "valor", "carga"]):
        comp_col = next(k for k, v in cols_lower.items() if v == "component")
        load_col = next(k for k, v in cols_lower.items() if v in ("loading", "valor", "carga"))
        var_col = next((k for k, v in cols_lower.items() if v in ("variable", "feature", "variavel", "atributo")), None)
        if var_col is None:
            non_num = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != comp_col]
            var_col = non_num[0] if non_num else comp_col
        out = df[[var_col, comp_col, load_col]].copy()
        out.columns = ["variable", "component", "loading"]
        return out
    pc_cols = [c for c in df.columns if c.lower().startswith("pc") or c.lower().startswith("component")]
    if pc_cols:
        var_candidates = [c for c in df.columns if c not in pc_cols]
        if len(var_candidates) == 0:
            df = df.copy()
            df["variable"] = df.index.astype(str)
            var_col = "variable"
        else:
            var_col = var_candidates[0]
        long = df.melt(id_vars=[var_col], value_vars=pc_cols, var_name="component", value_name="loading")
        long.columns = ["variable", "component", "loading"]
        return long
    return pd.DataFrame(columns=["variable", "component", "loading"])


def _prep_scores(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    pc_cols = [c for c in df.columns if c.lower().startswith("pc")]
    id_col = cols.get("sq") or cols.get("id") or cols.get("codigo") or cols.get("code")
    ano_col = cols.get("ano")
    return pc_cols, id_col, ano_col


def _render_evr_section(df_evr: pd.DataFrame):
    cols = {c.lower(): c for c in df_evr.columns}
    if "explained_variance_ratio" in cols:
        evr_col = cols["explained_variance_ratio"]
        comp_col = None
    else:
        evr_col = next((c for c in df_evr.columns if "explained" in c.lower() and "ratio" in c.lower()), None)
        comp_col = next((c for c in df_evr.columns if c.lower().startswith("comp") or c.lower().startswith("pc")), None)
    df = df_evr.copy()
    if comp_col is None:
        df = df.reset_index(drop=True)
        df["component"] = [f"PC{i+1}" for i in range(len(df))]
        comp_col = "component"
    else:
        df["component"] = df[comp_col].astype(str)
    df["explained_variance_ratio"] = df[evr_col].astype(float)
    df = df[["component", "explained_variance_ratio"]].dropna()
    try:
        df = df.sort_values("component")
    except Exception:
        pass
    df["cumulative"] = df["explained_variance_ratio"].cumsum()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            df,
            x="component",
            y="explained_variance_ratio",
            title="Scree — Variância explicada por componente",
        )
        st.plotly_chart(fig, use_container_width=True)
        download_plotly_png(fig, f"heatmap_{analise_tipo}")
    with c2:
        fig2 = px.line(
            df, x="component", y="cumulative", markers=True, title="Variância explicada acumulada"
        )
        st.plotly_chart(fig2, use_container_width=True)
        download_plotly_png(fig2, f"heatmap_{analise_tipo}")

    st.subheader("Tabela — Variância explicada")
    st.dataframe(df, use_container_width=True)


def _render_loadings_section(df_load: pd.DataFrame):
    long = _tidy_loadings(df_load)
    if long.empty:
        st.warning("Não foi possível identificar a estrutura de *loadings* deste arquivo.")
        st.dataframe(df_load.head(), use_container_width=True)
        return
    comps = sorted(long["component"].astype(str).unique(), key=lambda x: (len(x), x))
    c1, c2 = st.columns([2, 1])
    with c1:
        comp_sel = st.selectbox("Componente", comps, index=0, key="pca_comp")
    with c2:
        topn = st.slider("Top |loading|", 5, 30, 15, key="pca_topn")

    sub = long[long["component"].astype(str) == str(comp_sel)].copy()
    sub["abs_loading"] = sub["loading"].abs()
    sub = sub.sort_values("abs_loading", ascending=False).head(topn)
    fig = px.bar(
        sub.sort_values("abs_loading"),
        x="abs_loading",
        y="variable",
        orientation="h",
        title=f"Maiores |loadings| — {comp_sel}",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Tabela — Loadings")
    st.dataframe(sub.drop(columns=["abs_loading"]), use_container_width=True)


def _render_scores_section(df_scores: pd.DataFrame, repo, branch, pick_existing_dir, list_files, load_parquet, load_csv):
    pc_cols, id_col, ano_col = _prep_scores(df_scores)
    if not pc_cols:
        st.warning("Arquivo de *scores* sem colunas de PCs identificáveis.")
        st.dataframe(df_scores.head(), use_container_width=True)
        return

    if ano_col:
        anos = sorted([int(x) for x in df_scores[ano_col].dropna().unique()])
        ano_sel = st.select_slider("Ano (scores)", options=anos, value=anos[-1], key="pca_scores_ano")
        df_scores = df_scores[df_scores[ano_col] == ano_sel]

    pc_x = st.selectbox("PC eixo X", pc_cols, index=0, key="pca_scores_x")
    pc_y = st.selectbox("PC eixo Y", pc_cols, index=1 if len(pc_cols) > 1 else 0, key="pca_scores_y")
    hover_cols = [pc_x, pc_y]
    if id_col:
        hover_cols.insert(0, id_col)

    fig = px.scatter(
        df_scores, x=pc_x, y=pc_y, hover_data=hover_cols, title=f"Biplot (scores) — {pc_x} × {pc_y}"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Tabela — Scores (colunas selecionadas)")
    st.dataframe(df_scores[hover_cols].dropna(how="all"), use_container_width=True)


def render_pca_tab_inline(repo, branch, pick_existing_dir, list_files, load_parquet, load_csv):
    st.subheader("Arquivos de PCA (sem recálculo)")
    base_dir = _find_pca_base_dir(repo, branch, pick_existing_dir)
    st.caption(f"Diretório PCA: `{base_dir}`")

    files_all = list_files(repo, base_dir, branch, (".csv", ".parquet"))
    if not files_all:
        st.info("Nenhum arquivo encontrado em `Data/analises/PCA` (ou variações).")
        return

    nomes = [f["name"] for f in files_all]
    evr_default = [n for n in nomes if "variancia" in n.lower() or "var_exp" in n.lower()]
    pipe_default = [n for n in nomes if n.lower().startswith("pca")]

    st.markdown("### 1) Variância explicada")
    evr_sel = st.selectbox(
        "Selecione arquivo de variância explicada",
        evr_default or nomes,
        index=0,
        key="pca_evr_file",
    )
    evr_obj = next(x for x in files_all if x["name"] == evr_sel)
    df_evr = (
        load_parquet(repo, evr_obj["path"], branch)
        if evr_obj["name"].endswith(".parquet")
        else load_csv(repo, evr_obj["path"], branch)
    )

    kind_evr = _classify_pca_file(df_evr)
    if kind_evr == "evr":
        _render_variancia_file(df_evr)
    else:
        st.warning("Este arquivo não parece conter variância explicada. Exibindo preview:")
        st.dataframe(df_evr.head(), use_container_width=True)

    st.divider()

    st.markdown("### 2) Pipeline / Modelo (opcional)")
    pipe_sel = st.selectbox(
        "Selecione arquivo de pipeline/modelo",
        pipe_default or nomes,
        index=0,
        key="pca_pipe_file",
    )
    pipe_obj = next(x for x in files_all if x["name"] == pipe_sel)
    df_pipe = (
        load_parquet(repo, pipe_obj["path"], branch)
        if pipe_obj["name"].endswith(".parquet")
        else load_csv(repo, pipe_obj["path"], branch)
    )

    kind_pipe = _classify_pca_file(df_pipe)
    if kind_pipe == "pipeline":
        _render_pipeline_file(df_pipe)
    else:
        st.info("Arquivo não reconhecido como pipeline. Exibindo preview:")
        st.dataframe(df_pipe.head(), use_container_width=True)


# ==========================
# SIDEBAR — Repositório e Mapbox
# ==========================
with st.sidebar:
    st.header("🔗 Fonte dos Dados (GitHub)")
    repo_input = st.text_input("owner/repo", value="emiliobneto/UrbanTechCluster")
    branch_input = st.text_input("branch (vazio = auto)", value="")
    try:
        repo = normalize_repo(repo_input)
        branch = resolve_branch(repo, branch_input)
        st.caption(f"Usando: **{repo}@{branch}**")
    except Exception as e:
        st.error(f"Configuração inválida: {e}")
        st.stop()
    st.divider()
    st.header("🗺️ Mapbox (opcional)")
    st.caption("Defina `mapbox.token` em secrets para habilitar satélite.")

if not repo or not branch:
    st.stop()

# ==========================
# TABS
# ==========================
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Principal", "🧬 Clusterização", "📊 Univariadas", "🧠 ML → PCA"])

# -----------------------------------------------------------------------------
# ABA 1 — Principal (mapa + dados por SQ + recortes)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ABA 1 — Principal (mapa + dados por SQ + recortes) — VERSÃO REESCRITA
# -----------------------------------------------------------------------------
with tab1:
    # ---------------- Helpers locais (evitam colisões com outras abas) ----------------
    def _t1_download_df(df: pd.DataFrame, base_name: str):
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Baixar CSV", csv, file_name=f"{base_name}.csv", mime="text/csv", key=f"t1_dl_{base_name}")

    def _t1_png_mapa_300dpi(gdf_pol, value_col, cmap_dict, gdf_overlay=None, titulo=None, dpi=300):
        """Gera PNG de mapa temático em alta resolução (300dpi) via matplotlib."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
        # plota por classe/categoria
        for k, hexc in cmap_dict.items():
            sub = gdf_pol[gdf_pol[value_col] == k]
            if not sub.empty:
                try:
                    sub.plot(ax=ax, color=hexc, linewidth=0, edgecolor="none")
                except Exception:
                    # fallback se houver geometria inválida
                    sub = sub.buffer(0)
                    sub.plot(ax=ax, color=hexc, linewidth=0, edgecolor="none")
        if gdf_overlay is not None:
            try:
                gdf_overlay.boundary.plot(ax=ax, linewidth=1)
            except Exception:
                ensure_wgs84(gdf_overlay).boundary.plot(ax=ax, linewidth=1)
        ax.set_axis_off()
        if titulo:
            ax.set_title(titulo)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    # ---------------- Cabeçalho / base map ----------------
    st.subheader("Quadras e camadas adicionais (GPKG)")
    colA, colB = st.columns([2, 1], gap="large")
    with colA:
        st.caption("Carrega `Data/mapa/quadras.gpkg` e camadas auxiliares.")
    with colB:
        basemap = st.radio("Plano de fundo", ["OpenStreetMap", "Satélite (Mapbox)"], index=0, key="t1_base")

    # ---------------- Carrega quadras com fallback e cache ----------------
    quadras_path_default = "Data/mapa/quadras.gpkg"
    quadras_path_used = quadras_path_default
    gdf_quadras = st.session_state.get("gdf_quadras_cached")
    if gdf_quadras is None:
        first_err = None
        try:
            gdf_quadras = load_gpkg(repo, quadras_path_default, branch)
        except Exception as e:
            first_err = e
            all_paths = github_tree_paths(repo, branch)
            candidates = [p for p in all_paths if p.lower().endswith("quadras.gpkg")]
            candidates = sorted(candidates, key=lambda p: ("/data/" not in p.lower(), "/mapa/" not in p.lower(), len(p)))
            if not candidates:
                st.error(f"Não encontrei 'quadras.gpkg'. Erro ao tentar '{quadras_path_default}': {first_err}")
                st.stop()
            quadras_path_used = st.selectbox("Selecione o arquivo de quadras:", candidates, index=0, key="t1_quadras_sel")
            gdf_quadras = load_gpkg(repo, quadras_path_used, branch)
            st.success(f"Carregado: {quadras_path_used}")
        st.session_state["gdf_quadras_cached"] = gdf_quadras

    # Detecta coluna SQ na camada de quadras
    sq_col_quadras = "SQ" if "SQ" in gdf_quadras.columns else next((c for c in gdf_quadras.columns if str(c).upper() == "SQ"), None)
    if not sq_col_quadras:
        st.error("Camada de quadras não possui coluna 'SQ'.")
        st.stop()

    # ---------------- Camadas auxiliares (opcional) ----------------
    loaded_layers = []
    other_layers_paths = []
    try:
        mapa_dir = pick_existing_dir(repo, branch, ["Data/mapa", "data/mapa", "Data/Mapa"])
        mapa_files = list_files(repo, mapa_dir, branch, (".gpkg",))
        other_layers = [f for f in mapa_files if f["name"].lower() != "quadras.gpkg"]
        layer_names = [f["name"] for f in other_layers]
        sel_layers = st.multiselect("Camadas auxiliares (opcional)", layer_names, default=[], key="t1_layers")
        for nm in sel_layers:
            fobj = next(x for x in other_layers if x["name"] == nm)
            g = load_gpkg(repo, fobj["path"], branch)
            loaded_layers.append((nm, g))
            other_layers_paths.append(fobj["path"])
    except Exception as e:
        st.warning(f"Não foi possível listar/ler camadas em Data/mapa: {e}")

    # ---------------- Dados por SQ — seleção de fonte/arquivo/variável/ano ----------------
    st.subheader("Dados por `SQ` para espacialização")
    col1, col2, col3 = st.columns([1.6, 1, 1.2], gap="large")

    with col1:
        src_label = st.radio("Origem dos dados", ["originais", "winsorize"], index=0, horizontal=True, key="t1_src")
        base_dir = pick_existing_dir(
            repo, branch,
            [
                f"Data/dados/{src_label}",
                f"Data/dados/{'Originais' if src_label=='originais' else 'winsorizados'}",
                f"Data/dados/{'originais' if src_label=='originais' else 'winsorize'}",
            ],
        )
        parquets_all = list_files(repo, base_dir, branch, (".parquet",))
        incl_pred = st.checkbox("Incluir arquivos de predição (pred_*)", value=True, key="t1_incl_pred")
        parquet_files = [f for f in parquets_all if incl_pred or not f["name"].lower().startswith("pred_")]
        if not parquet_files:
            st.warning(f"Nenhum .parquet encontrado em {base_dir}.")
            st.stop()
        sel_file = st.selectbox("Arquivo .parquet com variáveis", [f["name"] for f in parquet_files], key="t1_varfile")
        fobj = next(x for x in parquet_files if x["name"] == sel_file)
        data_file_path = fobj["path"]
        df_vars = load_parquet(repo, data_file_path, branch)

    with col2:
        join_col = next((c for c in df_vars.columns if str(c).upper() == "SQ"), None)
        if join_col is None:
            st.error("Dataset selecionado não possui coluna 'SQ'.")
            st.stop()
        years_col = next((c for c in df_vars.columns if str(c).lower() in ("ano", "year")), None)
        years = sorted([int(y) for y in df_vars[years_col].dropna().unique()]) if years_col else []
        year = st.select_slider("Ano", options=years, value=years[-1], key="t1_ano") if years else None
        if year is not None and years_col:
            df_vars = df_vars[df_vars[years_col] == year]

    with col3:
        id_like = {c for c in df_vars.columns if str(c).lower() in {"sq", "id", "codigo", "code"}}
        time_like = {c for c in df_vars.columns if str(c).lower() in {"ano", "year"}}
        ignore_cols = id_like | time_like
        num_cols = [c for c in df_vars.columns if pd.api.types.is_numeric_dtype(df_vars[c])]
        var_options = [c for c in num_cols if c not in ignore_cols] or [c for c in df_vars.columns if c not in ignore_cols]
        var_sel = st.selectbox("Variável a mapear", var_options, key="t1_varname")
        n_classes = st.slider("Quebras (Jenks)", min_value=4, max_value=8, value=6, key="t1_jenks")

    # ---------------- Merge com quadras ----------------
    gdf = gdf_quadras.merge(df_vars[[join_col, var_sel]], left_on=sq_col_quadras, right_on=join_col, how="left")

    # ---------------- Classificação (Jenks → fallback quantis) ----------------
    series = gdf[var_sel]
    legend_kind, legend_info, cmap = None, None, None

    if is_categorical(series):
        cats = [c for c in series.dropna().unique()]
        try:
            cats_sorted = sorted(cats, key=lambda x: float(x))
        except Exception:
            cats_sorted = sorted(cats, key=lambda x: str(x))
        palette = pick_categorical(len(cats_sorted))
        cmap = {cat: palette[i] for i, cat in enumerate(cats_sorted)}
        gdf["value"] = series
        legend_kind = "categorical"
        legend_info = cmap
    else:
        vals = series.dropna().astype(float).values
        uniq = np.unique(vals)
        k = max(4, min(8, n_classes))
        if len(uniq) < max(4, k):
            k = min(len(uniq), max(2, k))
        try:
            import mapclassify as mc
            nb = mc.NaturalBreaks(vals, k=k, initial=200)
            bins = [-float("inf")] + list(nb.bins)
            binned = pd.cut(series, bins=bins, labels=False, include_lowest=True)
            gdf["value"] = binned
            palette = pick_sequential(k)
            cmap = {i: palette[i] for i in range(len(palette))}
            legend_kind = "numeric"
            legend_info = (bins, palette)
        except Exception:
            # Fallback: quantis
            try:
                labels = list(range(k))
                binned, bins = pd.qcut(series, q=k, labels=labels, retbins=True, duplicates="drop")
                binned = pd.Series(binned, index=series.index).astype("float").astype("Int64")
                gdf["value"] = binned
                palette = pick_sequential(len(np.unique(binned.dropna())))
                cmap = {i: palette[i] for i in range(len(palette))}
                legend_kind = "numeric"
                legend_info = (bins, palette)
            except Exception as e:
                st.error(f"Falha na classificação dos valores ({e}).")
                st.stop()

    # ---------------- GeoJSON + camadas de mapa ----------------
    geojson = make_geojson(gdf)
    for feat in geojson.get("features", []):
        val = feat.get("properties", {}).get("value", None)
        if legend_kind == "numeric":
            hexc = cmap.get(val, "#999999")
        else:
            hexc = legend_info.get(val, "#999999")
        feat.setdefault("properties", {})["fill_color"] = hex_to_rgba(hexc)

    layers = [render_geojson_layer(geojson, name="quadras")]
    for nm, g in loaded_layers:
        gj = make_geojson(g)
        try:
            geoms = set(g.geometry.geom_type.unique())
        except Exception:
            geoms = {"Polygon"}
        if geoms <= {"LineString", "MultiLineString"}:
            layers.append(render_line_layer(gj, nm))
        elif geoms <= {"Point", "MultiPoint"}:
            layers.append(render_point_layer(gj, nm))
        else:
            layers.append(render_geojson_layer(gj, nm))

    # ---------------- Mapa + Legenda ----------------
    st.markdown("#### Mapa — Quadras + Camadas auxiliares")
    map_col, legend_col = st.columns([4, 1], gap="large")
    with map_col:
        if basemap.startswith("Satélite"):
            deck(layers, satellite=True)
        else:
            osm_basemap_deck(layers)

        # Export PNG 300 DPI (mapa geral) — sob demanda para evitar custo desnecessário
        if st.button("🖼️ Gerar PNG 300 DPI (mapa geral)", key="t1_btn_png_geral"):
            try:
                titulo = f"{var_sel}" + (f" — {year}" if years and year is not None else "")
                png_bytes = _t1_png_mapa_300dpi(gdf[["value", "geometry"]].dropna(subset=["value"]), "value", cmap, gdf_overlay=None, titulo=titulo, dpi=300)
                st.download_button("Baixar PNG 300 DPI (mapa geral)", png_bytes, file_name=f"mapa_geral_{var_sel}{'_'+str(year) if years and year is not None else ''}.png", mime="image/png", key="t1_dl_png_geral")
            except Exception as e:
                st.caption(f"Export PNG indisponível ({e})")

        # Export tabela (SQ + variável)
        df_export = gdf[[sq_col_quadras, var_sel]].rename(columns={sq_col_quadras: "SQ"})
        _t1_download_df(df_export, f"dados_mapa_{var_sel}{'_'+str(year) if years and year is not None else ''}")

    with legend_col:
        if legend_kind == "categorical":
            render_legend_categorical(legend_info, title=f"Legenda — {var_sel}")
        elif legend_kind == "numeric":
            bins, palette = legend_info
            render_legend_numeric(bins, palette, title=f"Legenda — {var_sel}")
        else:
            st.caption("Sem legenda.")

    # ---------------- Recortes: filtra métricas apenas para a área ----------------
    st.subheader("Recortes espaciais (GPKG)")
    st.caption("Selecione um GPKG em `Data/mapa/recortes` para filtrar os SQs e ver métricas apenas dessa área.")
    rec_dir = None
    recorte_file_path = None
    try:
        rec_dir = pick_existing_dir(repo, branch, ["Data/mapa/recortes", "Data/Mapa/recortes", "data/mapa/recortes"])
        recorte_files = list_files(repo, rec_dir, branch, (".gpkg",))
        if not recorte_files:
            st.info("Nenhum GPKG de recorte encontrado.")
        else:
            colR0, colR1 = st.columns([3, 2], gap="large")

            with colR0:
                rec_sel_name = st.selectbox("Arquivo de recorte (.gpkg)", [f["name"] for f in recorte_files], index=0, key="t1_rec_file")
                rec_obj = next(x for x in recorte_files if x["name"] == rec_sel_name)
                recorte_file_path = rec_obj["path"]
                gdf_rec = load_gpkg(repo, recorte_file_path, branch)

                # Desenha apenas o recorte selecionado
                layers_rec = [render_line_layer(make_geojson(gdf_rec), name="recorte")]
                st.markdown("#### Mapa — Recorte selecionado")
                if basemap.startswith("Satélite"):
                    deck(layers_rec, satellite=True)
                else:
                    osm_basemap_deck(layers_rec)

            with colR1:
                st.markdown("**Legenda — Recorte**")
                _legend_row("#444444", "Contorno do recorte")

                # Interseção dos SQs com o recorte (respeitando CRS)
                try:
                    import geopandas as gpd
                    gq = ensure_wgs84(gdf_quadras[[sq_col_quadras, "geometry"]].copy())
                    gr = ensure_wgs84(gdf_rec[["geometry"]].copy())
                    # sjoin quando possível
                    try:
                        sq_sel = gpd.sjoin(gq, gr, predicate="intersects", how="inner")[sq_col_quadras].unique().tolist()
                    except Exception:
                        bbox = gr.total_bounds  # fallback por bbox
                        sq_sel = gq.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]][sq_col_quadras].unique().tolist()
                except Exception as e:
                    st.error(f"Falha ao cruzar recorte com SQs: {e}")
                    sq_sel = []

                st.metric("SQs no recorte", len(sq_sel))

            # Filtra dados (df_vars já foi filtrado por Ano)
            df_vars_rec = df_vars[df_vars[join_col].isin(sq_sel)].copy()

            # Escolha de métricas + exibição
            st.markdown("#### Métricas para a área recortada")
            id_like = {c for c in df_vars_rec.columns if str(c).lower() in {"sq", "id", "codigo", "code"}}
            time_like = {c for c in df_vars_rec.columns if str(c).lower() in {"ano", "year"}}
            ignore_cols = id_like | time_like
            num_cols_rec = [c for c in df_vars_rec.columns if pd.api.types.is_numeric_dtype(df_vars_rec[c])]
            var_opts_rec = [c for c in num_cols_rec if c not in ignore_cols] or [c for c in df_vars_rec.columns if c not in ignore_cols]

            leftC, rightC = st.columns([3, 2], gap="large")
            with leftC:
                vars_escolhidas = st.multiselect(
                    "Variáveis (métricas) a exibir",
                    var_opts_rec,
                    default=[var_sel] if var_sel in var_opts_rec else var_opts_rec[: min(5, len(var_opts_rec))],
                    key="t1_vars_rec"
                )
                modo = st.radio("Exibição", ["Por SQ", "Resumo (estatísticas)"], horizontal=True, index=0, key="t1_modo_rec")

                if vars_escolhidas:
                    if modo == "Por SQ":
                        df_out = df_vars_rec[[join_col] + vars_escolhidas].sort_values(join_col)
                        st.dataframe(df_out, use_container_width=True)
                        _t1_download_df(df_out, f"recorte_porSQ_{os.path.splitext(rec_sel_name)[0]}")
                    else:
                        desc = df_vars_rec[vars_escolhidas].describe().T
                        st.dataframe(desc, use_container_width=True)
                        _t1_download_df(desc.reset_index().rename(columns={"index": "variavel"}), f"recorte_resumo_{os.path.splitext(rec_sel_name)[0]}")

            with rightC:
                st.markdown("**Exportar dados/arquivos da área recortada**")
                base_nome = f"recorte_{os.path.splitext(rec_sel_name)[0]}"

                # GeoJSON com SQs do recorte
                try:
                    gdf_clip = gdf_quadras[gdf_quadras[sq_col_quadras].isin(sq_sel)][[sq_col_quadras, "geometry"]].copy()
                    gj_clip = make_geojson(gdf_clip)
                    st.download_button(
                        "📥 GeoJSON (SQs do recorte)",
                        data=json.dumps(gj_clip),
                        file_name=f"{base_nome}_sqs.geojson",
                        mime="application/geo+json",
                        key="t1_dl_geojson_rec"
                    )
                except Exception:
                    pass

                # PNG 300 DPI do mapa temático recortado (sob demanda)
                if st.button("🖼️ Gerar PNG 300 DPI (mapa no recorte)", key="t1_btn_png_rec"):
                    try:
                        gdf_color = gdf[gdf[sq_col_quadras].isin(sq_sel)][[sq_col_quadras, "value", "geometry"]].copy()
                        titulo = f"{var_sel}" + (f" — {year}" if years and year is not None else "")
                        png_bytes = _t1_png_mapa_300dpi(gdf_color.dropna(subset=["value"]), "value", cmap, gdf_overlay=gdf_rec, titulo=titulo, dpi=300)
                        st.download_button("Baixar PNG 300 DPI (mapa do recorte)", png_bytes, file_name=f"{base_nome}_mapa_300dpi.png", mime="image/png", key="t1_dl_png_rec")
                    except Exception as e:
                        st.caption(f"Export PNG indisponível ({e})")
    except Exception as e:
        st.warning(f"Não foi possível listar/ler recortes: {e}")

    # ---------------- Debug — caminhos usados ----------------
    with st.expander("🔎 Debug — caminhos usados (Aba 1)"):
        debug_info = {
            "repo@branch": f"{repo}@{branch}",
            "quadras_path_usado": quadras_path_used,
            "mapa_dir": mapa_dir if 'mapa_dir' in locals() else None,
            "camadas_auxiliares_sel": other_layers_paths,
            "dados_base_dir": base_dir,
            "arquivo_dados_selecionado": data_file_path,
            "recortes_dir": rec_dir,
            "arquivo_recorte_sel": recorte_file_path,
            "coluna_SQ_quadras": sq_col_quadras,
            "coluna_SQ_dados": join_col,
            "coluna_ano": years_col if 'years_col' in locals() else None,
            "ano_selecionado": year if 'year' in locals() else None,
            "variavel_mapeada": var_sel,
            "classes_k": n_classes,
            "legend_kind": legend_kind,
        }
        st.code(json.dumps(debug_info, ensure_ascii=False, indent=2), language="json")

# -----------------------------------------------------------------------------
# ABA 2 — Clusterização (somente leitura/visualização)
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Mapa de Clusters + Métricas + Spearman (somente leitura)")

    # 2.1) Clusters (tolerante a variações de diretório/arquivo)
    clusters_dir = pick_existing_dir(
        repo,
        branch,
        ["Data/dados/Originais", "Data/dados/originais", "data/dados/originais"],
    )
    all_in_dir = list_files(repo, clusters_dir, branch, (".csv", ".parquet"))
    cand = [
        f
        for f in all_in_dir
        if re.fullmatch(r"(?i)EstagioClusterizacao\.(csv|parquet)", f["name"])
    ]
    if not cand:
        cand = [
            f
            for f in all_in_dir
            if re.search(r"(?i)estagio", f["name"]) and re.search(r"(?i)cluster", f["name"])
        ]
    if not cand:
        st.error(
            "Não encontrei arquivo de clusterização. Coloque `EstagioClusterizacao.csv`/`.parquet` ou nome contendo 'estagio' e 'cluster'."
        )
        st.stop()

    est_file = cand[0]
    df_est = (
        load_parquet(repo, est_file["path"], branch)
        if est_file["name"].lower().endswith(".parquet")
        else load_csv(repo, est_file["path"], branch)
    )

    years = sorted([int(y) for y in df_est["Ano"].dropna().unique()]) if "Ano" in df_est.columns else []
    year_sel = (
        st.select_slider("Ano", options=years, value=years[-1], key="clu_ano") if years else None
    )
    if year_sel is not None:
        df_est = df_est[df_est["Ano"] == year_sel]

    cluster_cols = [c for c in df_est.columns if re.search(r"(?i)(cluster|estagio|label)", c)]
    if not cluster_cols:
        st.error("Não encontrei coluna de cluster (ex.: EstagioClusterizacao, Cluster, Label).")
        st.stop()
    preferred = next((c for c in cluster_cols if c.lower() == "estagioclusterizacao"), cluster_cols[0])
    cluster_col = st.selectbox(
        "Coluna de cluster", cluster_cols, index=cluster_cols.index(preferred), key="clu_cluster_col"
    )

    gdf_quadras = st.session_state.get("gdf_quadras_cached")
    if gdf_quadras is None:
        try:
            gdf_quadras = load_gpkg(repo, "Data/mapa/quadras.gpkg", branch)
            st.session_state["gdf_quadras_cached"] = gdf_quadras
        except Exception as e:
            st.error(
                f"Não foi possível carregar as quadras (Data/mapa/quadras.gpkg). Detalhe: {e}"
            )
            st.stop()

    join_est = next((c for c in df_est.columns if str(c).upper() == "SQ"), None)
    join_quad = next((c for c in gdf_quadras.columns if str(c).upper() == "SQ"), None)
    if not (join_est and join_quad):
        st.error("Tanto a tabela de clusters quanto as quadras precisam ter a coluna 'SQ' (qualquer caixa).")
        st.stop()

    gdfc = gdf_quadras.merge(
        df_est[[join_est, cluster_col]], left_on=join_quad, right_on=join_est, how="left"
    )

    cats = gdfc[cluster_col].dropna().unique().tolist()
    try:
        cats_sorted = sorted(cats, key=lambda x: float(x))
    except Exception:
        cats_sorted = sorted(cats, key=lambda x: str(x))
    palette = pick_categorical(len(cats_sorted))
    cmap = {cat: palette[i] for i, cat in enumerate(cats_sorted)}

    gdfc["value"] = gdfc[cluster_col]
    gj = make_geojson(gdfc)
    for feat in gj.get("features", []):
        val = feat.get("properties", {}).get("value", None)
        hexc = cmap.get(val, "#999999")
        feat.setdefault("properties", {})["fill_color"] = hex_to_rgba(hexc)

    colM, colL = st.columns([4, 1], gap="large")
    with colM:
        st.markdown("#### Mapa — Clusters por SQ")
        base_map = st.radio(
            "Plano de fundo", ["OpenStreetMap", "Satélite (Mapbox)"], index=0, horizontal=True, key="clu_base"
        )
        if base_map.startswith("Satélite"):
            deck([render_geojson_layer(gj, name="clusters")], satellite=True)
        else:
            osm_basemap_deck([render_geojson_layer(gj, name="clusters")])
    with colL:
        st.markdown(f"**Legenda — {cluster_col}**")
        for k in cats_sorted:
            _legend_row(cmap[k], str(k))

    # Recortes
    st.markdown("#### Mapa — Recortes (opcional)")
    rec_dir = pick_existing_dir(
        repo, branch, ["Data/mapa/recortes", "Data/Mapa/recortes", "data/mapa/recortes"]
    )
    try:
        rec_files = list_files(repo, rec_dir, branch, (".gpkg",))
        if rec_files:
            rec_sel = st.selectbox(
                "Arquivo de recorte", [f["name"] for f in rec_files], index=0, key="clu_rec_file"
            )
            rec_obj = next(x for x in rec_files if x["name"] == rec_sel)
            gdf_rec = load_gpkg(repo, rec_obj["path"], branch)
            layers_rec = [render_line_layer(make_geojson(gdf_rec), name="recorte")]
            colRm, colRl = st.columns([4, 1], gap="large")
            with colRm:
                if base_map.startswith("Satélite"):
                    deck(layers_rec, satellite=True)
                else:
                    osm_basemap_deck(layers_rec)
            with colRl:
                st.markdown("**Legenda — Recorte**")
                _legend_row("#444444", "Contorno do recorte")
        else:
            st.caption(f"Pasta `{rec_dir}` vazia.")
    except Exception as e:
        st.warning(f"Não foi possível listar/ler recortes: {e}")

    st.divider()

    # 2.4) Métricas por cluster/ano
    st.subheader("Métricas por cluster/ano")
    versao = st.radio(
        "Versão de análise", ["originais", "winsorizados"], index=0, horizontal=True, key="clu_ver"
    )
    base_metrics = (
        pick_existing_dir(repo, branch, ["Data/analises/original", "Data/Analises/original", "data/analises/original"])  # noqa: E501
        if versao == "originais"
        else pick_existing_dir(
            repo,
            branch,
            ["Data/analises/winsorizados", "Data/Analises/winsorizados", "data/analises/winsorizados"],
        )
    )

    met_files = [
        f
        for f in list_files(repo, base_metrics, branch, (".parquet", ".csv"))
        if f["name"].lower() in ("metricas.parquet", "metricas.csv")
    ]
    if not met_files:
        met_files = [
            f
            for f in list_files(repo, base_metrics, branch, (".parquet", ".csv"))
            if re.search(r"(?i)metrica|metrics", f["name"]) is not None
        ]

    if met_files:
        met_files = sorted(
            met_files, key=lambda f: 0 if f["name"].lower().endswith(".parquet") else 1
        )
        sel_met = st.selectbox(
            "Arquivo de métricas", [f["name"] for f in met_files], index=0, key="clu_metrics_file"
        )
        met_obj = next(x for x in met_files if x["name"] == sel_met)
        dfm = (
            load_parquet(repo, met_obj["path"], branch)
            if met_obj["name"].endswith(".parquet")
            else load_csv(repo, met_obj["path"], branch)
        )

        if "Ano" in dfm.columns:
            years_m = sorted([int(y) for y in dfm["Ano"].dropna().unique()])
            if years_m:
                year_m = st.select_slider(
                    "Ano (tabela)", options=years_m, value=years_m[-1], key="clu_met_ano"
                )
                dfm = dfm[dfm["Ano"] == year_m]

        cl_cols = [c for c in dfm.columns if re.search(r"(?i)(cluster|estagio|label)", c)]
        if cl_cols:
            valores = sorted(dfm[cl_cols[0]].dropna().unique().tolist(), key=lambda x: str(x))
            cl_sel = st.multiselect(
                "Filtrar clusters", valores, default=None, key="clu_met_clusters"
            )
            if cl_sel:
                dfm = dfm[dfm[cl_cols[0]].isin(cl_sel)]

        st.dataframe(dfm, use_container_width=True)
        download_df(dfm, f"metricas_{versao}")
    else:
        st.info(f"Não encontrei 'metricas.csv'/'metricas.parquet' em `{base_metrics}`.")

    st.divider()

    # 2.5) Spearman (pares) — tabela + heatmap
    st.subheader("Spearman (pares) — tabela e heatmap")
    base_sp = pick_existing_dir(
        repo, branch, ["Data/analises/original", "Data/Analises/original", "data/analises/original"]
    )
    df_sp = None
    sp_candidates = [
        f
        for f in list_files(repo, base_sp, branch, (".csv", ".parquet"))
        if ("spearman" in f["name"].lower()) and ("pairs" in f["name"].lower())
    ]
    if not sp_candidates:
        base_sp_alt = pick_existing_dir(
            repo,
            branch,
            ["Data/analises/winsorizados", "Data/Analises/winsorizados", "data/analises/winsorizados"],
        )
        sp_candidates = [
            f
            for f in list_files(repo, base_sp_alt, branch, (".csv", ".parquet"))
            if ("spearman" in f["name"].lower()) and ("pairs" in f["name"].lower())
        ]

    if sp_candidates:
        sp_candidates = sorted(sp_candidates, key=lambda x: x["name"])
        sp_sel = st.selectbox(
            "Selecione arquivo Spearman (pares)", [f["name"] for f in sp_candidates], index=0, key="clu_spearman_sel"
        )
        sp_obj = next(x for x in sp_candidates if x["name"] == sp_sel)
        df_sp = (
            load_parquet(repo, sp_obj["path"], branch)
            if sp_obj["name"].endswith(".parquet")
            else load_csv(repo, sp_obj["path"], branch)
        )
        spearman_title = sp_obj["name"]

    if df_sp is not None:
        st.markdown(f"**Tabela — {spearman_title}**")
        st.dataframe(df_sp, use_container_width=True)
        download_df(df_sp, "spearman_pairs")

        import re as _re

        cand_i = next((c for c in df_sp.columns if _re.search(r"(var|col).*a$", c.lower())), None)
        cand_j = next((c for c in df_sp.columns if _re.search(r"(var|col).*b$", c.lower())), None)
        cand_r = next(
            (
                c
                for c in df_sp.columns
                if any(k in c.lower() for k in ["rho", "spearman", "corr", "coef"])
            ),
            None,
        )
        if not (cand_i and cand_j and cand_r):
            text_cols = [c for c in df_sp.columns if not pd.api.types.is_numeric_dtype(df_sp[c])]
            num_cols = [c for c in df_sp.columns if pd.api.types.is_numeric_dtype(df_sp[c])]
            if len(text_cols) >= 2 and num_cols:
                cand_i, cand_j, cand_r = text_cols[0], text_cols[1], num_cols[0]

        if cand_i and cand_j and cand_r:
            def _pairs_to_matrix(df_pairs, i_col, j_col, val_col):
                M = df_pairs.pivot_table(index=i_col, columns=j_col, values=val_col, aggfunc="mean")
                M = M.combine_first(M.T)
                M = np.maximum(M, M.T)
                return M

            M = _pairs_to_matrix(df_sp, cand_i, cand_j, cand_r)
            if M is not None and not M.empty:
                fig = px.imshow(
                    M,
                    text_auto=False,
                    color_continuous_scale="Inferno",
                    title="Heatmap — Spearman (pares → matriz)",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Não consegui identificar as colunas para montar o heatmap automaticamente.")
    else:
        st.info("Arquivo de Spearman (pares) não encontrado nas pastas de análise.")

# -----------------------------------------------------------------------------
# ABA 3 — Univariadas (somente leitura/exibição)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Seleção de versão e tipo de análise")
    versao_u = st.radio(
        "Versão", ["originais", "winsorizados"], index=0, horizontal=True, key="uni_ver"
    )
    base_u = pick_existing_dir(
        repo,
        branch,
        ["Data/analises/original", "Data/analises/Original"]
        if versao_u == "originais"
        else ["Data/analises/winsorizados", "Data/analises/Winsorizados"],
    )
    analise_tipo = st.selectbox(
        "Tipo de análise",
        ["chi2", "spearman", "pearson", "ttest", "pairwise", "univariadas", "correlacao_matriz"],
        key="uni_tipo",
    )

    padroes = {
        "chi2": (r"chi", r"chi2"),
        "spearman": (r"spearman",),
        "pearson": (r"pearson", r"correl"),
        "ttest": (r"ttest", r"t-test"),
        "pairwise": (r"pairwise",),
        "univariadas": (r"univariad", r"descri", r"summary"),
        "correlacao_matriz": (
            r"corr(_|.*)matrix",
            r"correlation(_|.*)matrix",
            r"correlacao.*matriz",
            r"pearson.*matrix",
            r"spearman.*matrix",
        ),
    }
    found = find_files_by_patterns(repo, branch, [base_u], patterns=padroes.get(analise_tipo, ()))
    if not found:
        st.info(f"Nenhum arquivo encontrado em `{base_u}` para {analise_tipo}.")
    else:
        sel_file = st.selectbox("Arquivo", [f["name"] for f in found], key="uni_file")
        fobj = next(x for x in found if x["name"] == sel_file)
        df_any = load_tabular(repo, fobj["path"], branch)
        st.dataframe(df_any, use_container_width=True)
        download_df(df_any, f"{analise_tipo}_{versao_u}")

        if analise_tipo in ("spearman", "pearson", "correlacao_matriz"):
            if (df_any.shape[0] == df_any.shape[1]) and (
                set(df_any.columns) == set(df_any.index.astype(str))
            ):
                fig = px.imshow(
                    df_any, text_auto=False, color_continuous_scale="Inferno", title=f"Heatmap — {analise_tipo}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                cand_i = next((c for c in df_any.columns if re.search(r"(var|col).*a", c.lower())), None)
                cand_j = next((c for c in df_any.columns if re.search(r"(var|col).*b", c.lower())), None)
                cand_r = next(
                    (
                        c
                        for c in df_any.columns
                        if any(k in c.lower() for k in ["rho", "pearson", "spearman", "corr", "coef"])
                    ),
                    None,
                )
                if cand_i and cand_j and cand_r:
                    M = pairs_to_matrix(df_any, cand_i, cand_j, cand_r, sym_max=True)
                    if M is not None:
                        fig = px.imshow(
                            M,
                            text_auto=False,
                            color_continuous_scale="Inferno",
                            title=f"Heatmap — {analise_tipo} (pares → matriz)",
                        )
                        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# ABA 4 — PCA (em arquivo separado)
# -----------------------------------------------------------------------------
with tab4:
    render_pca_tab_inline(
        repo=repo,
        branch=branch,
        pick_existing_dir=pick_existing_dir,
        list_files=list_files,
        load_parquet=load_parquet,
        load_csv=load_csv,
    )




