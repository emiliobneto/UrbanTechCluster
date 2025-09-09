from __future__ import annotations
import streamlit as st
import os, io, json, tempfile
from typing import Iterable, Sequence, Dict, List, Tuple,  Any
import requests
import streamlit as st
import pandas as pd
import re, ast, unicodedata
import numpy as np
import plotly.express as px

# ===== Configuração de página — declarar UMA única vez =====
st.set_page_config(page_title="UrbanTechCluster — Consolidade", layout="wide")

def main():
    st.title("UrbanTechCluster — Visualização consolidada")

    # Sidebar — seleção de origem GitHub/local
    with st.sidebar:
        st.header("🔗 Dados")
        repo_in = st.text_input(
            "owner/repo (opcional se arquivos estiverem locais)",
            value="emiliobneto/UrbanTechCluster",
        )
        branch_in = st.text_input("branch (vazio = default do repo)", value="")
        if st.button("🧹 Limpar cache"):
            st.cache_data.clear(); st.cache_resource.clear()
            st.success("Caches limpos — recarregue a página.")

    repo_raw = repo_in.strip()
    branch_in = branch_in.strip()
    
    repo = ""
    if repo_raw:
        try:
            repo = normalize_repo(repo_raw)
        except Exception as e:
            st.error(f"Repo inválido: {e}")
            repo = ""
    
    branch = resolve_branch(repo, branch_in) if repo else "main"
    if repo:
        st.caption(f"Usando: **{repo}@{branch}** (prioriza arquivos locais; GitHub é fallback).")


    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Principal", "🧬 Clusterização", "📊 Univariadas", "🧠 ML → PCA", "🤖 Clusterizador",
    ])

    with tab1:
        render_tab_principal(repo or "", branch)
    with tab2:
        render_tab_clusterizacao(repo or "", branch)
    with tab3:
        render_tab_univariadas(repo or "", branch)
    with tab4:
        render_tab_pca(repo or "", branch)
    with tab5:
        render_tab_clusterizador(repo or "", branch)

# geopandas é opcional — só exigimos ao ler GPKG
try:
    import geopandas as gpd  # type: ignore
except Exception:
    gpd = None

API_BASE = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"


# -------------------- secrets / headers --------------------

def read_secret(path: Iterable[str], default=None):
    cur = st.secrets
    try:
        for p in path:
            cur = cur[p]
        return cur
    except Exception:
        return default


def github_headers() -> Dict[str, str]:
    token = read_secret(["github", "token"], None)
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "UTClean/1.0",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


# -------------------- helpers de repo/branch --------------------

def normalize_repo(owner_repo: str) -> str:
    s = (owner_repo or "").strip()
    s = s.replace("https://github.com/", "").replace("http://github.com/", "").strip("/")
    parts = [p for p in s.split("/") if p]
    if len(parts) < 2:
        raise RuntimeError("Informe no formato owner/repo (ex.: emiliobneto/UrbanTechCluster).")
    return f"{parts[0]}/{parts[1]}"


@st.cache_data(show_spinner=False, ttl=600)
def resolve_branch(owner_repo: str, branch: str | None) -> str:
    if branch:
        return branch
    try:
        owner_repo = normalize_repo(owner_repo)
        r = requests.get(f"{API_BASE}/repos/{owner_repo}", headers=github_headers(), timeout=10)
        if r.status_code == 200:
            return r.json().get("default_branch", "main")
    except Exception:
        pass
    return "main"


# -------------------- Local + GitHub I/O --------------------

def _local_bytes(rel_path: str) -> bytes | None:
    try:
        p = rel_path if os.path.isabs(rel_path) else os.path.join(os.getcwd(), rel_path)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                return f.read()
    except Exception:
        pass
    return None


def build_raw_url(owner_repo: str, path: str, branch: str) -> str:
    owner_repo = normalize_repo(owner_repo)
    return f"{RAW_BASE}/{owner_repo}/{branch}/{path.lstrip('/')}"


def fetch_bytes(owner_repo: str, path: str, branch: str) -> bytes:
    # 1) local
    data = _local_bytes(path)
    if data is not None:
        return data

    # 2) API raw (privado com token)
    try:
        owner_repo = normalize_repo(owner_repo)
        url = f"{API_BASE}/repos/{owner_repo}/contents/{path}?ref={branch}"
        r = requests.get(url, headers={**github_headers(), "Accept": "application/vnd.github.v3.raw"}, timeout=15)
        if r.status_code == 200:
            data = r.content
        else:
            data = None
    except Exception:
        data = None

    # 3) raw.githubusercontent (público)
    if data is None:
        try:
            raw_url = build_raw_url(owner_repo, path, branch)
            r = requests.get(raw_url, headers=github_headers(), timeout=15)
            if r.status_code == 200:
                data = r.content
        except Exception:
            data = None

    if not data:
        raise RuntimeError(f"Não consegui ler '{path}'. Verifique se existe localmente ou no GitHub (repo/branch/perm).")

    # sanity checks
    head = data[:200].strip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise RuntimeError("Recebi HTML em vez do arquivo. Repo privado sem token ou rate limit.")
    if data.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError("Arquivo está em Git LFS (ponteiro). Baixe-o localmente ou use token.")

    return data


@st.cache_data(show_spinner=False, ttl=600)
def list_files(owner_repo: str, path: str, branch: str, exts: Sequence[str] = (".csv", ".parquet", ".gpkg")) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []

    # local
    local_dir = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
    if os.path.isdir(local_dir):
        try:
            for nm in sorted(os.listdir(local_dir)):
                full = os.path.join(local_dir, nm)
                if os.path.isfile(full) and any(nm.lower().endswith(e) for e in exts):
                    out.append({"name": nm, "path": f"{path.rstrip('/')}/{nm}"})
            return out
        except Exception:
            pass

    # GitHub
    try:
        owner_repo = normalize_repo(owner_repo)
        url = f"{API_BASE}/repos/{owner_repo}/contents/{path}?ref={branch}"
        r = requests.get(url, headers=github_headers(), timeout=15)
        if r.status_code == 200 and isinstance(r.json(), list):
            for it in r.json():
                if it.get("type") == "file":
                    nm = it.get("name", "")
                    if any(nm.lower().endswith(e) for e in exts):
                        out.append({"name": nm, "path": f"{path.rstrip('/')}/{nm}"})
    except Exception:
        pass

    return out


# -------------------- loaders cacheados --------------------

@st.cache_data(show_spinner=True, ttl=600)
def load_csv(owner_repo: str, path: str, branch: str) -> pd.DataFrame:
    b = fetch_bytes(owner_repo, path, branch)
    return pd.read_csv(io.BytesIO(b), low_memory=False, usecols=lambda c: not str(c).startswith("Unnamed"))


@st.cache_data(show_spinner=True, ttl=600)
def load_parquet(owner_repo: str, path: str, branch: str) -> pd.DataFrame:
    b = fetch_bytes(owner_repo, path, branch)
    return pd.read_parquet(io.BytesIO(b), engine="pyarrow")


@st.cache_data(show_spinner=True, ttl=1200)
def load_gpkg(owner_repo: str, path: str, branch: str, layer: str | None = None):
    if gpd is None:
        raise RuntimeError("Instale geopandas e pyogrio para ler GPKG (pip install geopandas pyogrio).")
    b = fetch_bytes(owner_repo, path, branch)
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
        tmp.write(b); tmp.flush()
        tmp_path = tmp.name
    try:
        try:
            return gpd.read_file(tmp_path, layer=layer, engine="pyogrio")
        except Exception:
            return gpd.read_file(tmp_path, layer=layer)
    finally:
        try: os.remove(tmp_path)
        except Exception: pass

# Paleta categórica consistente
CATEGORICAL = [
    "#7c3aed", "#d946ef", "#fb7185", "#f97316", "#f59e0b",
    "#10b981", "#22d3ee", "#60a5fa", "#34d399", "#f43f5e",
]

# Paleta fixa pedida (cores do mock)
PALETA_FIXA = {
    "area_verde_mata": "#419E5F",
    "rios_corpos_dagua": "#5EA3BD",
    "0": "#E1DB8D",
    "1": "#E0B451",
    "2": "#E1683F",
    "3": "#8F3743",
    }
CLUSTER_COLOR_NA = "#BFBFBF"

WINSOR_FLAGS = ("winso", "winsor", "winsoriz", "_wins", "wins_")
VALID_EXTS = {".parquet", ".csv"}


def is_winsorized(filename: str) -> bool:
    name = filename.lower()
    return any(flag in name for flag in WINSOR_FLAGS)


def sanitize_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for col in df2.columns:
        if pd.api.types.is_object_dtype(df2[col]):
            df2[col] = df2[col].astype(str).str.replace(r"^=", "'=", regex=True)
    return df2


def is_categorical(series: pd.Series) -> bool:
    if series.dtype.kind in ("O", "b", "M", "m", "U", "S"):
        return True
    return series.dropna().nunique() <= 12


def pick_categorical(k: int):
    if k <= len(CATEGORICAL):
        return CATEGORICAL[:k]
    reps = (k // len(CATEGORICAL)) + 1
    return (CATEGORICAL * reps)[:k]


def hex_to_rgba(hex_color: str, alpha: int = 185) -> List[int]:
    try:
        h = hex_color.strip().lstrip("#")
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
        return [r, g, b, int(alpha)]
    except Exception:
        return [153, 153, 153, int(alpha)]


def px_seq(k: int) -> List[str]:
    base = px.colors.sequential.Viridis
    if k >= len(base):
        return base
    idxs = np.linspace(0, len(base) - 1, k).round().astype(int).tolist()
    return [base[i] for i in idxs]


def norm_sq_series(s: pd.Series, digits: int = 6) -> pd.Series:
    s = s.astype("string").str.replace(r"\D", "", regex=True).fillna("")
    s = s.str[-digits:].str.zfill(digits)
    return s.mask(s.eq(""))


def norm_sq_scalar(x, digits: int = 6):
    s = re.sub(r"\D", "", str(x)) if x is not None else ""
    if not s:
        return None
    if len(s) > digits:
        s = s[-digits:]
    return s.zfill(digits)


def classify_numeric(series: pd.Series, k: int = 6):
    s = pd.to_numeric(series, errors="coerce")
    s_no_na = s.dropna()
    if s_no_na.empty:
        return pd.Series(index=series.index, dtype="Int64"), np.array([])
    k = int(max(2, min(9, k)))
    try:
        labels = list(range(k))
        cats, bins = pd.qcut(s_no_na, q=k, labels=labels, retbins=True, duplicates="drop")
        cats = pd.Series(cats, index=s_no_na.index).astype("float").astype("Int64")
    except Exception:
        vmin, vmax = float(s_no_na.min()), float(s_no_na.max())
        if vmin == vmax:
            cats = pd.Series([0] * len(s_no_na), index=s_no_na.index, dtype="Int64")
            bins = np.array([vmin, vmax])
        else:
            bins = np.linspace(vmin, vmax, num=k + 1)
            idx = np.digitize(s_no_na, bins[1:-1], right=True)
            cats = pd.Series(idx, index=s_no_na.index, dtype="Int64")
    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    out.loc[cats.index] = cats
    return out, bins

def _norm_txt(x: str) -> str:
    """minúsculas + sem acentos, para reconhecer nomes de colunas."""
    s = unicodedata.normalize("NFKD", str(x))
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()

def _tidy_metrics_shape(df: pd.DataFrame):
    """
    Detecta colunas (cluster, ano, variavel, metrica, valor). Se as métricas vierem em 'wide',
    transforma para 'long' (metric, value).
    """
    cols_norm = {_norm_txt(c): c for c in df.columns}

    cl_col   = next((cols_norm[k] for k in cols_norm if ("estagio" in k or "cluster" in k or k.endswith("label"))), None)
    year_col = next((cols_norm[k] for k in cols_norm if k in {"ano", "year"}), None)
    var_col  = next((cols_norm[k] for k in cols_norm if any(t in k for t in ["variavel","variable","feature","indicador","coluna"])), None)
    met_col  = next((cols_norm[k] for k in cols_norm if any(t in k for t in ["metrica","metric","estatistica","stat"])), None)
    val_col  = next((cols_norm[k] for k in cols_norm if k in {"valor","value","val"}), None)

    df2 = df.copy()

    # Caso já esteja em "long" (tem metrica + valor)
    if met_col and val_col:
        need = [c for c in [cl_col, year_col, var_col, met_col, val_col] if c]
        return df2[need], {"cluster": cl_col, "year": year_col, "var": var_col, "metric": met_col, "value": val_col}

    # Caso "wide": derrete colunas numéricas (exceto id)
    id_cols = [c for c in [cl_col, year_col, var_col] if c]
    metric_cols = [c for c in df2.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df2[c])]
    if not metric_cols:  # fallback
        metric_cols = [c for c in df2.columns if c not in id_cols]

    long = df2.melt(id_vars=id_cols, value_vars=metric_cols, var_name="metric", value_name="value")
    return long, {"cluster": cl_col, "year": year_col, "var": var_col, "metric": "metric", "value": "value"}

def _load_metrics(repo: str, branch: str, winsor: bool):
    """
    Lê Parquet (preferido, mais leve). Se não houver, tenta CSV.
    Bases:
      - original  -> Data/analises/original/metricas.parquet|csv
      - winsoriz. -> Data/analises/winsorizados/metricas.parquet|csv
    """
    base = "Data/analises/winsorizados" if winsor else "Data/analises/original"
    # 1) tenta nomes padrão
    try:
        return _tidy_metrics_shape(load_parquet(repo, f"{base}/metricas.parquet", branch))
    except Exception:
        pass
    try:
        return _tidy_metrics_shape(load_csv(repo, f"{base}/metricas.csv", branch))
    except Exception:
        pass
    # 2) procura qualquer arquivo "metrica*"
    files = list_files(repo, base, branch, (".parquet", ".csv"))
    cand = [f for f in files if "metrica" in f["name"].lower()]
    if not cand:
        raise RuntimeError("Arquivos de métricas não encontrados.")
    # prefere parquet
    f = next((x for x in cand if x["name"].lower().endswith(".parquet")), cand[0])
    df = load_parquet(repo, f["path"], branch) if f["name"].lower().endswith(".parquet") else load_csv(repo, f["path"], branch)
    return _tidy_metrics_shape(df)

# -------- PCA helpers (reutilizados) --------

def classify_pca_file(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    if "explained_variance_ratio" in cols or any("variancia" in c and "explic" in c for c in cols):
        return "evr"
    if ("component" in cols and any(x in cols for x in ["loading", "valor", "carga"])) or any(
        str(c).lower().startswith("pc") for c in df.columns
    ):
        if "component" in cols and any(x in cols for x in ["loading", "valor", "carga"]):
            return "loadings_long"
        pc_like = [c for c in df.columns if str(c).lower().startswith("pc")]
        return "loadings_wide" if len(pc_like) >= 2 else "unknown"
    pc_cols = [c for c in df.columns if str(c).lower().startswith("pc")]
    if pc_cols:
        return "scores"
    return "unknown"


def tidy_loadings(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    # long
    if "component" in cols and any(k in cols for k in ["loading", "valor", "carga"]):
        comp = cols.get("component")
        load = cols.get("loading") or cols.get("valor") or cols.get("carga")
        non_num = [c for c in df.columns if c not in (comp, load) and not pd.api.types.is_numeric_dtype(df[c])]
        var_col = non_num[0] if non_num else comp
        out = df[[var_col, comp, load]].copy()
        out.columns = ["variable", "component", "loading"]
        return out
    # wide → melt PC*
    pc_cols = [c for c in df.columns if str(c).lower().startswith("pc")]
    if pc_cols:
        id_candidates = [c for c in df.columns if c not in pc_cols]
        id_col = id_candidates[0] if id_candidates else None
        long = df.melt(id_vars=[id_col] if id_col else None, value_vars=pc_cols, var_name="component", value_name="loading")
        long.columns = (["variable", "component", "loading"] if id_col else ["component", "loading"])
        if "variable" not in long.columns:
            long["variable"] = long.index.astype(str)
        return long[["variable", "component", "loading"]]
    return pd.DataFrame(columns=["variable", "component", "loading"])


def safe_literal_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    s = str(x).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return list(v)
    except Exception:
        pass
    s2 = s.strip("[]()")
    parts = [p.strip() for p in s2.split(",")]
    out = []
    for p in parts:
        if p:
            try:
                out.append(float(p))
            except Exception:
                pass
    return out

def try_load_clusters(repo: str, branch: str) -> pd.DataFrame | None:
    base = "Data/dados/Originais"
    files = list_files(repo, base, branch, (".csv", ".parquet"))
    cand = [f for f in files if f["name"].lower() in ("estagioclusterizacao.csv", "estagioclusterizacao.parquet")]
    if not cand:
        cand = [f for f in files if "estagio" in f["name"].lower() and "cluster" in f["name"].lower()]
    if not cand:
        return None
    f = cand[0]
    df = load_parquet(repo, f["path"], branch) if f["name"].lower().endswith(".parquet") else load_csv(repo, f["path"], branch)
    return df


# imports opcionais
try:
    import geopandas as gpd  # type: ignore
except Exception:
    gpd = None

try:
    import pydeck as pdk  # type: ignore
except Exception:
    pdk = None

def ensure_wgs84(gdf_in):
    try:
        if hasattr(gdf_in, "crs") and gdf_in.crs and str(gdf_in.crs).lower() not in ("epsg:4326", "wgs84"):
            return gdf_in.to_crs(4326)
    except Exception:
        pass
    return gdf_in


def make_geojson(gdf):
    if gpd is None:
        raise RuntimeError("geopandas é necessário para GeoJSON.")
    if "geometry" not in gdf.columns:
        raise RuntimeError("GeoDataFrame sem geometry.")
    gdf = ensure_wgs84(gdf)
    return json.loads(gdf.to_json())


def layer_geojson(geojson: Dict[str, Any], name: str = "layer"):
    if pdk is None:
        st.error("pydeck não instalado (pip install pydeck).")
        return None
    return pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        id=f"geojson-{re.sub(r'[^A-Za-z0-9_-]+', '-', str(name)).strip('-') or 'layer'}",
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        # Sem função JS: usa caminho dentro de properties
        get_fill_color="properties.fill_color",
        get_line_color=[80, 80, 80, 220],
        get_line_width=1,
        line_width_min_pixels=1,
        auto_highlight=True,
    )



def deck_osm(layers, view_state=None):
    if pdk is None:
        st.error("pydeck não instalado (pip install pydeck).")
        return
    tile = pdk.Layer(
        "TileLayer",
        data="https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
        opacity=0.5,
        )
    r = pdk.Deck(
        layers=[tile] + [l for l in layers if l is not None],
        initial_view_state=view_state or pdk.ViewState(latitude=-23.55, longitude=-46.63, zoom=10),
        map_style=None,
        tooltip={"text": "Valor: {properties.__value__}"},
    )
    st.pydeck_chart(r, use_container_width=True)


def add_lon_lat_from_geometry(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    if gdf is not None and "geometry" in gdf.columns:
        cent = gdf.geometry.centroid
        gdf = gdf.copy()
        gdf["lon"], gdf["lat"] = cent.x, cent.y
    return gdf

def render_tab_principal(repo: str, branch: str):
    st.subheader("🗺️ Principal — Quadras + Dados por SQ")

    # 1) Camada base (quadras)
    quadras_path = "Data/mapa/quadras.gpkg"
    try:
        gdf = load_gpkg(repo, quadras_path, branch)
    except Exception as e:
        st.error(f"Não consegui ler {quadras_path}: {e}")
        return

    gdf = ensure_wgs84(gdf)
    sq_geo = next((c for c in gdf.columns if str(c).upper() == "SQ"), None)
    if not sq_geo:
        st.error("Camada de quadras precisa ter coluna 'SQ'.")
        return

    # 2) Escolha de dados (originais/winsorize)
    colA, colB = st.columns([2, 1])
    with colA:
        src = st.radio("Origem dos dados", ["Originais", "winsorize"], horizontal=True, index=0, key="t1_src")
        base_dir = "Data/dados/Originais" if src == "Originais" else "Data/dados/winsorize"
        files = list_files(repo, base_dir, branch, exts=(".parquet", ".csv"))
        files = [f for f in files if f["name"].lower().endswith((".parquet", ".csv"))]
        if not files:
            st.warning(f"Nenhum .parquet/.csv em {base_dir} (local ou GitHub).")
            return
        sel_name = st.selectbox("Arquivo de dados", [f["name"] for f in files], key="t1_file")
        fobj = next((x for x in files if x["name"] == sel_name), None)
        if not fobj:
            st.error("Seleção inválida de arquivo.")
            return
        df = load_parquet(repo, fobj["path"], branch) if sel_name.lower().endswith(".parquet") else load_csv(repo, fobj["path"], branch)

    with colB:
        year_col = next((c for c in df.columns if str(c).lower() in ("ano", "year")), None)
        if year_col:
            anos = sorted(pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int).unique().tolist())
            if anos:
                ano_sel = st.select_slider("Ano", options=anos, value=anos[-1], key="t1_ano")
                df = df[pd.to_numeric(df[year_col], errors="coerce").astype("Int64") == ano_sel]
        id_like = {c for c in df.columns if str(c).lower() in {"sq", "id", "codigo", "code", "_sq_norm"}}
        time_like = {c for c in df.columns if str(c).lower() in {"ano", "year"}}
        candidates = [c for c in df.columns if c not in (id_like | time_like)]
        if not candidates:
            st.error("Não encontrei variáveis para mapear.")
            return
        var = st.selectbox("Variável a mapear", candidates, key="t1_var")

    # 3) JOIN por SQ normalizado
    df = df.copy()
    sq_df = next((c for c in df.columns if str(c).upper() == "SQ"), None)
    if not sq_df:
        st.error("Dataset selecionado não possui coluna 'SQ'.")
        return
    df["_SQ_norm"], gdf["_SQ_norm"] = norm_sq_series(df[sq_df]), norm_sq_series(gdf[sq_geo])

    gjoin = gdf[[sq_geo, gdf.geometry.name, "_SQ_norm"]].merge(
        df[["_SQ_norm", var]], on="_SQ_norm", how="left"
    )

    # 4) Pintura
    props_col = "__value__"
    legend: List[Tuple[str, Any, str]] = []  # (tipo, chave, cor)

    if is_categorical(df[var]):
        vals = gjoin[var].astype("string")
        cats_sorted = sorted([c for c in vals.dropna().unique()], key=lambda x: str(x))
        palette_map = {cats_sorted[i]: pick_categorical(len(cats_sorted))[i] for i in range(len(cats_sorted))}
        gjoin[props_col] = vals
        legend = [("cat", k, palette_map[k]) for k in cats_sorted]
    else:
        k = st.slider("Quebras (quantis)", 3, 9, 6, key="t1_k")
        labels, bins = classify_numeric(gjoin[var], k=k)
        gjoin[props_col] = labels
        pal = px_seq(k)
        palette_map = {i: pal[i] for i in range(len(pal))}
        legend = [("num", i, palette_map[i]) for i in range(len(pal))]

    gj = make_geojson(gjoin[[props_col, gjoin.geometry.name]].rename(columns={gjoin.geometry.name: "geometry"}))
    for feat in gj.get("features", []):
        v = feat.get("properties", {}).get(props_col, None)
        hexc = palette_map.get(v, "#999999")
        feat.setdefault("properties", {})["fill_color"] = hex_to_rgba(hexc)

    lyr = layer_geojson(gj, name="quadras")
    deck_osm([lyr])
    
    # 5) Legenda + Tabela e download
    st.markdown("**Legenda**")
    if not legend:
        st.caption("Sem classes definidas.")
    else:
        cols = st.columns(min(4, len(legend)))
        for i, (kind, k, hexc) in enumerate(legend):
            with cols[i % len(cols)]:
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px'>"
                    f"<span style='width:14px;height:14px;background:{hexc};display:inline-block;border-radius:3px'></span>"
                    f"<span>{('classe ' if kind=='num' else '')}{k}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    expo = gjoin[[sq_geo, var]].rename(columns={sq_geo: "SQ"})
    st.dataframe(sanitize_df_for_streamlit(expo.head(200)), use_container_width=True)
    csv = expo.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Baixar CSV (SQ + variável)", csv, file_name=f"dados_{var}.csv", mime="text/csv")

def render_tab_clusterizacao(repo: str, branch: str):
    st.subheader("🧬 Clusterização — mapa + resumo")

    # 1) Upload ou leitura automática
    colL, colR = st.columns([2, 1])
    with colL:
        up = st.file_uploader("Upload (opcional) EstagioClusterizacao.csv/parquet", type=["csv", "parquet"], key="t2_up")
    df_est, source = None, ""

    if up is not None:
        try:
            df_est = pd.read_parquet(up) if up.name.lower().endswith(".parquet") else pd.read_csv(up)
            source = f"(upload) {up.name}"
        except Exception as e:
            st.error(f"Falha ao ler upload: {e}")
            return
    else:
        base = "Data/dados/Originais"
        files = list_files(repo, base, branch, (".csv", ".parquet"))
        candidates = [
            f for f in files
            if re.search(r"(?i)est[aá]gio.*cluster", f["name"]) or f["name"].lower() in {"estagioclusterizacao.csv", "estagioclusterizacao.parquet"}
        ]
        if not candidates:
            st.warning("Não encontrei EstagioClusterizacao.* em Data/dados/Originais e nenhum upload foi feito.")
            return
        fobj = candidates[0]
        source = f"{base}/{fobj['name']}"
        df_est = load_parquet(repo, fobj["path"], branch) if fobj["name"].lower().endswith(".parquet") else load_csv(repo, fobj["path"], branch)

    # 2) Normalização básica
    sq_col = next((c for c in df_est.columns if str(c).upper() == "SQ"), None)
    if not sq_col:
        st.error("Arquivo de clusters precisa ter coluna 'SQ'.")
        return
    cand_cols = [c for c in df_est.columns if re.search(r"(?i)(cluster|est[aá]gio|label)", c)]
    if not cand_cols:
        st.error("Não encontrei coluna de cluster (ex.: EstagioClusterizacao, Cluster, Label).")
        return
    cl_col = next((c for c in cand_cols if c.lower() == "estagioclusterizacao"), cand_cols[0])

    ano_col = next((c for c in df_est.columns if str(c).lower() in ("ano", "year")), None)
    if ano_col:
        anos = sorted(pd.to_numeric(df_est[ano_col], errors="coerce").dropna().astype(int).unique().tolist())
        if anos:
            ano_sel = st.select_slider("Ano", options=anos, value=anos[-1], key="t2_ano")
            df_est = df_est[pd.to_numeric(df_est[ano_col], errors="coerce").astype("Int64") == ano_sel]

    df_est = df_est[[sq_col, cl_col]].copy()
    df_est["_SQ_norm"] = norm_sq_series(df_est[sq_col])
    df_est = df_est.dropna(subset=["_SQ_norm"])

    # 3) Quadras e join
    try:
        gdf = load_gpkg(repo, "Data/mapa/quadras.gpkg", branch)
    except Exception as e:
        st.error(f"Falha ao ler quadras.gpkg: {e}")
        return
    gdf = ensure_wgs84(gdf)
    sq_geo = next((c for c in gdf.columns if str(c).upper() == "SQ"), None)
    if not sq_geo:
        st.error("Camada de quadras precisa ter coluna 'SQ'.")
        return
    gdf["_SQ_norm"] = norm_sq_series(gdf[sq_geo])

    g = gdf[["geometry", "_SQ_norm"]].merge(df_est[["_SQ_norm", cl_col]], on="_SQ_norm", how="left")

    # 4) Cores por categoria — do código (0..3) vindo do arquivo EstagioClusterizacao.*
    cats = sorted(g[cl_col].dropna().astype(str).unique().tolist())
    
    def _color_for_code(val) -> str:
        v = str(val).strip()
        return CLUSTER_COLORS.get(v, CLUSTER_COLOR_NA)
    
    cmap = {c: _color_for_code(c) for c in cats}
    
    gj = make_geojson(g[[cl_col, "geometry"]])
    for feat in gj.get("features", []):
        v = feat.get("properties", {}).get(cl_col, None)
        hexc = _color_for_code(v)
        props = feat.setdefault("properties", {})
        props["fill_color"] = hex_to_rgba(hexc)
        props["__value__"] = v  # tooltip


    # 5) Resumo
    st.markdown("### Resumo")
    freq = g[cl_col].astype("string").value_counts(dropna=False).rename_axis("cluster").reset_index(name="n")
    st.dataframe(freq, use_container_width=True)
    fig = px.bar(freq[freq["cluster"].notna()], x="cluster", y="n", title="Contagem por cluster")
    st.plotly_chart(fig, use_container_width=True)
        # ====== Métricas por cluster e ano ======
    st.markdown("### Métricas por cluster e ano")

    use_wins = st.checkbox("Usar métricas winsorizadas", value=False, key="t2_metrics_wins")
    try:
        mdf, meta = _load_metrics(repo, branch, winsor=use_wins)
    except Exception as e:
        st.info(f"Não consegui carregar métricas ({'winsorizadas' if use_wins else 'originais'}): {e}")
        mdf, meta = None, None

    if isinstance(mdf, pd.DataFrame) and meta:
        clc, yrc, vrc, mtc, vlc = meta["cluster"], meta["year"], meta["var"], meta["metric"], meta["value"]

         # Seletores
        var_opts = sorted(mdf[vrc].dropna().astype(str).unique().tolist()) if vrc in mdf.columns else []
        met_opts = sorted(mdf[mtc].dropna().astype(str).unique().tolist()) if mtc in mdf.columns else []

        if vrc in mdf.columns and var_opts:
            vars_sel = st.multiselect("Variáveis", var_opts, default=var_opts[:1], key="t2_vars_sel")
        else:
            vars_sel = []

        # <<<<<< AQUI vira multiselect de Métricas >>>>>>
        if mtc in mdf.columns and met_opts:
            mets_sel = st.multiselect("Métricas", met_opts, default=met_opts[:1], key="t2_metric_sel_multi")
        else:
            mets_sel = []

        # Filtro
        dat = mdf.copy()
        if vars_sel and vrc in dat.columns:
            dat = dat[dat[vrc].astype(str).isin(vars_sel)]
        if mets_sel and mtc in dat.columns:
            dat = dat[dat[mtc].astype(str).isin([str(m) for m in mets_sel])]

        # Tabela cluster × (métrica, ano) – média se houver duplicatas
        if isinstance(dat, pd.DataFrame) and not dat.empty and clc in dat.columns:
            idx = [clc]
            # se há várias variáveis selecionadas, mantém variável no índice; se apenas uma, esconde para ficar compacto
            if vrc in dat.columns and (not vars_sel or len(vars_sel) != 1):
                idx.append(vrc)

            if yrc in dat.columns and mtc in dat.columns:
                piv = dat.pivot_table(index=idx, columns=[mtc, yrc], values=vlc, aggfunc="mean")
            elif mtc in dat.columns:
                piv = dat.pivot_table(index=idx, columns=mtc, values=vlc, aggfunc="mean")
            elif yrc in dat.columns:
                piv = dat.pivot_table(index=idx, columns=yrc, values=vlc, aggfunc="mean")
            else:
                piv = dat.groupby(idx, observed=True)[vlc].mean().to_frame("valor")

            # Flatten de colunas (métrica__ano)
            if isinstance(piv.columns, pd.MultiIndex):
                piv.columns = [f"{str(m)}__{str(a)}" for (m, a) in piv.columns]
            else:
                piv.columns = [str(c) for c in piv.columns]

            piv = piv.sort_index().reset_index()

            # arredonda para ficar legível
            num_cols = [c for c in piv.columns if c not in idx]
            piv[num_cols] = piv[num_cols].apply(pd.to_numeric, errors="coerce").round(4)

            st.dataframe(piv, use_container_width=True)
        else:
            st.caption("Sem dados de métricas com os filtros atuais.")

    else:
        st.caption("Arquivo(s) de métricas não disponíveis.")



def render_tab_univariadas(repo: str, branch: str):
    st.subheader("📊 Univariadas — distribuição e estatísticas")

    colA, colB = st.columns([2, 1])
    with colA:
        src = st.radio("Origem dos dados", ["Originais", "winsorize"], horizontal=True, index=0, key="t3_src")
        base_dir = "Data/dados/Originais" if src == "Originais" else "Data/dados/winsorize"
        files = list_files(repo, base_dir, branch, (".parquet", ".csv"))
        if not files:
            st.warning(f"Nenhum .parquet/.csv em {base_dir} (local ou GitHub).")
            return
        sel = st.selectbox("Arquivo", [f["name"] for f in files], key="t3_file")
        fobj = next((x for x in files if x["name"] == sel), None)
        if not fobj:
            st.error("Seleção inválida.")
            return
        df = load_parquet(repo, fobj["path"], branch) if sel.lower().endswith(".parquet") else load_csv(repo, fobj["path"], branch)

    with colB:
        year_col = next((c for c in df.columns if str(c).lower() in ("ano", "year")), None)
        if year_col:
            anos = sorted(pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int).unique().tolist())
            if anos:
                ano_sel = st.select_slider("Ano", options=anos, value=anos[-1], key="t3_ano")
                df = df[pd.to_numeric(df[year_col], errors="coerce").astype("Int64") == ano_sel]

    id_like = {c for c in df.columns if str(c).lower() in {"sq", "id", "codigo", "code"}}
    time_like = {c for c in df.columns if str(c).lower() in {"ano", "year"}}
    candidates = [c for c in df.columns if c not in (id_like | time_like)]
    if not candidates:
        st.warning("Não encontrei variáveis para analisar.")
        return
    var = st.selectbox("Variável", candidates, index=0, key="t3_var")

    join_clusters = st.checkbox("Juntar EstagioClusterizacao (opcional)", value=False, key="t3_joincl")
    df_cl, cl_col = None, None
    if join_clusters:
        df_cl = try_load_clusters(repo, branch)
        if df_cl is None:
            st.info("Arquivo EstagioClusterizacao.* não encontrado.")
        else:
            sq_a = next((c for c in df.columns if str(c).upper() == "SQ"), None)
            sq_b = next((c for c in df_cl.columns if str(c).upper() == "SQ"), None)
            if sq_a and sq_b:
                df["_SQ_norm"] = norm_sq_series(df[sq_a])
                df_cl["_SQ_norm"] = norm_sq_series(df_cl[sq_b])
                cl_cands = [c for c in df_cl.columns if ("estagio" in c.lower() or "cluster" in c.lower())]
                cl_col = cl_cands[0] if cl_cands else None
                if cl_col:
                    df = df.merge(df_cl[["_SQ_norm", cl_col]], on="_SQ_norm", how="left")

    st.markdown("### Distribuição")
    if is_categorical(df[var]):
        vc = df[var].astype("string").value_counts(dropna=False).rename_axis(var).reset_index(name="n")
        fig = px.bar(vc, x=var, y="n", title=f"Frequências — {var}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(vc, use_container_width=True)
    else:
        nb = st.slider("Bins (histograma)", 10, 80, 30, key="t3_bins")
        fig = px.histogram(df, x=var, nbins=nb, title=f"Histograma — {var}")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.box(df, y=var, points="outliers", title=f"Boxplot — {var}")
        st.plotly_chart(fig2, use_container_width=True)
        desc = df[[var]].describe().T
        st.markdown("### Estatísticas")
        st.dataframe(desc, use_container_width=True)
        if join_clusters and (cl_col in df.columns):
            st.markdown("### Por cluster (boxplot)")
            fig3 = px.box(df, x=cl_col, y=var, points="outliers", title=f"{var} por {cl_col}")
            st.plotly_chart(fig3, use_container_width=True)
            gb = df.groupby(cl_col, observed=True)[var].describe()
            st.dataframe(gb, use_container_width=True)

    csv = df[[c for c in [var, cl_col] if c in df.columns]].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Baixar CSV (variável + cluster se houver)", csv, file_name=f"univariada_{var}.csv", mime="text/csv")

def render_tab_pca(repo: str, branch: str):
    st.subheader("🧠 PCA — variância, loadings e scores (sem recálculo)")

    base = "Data/analises/PCA"
    files = list_files(repo, base, branch, (".csv", ".parquet"))
    if not files:
        st.info("Nenhum arquivo encontrado em Data/analises/PCA.")
        return

    names = [f["name"] for f in files]

    # 1) EVR
    st.markdown("### 1) Variância explicada (scree)")
    evr_name = st.selectbox("Arquivo de variância explicada", names, key="t4_evr")
    evr_obj = next((x for x in files if x["name"] == evr_name), None)
    df_evr = load_parquet(repo, evr_obj["path"], branch) if evr_name.lower().endswith(".parquet") else load_csv(repo, evr_obj["path"], branch)
    kind = classify_pca_file(df_evr)

    if kind == "evr":
        cols = {c.lower(): c for c in df_evr.columns}
        if "explained_variance_ratio" in cols:
            evr_col = cols["explained_variance_ratio"]
            dfp = df_evr[[evr_col]].copy()
            dfp["component"] = [f"PC{i+1}" for i in range(len(dfp))]
            dfp["explained_variance_ratio"] = pd.to_numeric(dfp[evr_col], errors="coerce")
        else:
            row = df_evr.iloc[0]
            arr = None
            for c in df_evr.columns:
                if "variancia" in c.lower() or "explained" in c.lower():
                    arr = safe_literal_list(row[c])
                    break
            if not arr:
                st.dataframe(df_evr.head(), use_container_width=True)
                st.warning("Não identifiquei a coluna de variância.")
                return
            dfp = pd.DataFrame({"component": [f"PC{i+1}" for i in range(len(arr))], "explained_variance_ratio": arr})
        dfp["cumulative"] = dfp["explained_variance_ratio"].cumsum()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(dfp, x="component", y="explained_variance_ratio", title="Scree — Variância explicada")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.line(dfp, x="component", y="cumulative", markers=True, title="Variância acumulada")
            st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(dfp, use_container_width=True)
    else:
        st.info("Arquivo selecionado não parece conter variância explicada.")
        st.dataframe(df_evr.head(), use_container_width=True)

    st.divider()

    # 2) Loadings
    st.markdown("### 2) Loadings")
    load_name = st.selectbox("Arquivo de loadings", names, index=min(1, len(names) - 1), key="t4_load")
    load_obj = next((x for x in files if x["name"] == load_name), None)
    df_load = load_parquet(repo, load_obj["path"], branch) if load_name.lower().endswith(".parquet") else load_csv(repo, load_obj["path"], branch)

    long = tidy_loadings(df_load)
    if long.empty:
        st.info("Não foi possível identificar estrutura de loadings.")
        st.dataframe(df_load.head(), use_container_width=True)
    else:
        comps = sorted(long["component"].astype(str).unique(), key=lambda x: (len(x), x))
        c1, c2 = st.columns([2, 1])
        with c1:
            comp = st.selectbox("Componente", comps, key="t4_comp")
        with c2:
            topn = st.slider("Top |loading|", 5, 30, 15, key="t4_topn")
        sub = long[long["component"].astype(str) == str(comp)].copy()
        sub["abs_loading"] = sub["loading"].abs()
        sub = sub.sort_values("abs_loading", ascending=False).head(topn)
        fig = px.bar(sub.sort_values("abs_loading"), x="abs_loading", y="variable", orientation="h", title=f"Maiores |loadings| — {comp}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sub.drop(columns=["abs_loading"]), use_container_width=True)

    st.divider()

    # 3) Scores
    st.markdown("### 3) Scores (dispersão)")
    score_name = st.selectbox("Arquivo de scores", names, index=min(2, len(names) - 1), key="t4_scores")
    sc_obj = next((x for x in files if x["name"] == score_name), None)
    df_sc = load_parquet(repo, sc_obj["path"], branch) if score_name.lower().endswith(".parquet") else load_csv(repo, sc_obj["path"], branch)

    pc_cols = [c for c in df_sc.columns if str(c).lower().startswith("pc")]
    if len(pc_cols) < 2:
        st.info("Arquivo de scores não possui pelo menos duas colunas PC*. Exibindo preview.")
        st.dataframe(df_sc.head(), use_container_width=True)
        return
    pcx = st.selectbox("PC eixo X", pc_cols, index=0, key="t4_pcx")
    pcy = st.selectbox("PC eixo Y", pc_cols, index=1, key="t4_pcy")
    color_col = st.selectbox("Colorir por (opcional)", ["(nenhum)"] + [c for c in df_sc.columns if c not in pc_cols], key="t4_color")
    color_kw = {} if color_col == "(nenhum)" else {"color": color_col}
    fig = px.scatter(df_sc, x=pcx, y=pcy, title=f"Scores — {pcx} × {pcy}", **color_kw)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_sc[[pcx, pcy] + ([] if color_col == "(nenhum)" else [color_col])].head(200), use_container_width=True)

def _list_subdirs(repo: str, base: str, branch: str) -> List[str]:
    out: List[str] = []
    local_dir = base if os.path.isabs(base) else os.path.join(os.getcwd(), base)
    if os.path.isdir(local_dir):
        try:
            for nm in sorted(os.listdir(local_dir)):
                full = os.path.join(local_dir, nm)
                if os.path.isdir(full):
                    out.append(nm)
            return out
        except Exception:
            pass
    # GitHub fallback
    import requests
    try:
        url = f"https://api.github.com/repos/{normalize_repo(repo)}/contents/{base}?ref={branch}"
        r = requests.get(url, headers={}, timeout=15)
        if r.status_code == 200 and isinstance(r.json(), list):
            for it in r.json():
                if it.get("type") == "dir":
                    out.append(it.get("name"))
    except Exception:
        pass
    return out


def _load_first_existing(repo: str, base: str, branch: str, names: List[str]) -> tuple[pd.DataFrame | None, str | None]:
    files = list_files(repo, base, branch, (".csv", ".parquet", ".json", ".txt"))
    low_map = {f["name"].lower(): f for f in files}
    for nm in names:
        f = low_map.get(nm.lower())
        if not f:
            continue
        try:
            if f["name"].lower().endswith(".parquet"):
                return load_parquet(repo, f["path"], branch), f["name"]
            if f["name"].lower().endswith(".csv"):
                return load_csv(repo, f["path"], branch), f["name"]
            if f["name"].lower().endswith(".json"):
                b = fetch_bytes(repo, f["path"], branch)
                return pd.DataFrame({"__raw_json__": [json.loads(b.decode("utf-8", errors="replace"))]}), f["name"]
            if f["name"].lower().endswith(".txt"):
                b = fetch_bytes(repo, f["path"], branch)
                return pd.DataFrame({"__raw__": [b.decode("utf-8", errors="replace")] }), f["name"]
        except Exception:
            continue
    return None, None


def render_tab_clusterizador(repo: str, branch: str):
    st.subheader("🤖 Clusterizador (ANN) — relatórios e mapas")

    ann_root = "Data/ANN"
    subdirs = _list_subdirs(repo, ann_root, branch)
    run_sel = st.selectbox("Execução (subpasta em Data/ANN)", options=((["(raiz)"] + subdirs) if subdirs else ["(raiz)"]), key="t5_run")
    base = ann_root if run_sel == "(raiz)" else f"{ann_root}/{run_sel}"
    st.caption(f"Lendo arquivos de: {base}")

    # 1) Histórico por época
    st.markdown("### 📈 Histórico por época")
    df_hist, hist_name = _load_first_existing(repo, base, branch, [
        "metrics_over_epochs.csv", "keras_history.csv", "val_metrics_per_epoch.csv",
    ])
    if isinstance(df_hist, pd.DataFrame) and not df_hist.empty and "__raw__" not in df_hist.columns:
        st.caption(f"Arquivo: {hist_name}")
        epoch_col = next((c for c in df_hist.columns if c.lower() == "epoch"), None)
        metric_cols = [c for c in df_hist.columns if any(k in c.lower() for k in ["loss", "acc", "auc", "precision", "recall", "f1", "mae", "mse", "rmse"])]
        if not metric_cols:
            st.info("Nenhuma métrica reconhecida.")
        else:
            for m in metric_cols:
                ycols = [m] + [c for c in df_hist.columns if c.lower() == f"val_{m.lower()}"]
                fig = px.line(df_hist, x=epoch_col if epoch_col else df_hist.index, y=ycols, markers=True, title=f"{m} por época")
                st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_hist.head(), use_container_width=True)
    else:
        st.info("Histórico não encontrado.")

    st.divider()

    # 2) AUC por classe
    st.markdown("### 📊 AUC por classe")
    df_auc, auc_name = _load_first_existing(repo, base, branch, ["auc_summary.csv", "roc_auc.csv"])
    if isinstance(df_auc, pd.DataFrame) and not df_auc.empty and "__raw__" not in df_auc.columns:
        cols = {c.lower(): c for c in df_auc.columns}
        class_col = cols.get("class") or cols.get("label") or list(df_auc.columns)[0]
        auc_col = cols.get("auc") or cols.get("roc_auc") or list(df_auc.columns)[1]
        fig = px.bar(df_auc, x=class_col, y=auc_col, title="AUC por classe")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_auc, use_container_width=True)
    else:
        st.info("AUC por classe não encontrado.")

    st.divider()

    # 3) Classification report
    st.markdown("### 🧾 Classification report")
    df_cr, cr_name = _load_first_existing(repo, base, branch, [
        "classificationreport.json", "classificationreport.txt", "classificationreport.csv",
    ])
    if isinstance(df_cr, pd.DataFrame) and not df_cr.empty:
        if "__raw_json__" in df_cr.columns:
            data = df_cr["__raw_json__"].iloc[0]
            try:
                df_show = pd.DataFrame(data).T.reset_index().rename(columns={"index": "label"})
            except Exception:
                df_show = pd.json_normalize(data)
        elif "__raw__" in df_cr.columns:
            lines = [l for l in df_cr["__raw__"].iloc[0].splitlines() if l.strip()]
            rows = []
            for ln in lines:
                parts = [p for p in ln.strip().split() if p]
                if len(parts) >= 5 and parts[0].lower() not in {"accuracy"}:
                    label = " ".join(parts[:-4])
                    prec, rec, f1, sup = parts[-4:]
                    def _float(x):
                        try:
                            return float(x)
                        except Exception:
                            return float('nan')
                    rows.append({"label": label, "precision": _float(prec), "recall": _float(rec), "f1": _float(f1), "support": _float(sup)})
                elif ln.lower().startswith("accuracy"):
                    toks = ln.split()
                    try:
                        acc = float(toks[1])
                        rows.append({"label": "accuracy", "f1": acc})
                    except Exception:
                        pass
            df_show = pd.DataFrame(rows) if rows else pd.DataFrame({"raw": [df_cr["__raw__"].iloc[0]]})
        else:
            df_show = df_cr.copy()
        st.dataframe(df_show, use_container_width=True)
        if "label" in df_show.columns and "f1" in df_show.columns:
            fig = px.bar(df_show[df_show["label"].str.lower() != "accuracy"], x="label", y="f1", title="F1 por classe")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Classification report não encontrado.")

    st.divider()

    # 4) Mapa de predições (opcional)
    st.markdown("### 🗺️ Mapa — classes previstas (ou estágios)")
    pred_df, pred_name = _load_first_existing(repo, base, branch, [
        "df25_com_previsoes.csv", "predictions_df25_with_meta.csv", "test_predictions_with_meta.csv",
    ])
    if isinstance(pred_df, pd.DataFrame) and not pred_df.empty and "__raw__" not in pred_df.columns:
        sq_col = next((c for c in pred_df.columns if str(c).upper() == "SQ"), None)
        if not sq_col:
            sq_col = next((c for c in pred_df.columns if str(c).lower() in ("id", "codigo", "code")), None)
        cat_candidates = [c for c in pred_df.columns if pred_df[c].dtype.kind in ("O", "U", "S") and c != sq_col and pred_df[c].nunique() <= 30]
        if not cat_candidates:
            st.info("Não encontrei coluna categórica adequada para mapear. Exibindo preview:")
            st.dataframe(pred_df.head(), use_container_width=True)
            return

        map_col = st.selectbox("Coluna a mapear", cat_candidates, key="t5_mapcol")

        # quadras
        try:
            gdf = load_gpkg(repo, "Data/mapa/quadras.gpkg", branch)
        except Exception as e:
            st.error(f"Falha ao ler quadras.gpkg: {e}")
            return
        gdf = ensure_wgs84(gdf)
        sq_geo = next((c for c in gdf.columns if str(c).upper() == "SQ"), None)
        if not sq_geo:
            st.error("Camada de quadras precisa ter coluna 'SQ'.")
            return

        # join
        pred_df = pred_df[[sq_col, map_col]].copy() if sq_col else pred_df[[map_col]].copy()
        if sq_col:
            pred_df["_SQ_norm"] = norm_sq_series(pred_df[sq_col])
            gdf["_SQ_norm"] = norm_sq_series(gdf[sq_geo])
            g = gdf.merge(pred_df[["_SQ_norm", map_col]], on="_SQ_norm", how="left")
        else:
            g = gdf.copy(); g[map_col] = pd.NA

        cats = sorted(g[map_col].dropna().astype(str).unique().tolist())
        pal = pick_categorical(len(cats))
        cmap = {cats[i]: pal[i] for i in range(len(cats))}
        cats = sorted(g[map_col].dropna().astype(str).unique().tolist())
        def _color_for_cluster(val: str) -> str:
            v = str(val).strip()
            if v.isdigit():
                return PALETA_FIXA.get(f"cluster_{int(v)}", PALETA_FIXA["nao_classificados"])
            return PALETA_FIXA["nao_classificados"]
        cmap = {c: _color_for_cluster(c) for c in cats}


        gj = make_geojson(g[[map_col, "geometry"]])
        for feat in gj.get("features", []):
            v = feat.get("properties", {}).get(map_col, None)
            hexc = cmap.get(str(v), "#999999")
            feat.setdefault("properties", {})["fill_color"] = hex_to_rgba(hexc)
        lyr = layer_geojson(gj, name=map_col)
        deck_osm([lyr])

        st.caption(f"Arquivo: {pred_name}")
        freq = g[map_col].astype("string").value_counts(dropna=False).rename_axis(map_col).reset_index(name="n")
        st.dataframe(freq, use_container_width=True)
    else:
        st.info("Arquivos de predição com meta não encontrados nesta execução.")

# --- entrypoint ---
if __name__ == "__main__":
    main()








