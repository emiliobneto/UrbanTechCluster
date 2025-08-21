import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data import list_files, load_parquet, load_csv, load_gpkg
from utils.stats_tools import chi2_between, corr_matrix, pairwise_ttests
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import requests
import itertools
from scipy import stats
from utils.maps import make_geojson, attach_fill_color, render_geojson_layer, render_line_layer, render_point_layer, deck, osm_basemap_deck
from utils.classify import jenks_breaks, is_categorical
from utils.colors import pick_sequential, pick_categorical
from utils.data import list_files, load_parquet, load_csv, load_gpkg
import mapclassify as mc
import pydeck as pdk
import json

from utils.colors import hex_to_rgb

st.set_page_config(
    page_title="MODELO DE REDE NEURAL ARTIFICIAL — Clusters SP",
    page_icon="🧠",
    layout="wide",
)

st.title("MODELO DE REDE NEURAL ARTIFICIAL PARA MAPEAMENTO DE CLUSTERS DE INTELIGÊNCIA E SUA APLICAÇÃO NO MUNICÍPIO DE SÃO PAULO")

with st.sidebar:
    st.header("🔗 Fonte dos Dados (GitHub)")
    st.session_state['gh_repo'] = st.text_input("owner/repo", value="emiliobneto/UrbanTechClusters")
    st.session_state['gh_branch'] = st.text_input("branch", value="main")
    st.caption("Listagem via GitHub Contents API e download via raw. Adicione `github.token` em secrets para evitar rate limit.")
    st.divider()
    st.header("🗺️ Mapbox (opcional)")
    st.caption("Defina `mapbox.token` em secrets para habilitar fundo satélite.")

st.markdown("Use as abas no menu: **Principal**, **Clusterização**, **Univariadas**, **ML → PCA**.")
st.info("Pastas esperadas no repositório: `Data/mapa`, `Data/dados/{originais,winsorize}`, `Data/analises`.")


try:
    import geopandas as gpd  # type: ignore
    _GPD = True
except Exception:
    gpd = None  # type: ignore
    _GPD = False

API_BASE = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"

def _secret(path: List[str], default=None):
    cur = st.secrets
    try:
        for p in path:
            cur = cur[p]
        return cur
    except Exception:
        return default

def _gh_headers():
    token = _secret(["github","token"], None)
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def build_raw_url(owner_repo: str, path: str, branch: str) -> str:
    owner_repo = owner_repo.strip("/")
    path = path.lstrip("/")
    return f"{RAW_BASE}/{owner_repo}/{branch}/{path}"

@st.cache_data(show_spinner=False)
def github_listdir(owner_repo: str, path: str, branch: str) -> List[Dict[str,Any]]:
    url = f"{API_BASE}/repos/{owner_repo}/contents/{path}?ref={branch}"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Falha listando {path}: {r.status_code} {r.text}")
    return r.json()

@dataclass
class GHFile:
    path: str
    type: str   # 'file'|'dir'
    name: str

def list_files(owner_repo: str, path: str, branch: str, exts: Tuple[str,...]) -> List[GHFile]:
    items = github_listdir(owner_repo, path, branch)
    out: List[GHFile] = []
    for it in items:
        if it.get("type") == "file":
            nm = it["name"]
            if any(nm.lower().endswith(e) for e in exts):
                out.append(GHFile(path=f"{path}/{nm}", type="file", name=nm))
    return out

@st.cache_data(show_spinner=True)
def load_parquet(owner_repo: str, path: str, branch: str) -> pd.DataFrame:
    url = build_raw_url(owner_repo, path, branch)
    return pd.read_parquet(url, engine="pyarrow")

@st.cache_data(show_spinner=True)
def load_csv(owner_repo: str, path: str, branch: str) -> pd.DataFrame:
    url = build_raw_url(owner_repo, path, branch)
    return pd.read_csv(url)

@st.cache_data(show_spinner=True)
def load_gpkg(owner_repo: str, path: str, branch: str, layer: Optional[str] = None):
    if not _GPD:
        raise RuntimeError("geopandas/pyogrio não disponíveis para ler GPKG.")
    url = build_raw_url(owner_repo, path, branch)
    content = requests.get(url, headers=_gh_headers(), timeout=120).content
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        return gpd.read_file(tmp.name, layer=layer)  # type: ignore

def load_any(owner_repo: str, path: str, branch: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".parquet"):
        return load_parquet(owner_repo, path, branch)
    if p.endswith(".csv"):
        return load_csv(owner_repo, path, branch)
    raise ValueError(f"Formato não suportado: {path}")

from __future__ import annotations
from typing import List, Tuple

def hex_to_rgb(hex_color: str) -> Tuple[int,int,int, int]:
    h = hex_color.lstrip('#')
    r,g,b = tuple(int(h[i:i+2], 16) for i in (0,2,4))
    return (r,g,b,180)

# Paletas sequenciais sem verde/azul (OrRd/Inferno-like)
SEQUENTIAL = {
    4: ['#fee8d8','#fdbb84','#fc8d59','#d7301f'],
    5: ['#feedde','#fdbe85','#fd8d3c','#e6550d','#a63603'],
    6: ['#feedde','#fdd0a2','#fdae6b','#fd8d3c','#e6550d','#a63603'],
    7: ['#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#e6550d','#a63603'],
    8: ['#fff5f0','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#99000d'],
}

# Categórica sem verde/azul (até ~20 classes)
CATEGORICAL = [
    '#7c3aed','#d946ef','#fb7185','#f97316','#f59e0b','#facc15','#fde047',
    '#a16207','#9a3412','#b91c1c','#ea580c','#be185d','#9333ea','#6b21a8',
    '#a21caf','#c026d3','#db2777','#e11d48','#eab308','#f43f5e'
]

def pick_sequential(n: int) -> List[str]:
    n = max(4, min(8, n))
    return SEQUENTIAL.get(n, SEQUENTIAL[6])

def pick_categorical(k: int) -> List[str]:
    if k <= len(CATEGORICAL):
        return CATEGORICAL[:k]
    reps = (k // len(CATEGORICAL)) + 1
    return (CATEGORICAL * reps)[:k]


def jenks_breaks(values: pd.Series, k: int) -> Tuple[List[float], pd.Series]:
    vals = values.dropna().astype(float).values
    if len(np.unique(vals)) < max(4, k):
        k = min(len(np.unique(vals)), max(2, k))
    nb = mc.NaturalBreaks(vals, k=k, initial=200)
    bins = [-float('inf')] + list(nb.bins)
    binned = pd.cut(values, bins=bins, labels=False, include_lowest=True)
    return bins, binned

def is_categorical(series: pd.Series) -> bool:
    if series.dtype.kind in ("O","b","M","m","U","S"):
        return True
    nun = series.dropna().nunique()
    return nun <= 12  # heurística


def ensure_wgs84(gdf):
    try:
        if hasattr(gdf, "crs") and gdf.crs and str(gdf.crs).lower() not in ("epsg:4326", "wgs84"):
            return gdf.to_crs(4326)
    except Exception:
        pass
    return gdf

def make_geojson(gdf) -> dict:
    gdf = ensure_wgs84(gdf)
    return json.loads(gdf.to_json())

def attach_fill_color(geojson_obj: dict, color_map: Dict[Any, str], prop: str = "value") -> dict:
    for feat in geojson_obj.get("features", []):
        val = feat.get("properties", {}).get(prop, None)
        hexc = color_map.get(val, "#999999")
        feat.setdefault("properties", {})["fill_color"] = hex_to_rgb(hexc)
    return geojson_obj

def render_geojson_layer(geojson_obj: dict, name: str = "Polygons") -> pdk.Layer:
    return pdk.Layer(
        "GeoJsonLayer",
        geojson_obj,
        pickable=True,
        stroked=False,
        filled=True,
        extruded=False,
        get_fill_color="properties.fill_color",
        get_line_color=[100,100,100],
        get_line_width=0.5,
        auto_highlight=True
    )

def render_line_layer(geojson_obj: dict, name: str = "Lines") -> pdk.Layer:
    return pdk.Layer(
        "GeoJsonLayer",
        geojson_obj,
        pickable=True,
        stroked=True,
        filled=False,
        get_line_color=[30,30,30],
        get_line_width=2
    )

def render_point_layer(geojson_obj: dict, name: str = "Points") -> pdk.Layer:
    return pdk.Layer(
        "GeoJsonLayer",
        geojson_obj,
        pickable=True,
        point_type="circle",
        get_fill_color=[60,60,60,220],
        get_radius=60,
    )

def deck(layers, satellite=False, initial_view_state=None):
    token = st.secrets.get("mapbox", {}).get("token", None)
    map_style = "mapbox://styles/mapbox/light-v11"
    if satellite:
        map_style = "mapbox://styles/mapbox/satellite-streets-v12"

    r = pdk.Deck(
        layers=layers,
        initial_view_state=initial_view_state or pdk.ViewState(latitude=-23.55, longitude=-46.63, zoom=10),
        map_style=map_style,
        api_keys={"mapbox": token} if token else None,
        tooltip={"text": "{name}\n{value}"}
    )
    st.pydeck_chart(r, use_container_width=True)

def osm_basemap_deck(layers, initial_view_state=None):
    tile = pdk.Layer(
        "TileLayer",
        data="https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
        min_zoom=0, max_zoom=19, tile_size=256,
    )
    r = pdk.Deck(
        layers=[tile] + layers,
        initial_view_state=initial_view_state or pdk.ViewState(latitude=-23.55, longitude=-46.63, zoom=10),
        map_style=None,
    )
    st.pydeck_chart(r, use_container_width=True)


def pairwise_ttests(df: pd.DataFrame, group_col: str, value_col: str, equal_var: bool = False) -> pd.DataFrame:
    groups = [g for g in df[group_col].dropna().unique()]
    rows = []
    for a,b in itertools.combinations(groups, 2):
        x = df.loc[df[group_col]==a, value_col].dropna().astype(float)
        y = df.loc[df[group_col]==b, value_col].dropna().astype(float)
        if len(x) >= 2 and len(y) >= 2:
            t, p = stats.ttest_ind(x, y, equal_var=equal_var, nan_policy='omit')
            rows.append({"grupo_a": a, "grupo_b": b, "t": float(t), "p_value": float(p)})
    return pd.DataFrame(rows)

def chi2_between(df: pd.DataFrame, col_a: str, col_b: str) -> Dict[str, Any]:
    tbl = pd.crosstab(df[col_a], df[col_b])
    chi2, p, dof, expected = stats.chi2_contingency(tbl, correction=False)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof), "table": tbl}

def corr_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    return num.corr(method=method).replace([np.inf, -np.inf], np.nan)



st.set_page_config(page_title="Principal", page_icon="🗺️", layout="wide")
st.title("🗺️ Aba Principal — Mapas e Recortes")

repo = st.session_state.get("gh_repo")
branch = st.session_state.get("gh_branch")
if not repo or not branch:
    st.warning("Configure o repositório e branch na página inicial.")
    st.stop()

# --- Camadas base ---
st.subheader("Quadras e camadas adicionais (GPKG)")
colA, colB = st.columns([2,1], gap="large")
with colA:
    st.caption("Carrega `Data/mapa/quadras.gpkg` e sobrepõe camadas auxiliares (`linhas`, `estações`, `água`, etc.).")
with colB:
    basemap = st.radio("Plano de fundo", ["OpenStreetMap", "Satélite (Mapbox)"], index=0)

try:
    gdf_quadras = load_gpkg(repo, "Data/mapa/quadras.gpkg", branch)
except Exception as e:
    st.error(f"Erro carregando Data/mapa/quadras.gpkg: {e}")
    st.stop()

# Outras camadas do diretório Data/mapa
try:
    mapa_files = list_files(repo, "Data/mapa", branch, (".gpkg",))
    other_layers = [f for f in mapa_files if f.name.lower() != "quadras.gpkg"]
    layer_names = [f.name for f in other_layers]
    sel_layers = st.multiselect("Camadas auxiliares (opcional)", layer_names, default=[])
    loaded_layers = []
    for nm in sel_layers:
        f = next(x for x in other_layers if x.name == nm)
        g = load_gpkg(repo, f.path, branch)
        loaded_layers.append((nm, g))
except Exception as e:
    st.warning(f"Não foi possível listar/ler camadas em Data/mapa: {e}")
    loaded_layers = []

# --- Dados por SQ ---
st.subheader("Dados por `SQ` para espacialização")
col1, col2, col3 = st.columns([1.6,1,1.2], gap="large")

with col1:
    src_type = st.radio("Origem dos dados", ["originais", "winsorize"], index=0, horizontal=True)
    base_path = f"Data/dados/{src_type}"
    parquet_files = list_files(repo, base_path, branch, (".parquet",))
    if not parquet_files:
        st.warning(f"Sem .parquet em {base_path}.")
        st.stop()
    sel_file = st.selectbox("Arquivo .parquet com variáveis", [f.name for f in parquet_files])
    file_obj = next(x for x in parquet_files if x.name == sel_file)
    df_vars = load_parquet(repo, file_obj.path, branch)

with col2:
    # coluna SQ
    join_col = next((c for c in df_vars.columns if c.upper()=="SQ" or c=="SQ"), None)
    if join_col is None:
        st.error("Dataset selecionado não possui coluna 'SQ'.")
        st.stop()
    # Ano
    years = sorted([int(y) for y in df_vars["Ano"].dropna().unique()]) if "Ano" in df_vars.columns else []
    if years:
        year = st.select_slider("Ano", options=years, value=years[-1])
        df_vars = df_vars[df_vars["Ano"]==year]
    else:
        year = None

with col3:
    var_options = [c for c in df_vars.columns if c not in (join_col, "Ano")]
    if not var_options:
        st.error("Nenhuma variável disponível além de SQ/Ano.")
        st.stop()
    var_sel = st.selectbox("Variável a mapear", var_options)
    n_classes = st.slider("Quebras (Jenks)", min_value=4, max_value=8, value=6)

# join
sq_col_quadras = "SQ" if "SQ" in gdf_quadras.columns else next((c for c in gdf_quadras.columns if c.upper()=="SQ"), None)
if not sq_col_quadras:
    st.error("Camada de quadras não possui coluna 'SQ'.")
    st.stop()

gdf = gdf_quadras.merge(df_vars[[join_col, var_sel]], left_on=sq_col_quadras, right_on=join_col, how="left")

# cores
series = gdf[var_sel]
if is_categorical(series):
    cats = [c for c in series.dropna().unique()]
    palette = pick_categorical(len(cats))
    cmap = {cat: palette[i] for i,cat in enumerate(cats)}
    gdf["value"] = series
else:
    _, binned = jenks_breaks(series, k=n_classes)
    gdf["value"] = binned
    k = len(set(b for b in binned.dropna().unique())) or n_classes
    palette = pick_sequential(k)
    cmap = {i: palette[i] for i in range(len(palette))}

geojson = make_geojson(gdf)
geojson = attach_fill_color(geojson, cmap, prop="value")

layers = [render_geojson_layer(geojson, name="quadras")]
for nm, g in loaded_layers:
    gj = make_geojson(g)
    try:
        geoms = set(g.geometry.geom_type.unique())
    except Exception:
        geoms = {"Polygon"}
    if geoms <= {"LineString","MultiLineString"}:
        layers.append(render_line_layer(gj, nm))
    elif geoms <= {"Point","MultiPoint"}:
        layers.append(render_point_layer(gj, nm))
    else:
        layers.append(render_geojson_layer(gj, nm))

st.markdown("#### Mapa — Quadras + Camadas auxiliares")
if basemap.startswith("Satélite"):
    deck(layers, satellite=True)
else:
    osm_basemap_deck(layers)

# --- Recortes ---
st.subheader("Recortes espaciais (GPKG)")
st.caption("Selecione um dos GPKGs em `Data/mapa/recortes` para visualizar.")
try:
    recorte_files = list_files(repo, "Data/mapa/recortes", branch, (".gpkg",))
    if not recorte_files:
        st.info("Pasta `Data/mapa/recortes` vazia.")
    else:
        rec_sel = st.selectbox("Arquivo de recorte", [f.name for f in recorte_files])
        rec_obj = next(x for x in recorte_files if x.name == rec_sel)
        gdf_rec = load_gpkg(repo, rec_obj.path, branch)
        gj_rec = make_geojson(gdf_rec)
        layers_rec = [render_geojson_layer(gj_rec, name="recorte")]
        st.markdown("#### Mapa — Recorte selecionado")
        if basemap.startswith("Satélite"):
            deck(layers_rec, satellite=True)
        else:
            osm_basemap_deck(layers_rec)
except Exception as e:
    st.warning(f"Não foi possível listar/ler recortes: {e}")

st.set_page_config(page_title="Clusterização", page_icon="🧬", layout="wide")
st.title("🧬 Aba Clusterização — Mapas e Métricas")

repo = st.session_state.get("gh_repo")
branch = st.session_state.get("gh_branch")
if not repo or not branch:
    st.warning("Configure o repositório e branch na página inicial.")
    st.stop()

# --- Mapa por EstagioClusterizacao ---
st.subheader("Mapa — EstagioClusterizacao (Data/dados/Originais)")
try:
    files_estagio = list_files(repo, "Data/dados/Originais", branch, (".parquet",))
    estagio_candidates = [f for f in files_estagio if "estagioclusterizacao" in f.name.lower() or f.name.lower().startswith("estagio")]
    if not estagio_candidates and files_estagio:
        estagio_candidates = [files_estagio[0]]
    if not estagio_candidates:
        st.error("Não encontrei parquet com EstagioClusterizacao em Data/dados/Originais.")
        st.stop()
    sel_estagio = st.selectbox("Arquivo EstagioClusterizacao", [f.name for f in estagio_candidates])
    est_file = next(x for x in estagio_candidates if x.name == sel_estagio)
    df_est = load_parquet(repo, est_file.path, branch)
except Exception as e:
    st.error(f"Erro carregando EstagioClusterizacao: {e}")
    st.stop()

years = sorted([int(y) for y in df_est["Ano"].dropna().unique()]) if "Ano" in df_est.columns else []
year = st.select_slider("Ano", options=years, value=years[-1]) if years else None
if year:
    df_est = df_est[df_est["Ano"] == year]

try:
    gdf_quadras = load_gpkg(repo, "Data/mapa/quadras.gpkg", branch)
except Exception as e:
    st.error(f"Erro carregando Data/mapa/quadras.gpkg: {e}")
    st.stop()

join_col_est = "SQ" if "SQ" in df_est.columns else None
join_col_quad = "SQ" if "SQ" in gdf_quadras.columns else None
if not (join_col_est and join_col_quad):
    st.error("É necessário que quadras e tabela de clusters possuam coluna 'SQ'.")
    st.stop()

cluster_cols = [c for c in df_est.columns if "cluster" in c.lower() or "estagio" in c.lower() or "label" in c.lower()]
if not cluster_cols:
    st.error("Não encontrei coluna de cluster (ex.: EstagioClusterizacao).")
    st.stop()
cluster_col = st.selectbox("Coluna de cluster", cluster_cols, index=0)

gdf = gdf_quadras.merge(df_est[[join_col_est, cluster_col]], left_on=join_col_quad, right_on=join_col_est, how="left")

cats = [c for c in gdf[cluster_col].dropna().unique()]
palette = pick_categorical(len(cats))
cmap = {cat: palette[i] for i,cat in enumerate(cats)}

gdf["value"] = gdf[cluster_col]
gj = make_geojson(gdf)
gj = attach_fill_color(gj, cmap, prop="value")

colA, colB = st.columns([1,1], gap="large")
with colA:
    st.markdown("#### Mapa — Clusters")
    base = st.radio("Plano de fundo", ["OpenStreetMap", "Satélite (Mapbox)"], index=0, horizontal=True)
    if base.startswith("Satélite"):
        deck([render_geojson_layer(gj, name="clusters")], satellite=True)
    else:
        osm_basemap_deck([render_geojson_layer(gj, name="clusters")])

with colB:
    st.markdown("#### Legenda")
    st.write(pd.DataFrame({"cluster": list(cmap.keys()), "cor": list(cmap.values())}))

# --- Métricas (originais / winsorizados) ---
st.subheader("Métricas por cluster/ano")

opt_vers = st.radio("Versão dos dados", ["originais", "winsorizados"], index=0, horizontal=True)
base_metrics = "Data/analises/original" if opt_vers=="originais" else "Data/analises/winsorizados"

try:
    files_metrics_csv = list_files(repo, base_metrics, branch, (".csv",))
    files_metrics_parq = list_files(repo, base_metrics, branch, (".parquet",))
    main_candidates = [f for f in files_metrics_parq+files_metrics_csv if "metrica" in f.name.lower() or "metrics" in f.name.lower()]
    files_all = main_candidates if main_candidates else (files_metrics_parq + files_metrics_csv)
    if not files_all:
        st.info(f"Sem arquivos de métricas em {base_metrics}.")
    else:
        sel_met = st.selectbox("Arquivo de métricas", [f.name for f in files_all])
        met_obj = next(x for x in files_all if x.name == sel_met)
        dfm = load_parquet(repo, met_obj.path, branch) if met_obj.name.endswith(".parquet") else load_csv(repo, met_obj.path, branch)

        cols_year = [c for c in dfm.columns if c.lower()=="ano"]
        cols_cluster = [c for c in dfm.columns if "cluster" in c.lower() or "estagio" in c.lower() or c.lower()=="label"]

        c1, c2 = st.columns(2)
        with c1:
            if cols_year:
                years_m = sorted([int(y) for y in dfm[cols_year[0]].dropna().unique()])
                year_m = st.select_slider("Ano (tabela)", options=years_m, value=years_m[-1])
                dfm = dfm[dfm[cols_year[0]]==year_m]
        with c2:
            if cols_cluster:
                cl_sel = st.multiselect("Clusters", sorted(dfm[cols_cluster[0]].dropna().unique().tolist()))
                if cl_sel:
                    dfm = dfm[dfm[cols_cluster[0]].isin(cl_sel)]
        st.dataframe(dfm, use_container_width=True)
except Exception as e:
    st.warning(f"Falha ao ler métricas: {e}")

# --- Spearman + t-tests ---
st.subheader("Associações (Spearman) e Testes t par-a-par")

try:
    spearman_file = "Data/analises/original/analise_clusters__spearman_pairs__20250820-180622.csv"
    df_spear = load_csv(repo, spearman_file, branch)
    st.markdown("**Spearman pairs (arquivo original)**")
    st.dataframe(df_spear, use_container_width=True)
except Exception as e:
    st.info(f"Spearman pairs não encontrado ou erro ao ler: {e}")

st.markdown("**Teste t par-a-par entre clusters**")
try:
    st.caption("Selecione um `.parquet` (originais/winsorize) e uma variável numérica; junta com clusters e calcula t-test entre pares.")
    src_type = st.radio("Origem", ["originais", "winsorize"], horizontal=True, index=0, key="tt_src")
    var_base = f"Data/dados/{src_type}"
    files_vars = list_files(repo, var_base, branch, (".parquet",))
    if files_vars:
        sel_vf = st.selectbox("Arquivo com variáveis", [f.name for f in files_vars], key="tt_file")
        vf_obj = next(x for x in files_vars if x.name == sel_vf)
        dfv = load_parquet(repo, vf_obj.path, branch)
        if year and "Ano" in dfv.columns:
            dfv = dfv[dfv["Ano"]==year]
        num_cols = [c for c in dfv.columns if c not in ("SQ","Ano") and pd.api.types.is_numeric_dtype(dfv[c])]
        if num_cols:
            vsel = st.selectbox("Variável numérica", num_cols, key="tt_var")
            dft = dfv.merge(df_est[["SQ", cluster_col]], on="SQ", how="inner")
            res = pairwise_ttests(dft, cluster_col, vsel)
            st.dataframe(res, use_container_width=True)
        else:
            st.info("Nenhuma variável numérica encontrada no arquivo selecionado.")
    else:
        st.info(f"Sem arquivos em {var_base}.")
except Exception as e:
    st.warning(f"Erro nos testes t: {e}")


st.set_page_config(page_title="Univariadas", page_icon="📊", layout="wide")
st.title("📊 Aba Univariadas — Testes e Correlações")

repo = st.session_state.get("gh_repo")
branch = st.session_state.get("gh_branch")
if not repo or not branch:
    st.warning("Configure o repositório e branch na página inicial.")
    st.stop()

st.subheader("Seleção de dataset")
src_type = st.radio("Origem", ["originais", "winsorizados"], index=0, horizontal=True)
base = f"Data/dados/{src_type}"

files_pq = list_files(repo, base, branch, (".parquet",))
files_csv = list_files(repo, base, branch, (".csv",))
files_all = files_pq + files_csv
if not files_all:
    st.error(f"Sem arquivos em {base}.")
    st.stop()

sel_file = st.selectbox("Arquivo de dados", [f.name for f in files_all])
fobj = next(x for x in files_all if x.name == sel_file)
df = load_parquet(repo, fobj.path, branch) if fobj.name.endswith(".parquet") else load_csv(repo, fobj.path, branch)

years = sorted([int(y) for y in df["Ano"].dropna().unique()]) if "Ano" in df.columns else []
if years:
    yr = st.select_slider("Ano", options=years, value=years[-1])
    df = df[df["Ano"]==yr]

exclude = {"SQ","Ano"}
num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])]

st.markdown("### Distribuições")
c1, c2 = st.columns(2)
with c1:
    sel_num = st.multiselect("Variáveis numéricas", num_cols, default=num_cols[:3] if len(num_cols)>=3 else num_cols)
    for col in sel_num:
        fig = px.histogram(df, x=col, nbins=50, title=f"Distribuição — {col}")
        st.plotly_chart(fig, use_container_width=True)

with c2:
    if cat_cols:
        sel_cat = st.multiselect("Variáveis categóricas (contagem)", cat_cols, default=cat_cols[:1])
        for col in sel_cat:
            vc = df[col].value_counts(dropna=False).reset_index()
            vc.columns = [col, "count"]
            fig = px.bar(vc, x=col, y="count", title=f"Frequência — {col}")
            st.plotly_chart(fig, use_container_width=True)

st.markdown("### Correlações e Colinearidade")
met = st.radio("Método", ["pearson","spearman"], horizontal=True, index=0)
if num_cols:
    base_cols = sel_num if sel_num else num_cols
    cm = corr_matrix(df[base_cols], method=met)
    fig = px.imshow(cm, text_auto=False, color_continuous_scale="Inferno", title=f"Correlação ({met})")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sem variáveis numéricas para correlação.")

st.markdown("### Testes estatísticos")
cA, cB = st.columns(2)
with cA:
    st.write("**Chi-Quadrado (duas variáveis categóricas)**")
    if len(cat_cols) >= 2:
        a = st.selectbox("Categoria A", cat_cols, index=0, key="chi_a")
        b = st.selectbox("Categoria B", cat_cols, index=1 if len(cat_cols)>1 else 0, key="chi_b")
        res = chi2_between(df[[a,b]].dropna(), a, b)
        st.json({"chi2": res["chi2"], "p_value": res["p_value"], "dof": res["dof"]})
        st.dataframe(res["table"], use_container_width=True)
    else:
        st.info("Selecione um arquivo com pelo menos duas variáveis categóricas.")

with cB:
    st.write("**t-test par-a-par (grupo categórico × variável numérica)**")
    if cat_cols and num_cols:
        grp = st.selectbox("Grupo (categ.)", cat_cols, key="tt_grp")
        val = st.selectbox("Variável (num.)", num_cols, key="tt_val")
        dft = df[[grp, val]].dropna()
        out = pairwise_ttests(dft, grp, val)
        st.dataframe(out, use_container_width=True)
    else:
        st.info("Necessário ao menos 1 categórica e 1 numérica.")

import streamlit as st
import plotly.express as px

from utils.data import list_files, load_parquet, load_csv

st.set_page_config(page_title="ML → PCA", page_icon="🧠", layout="wide")
st.title("🧠 Aba ML → PCA — Variância Explicada")

repo = st.session_state.get("gh_repo")
branch = st.session_state.get("gh_branch")
if not repo or not branch:
    st.warning("Configure o repositório e branch na página inicial.")
    st.stop()

base = "Data/analises/PCA"
files_pq = list_files(repo, base, branch, (".parquet",))
files_csv = list_files(repo, base, branch, (".csv",))
files_all = files_pq + files_csv
if not files_all:
    st.info(f"Sem arquivos em {base}.")
    st.stop()

st.caption("Selecione um arquivo PCA com colunas como `component` e `explained_variance_ratio` (ou similares).")
sel = st.selectbox("Arquivo PCA", [f.name for f in files_all])
fobj = next(x for x in files_all if x.name == sel)
df = load_parquet(repo, fobj.path, branch) if fobj.name.endswith(".parquet") else load_csv(repo, fobj.path, branch)

comp_col = next((c for c in df.columns if c.lower().startswith("comp")), df.columns[0])
evr_col = next((c for c in df.columns if "explained" in c.lower() and "ratio" in c.lower()), None)
if evr_col is None:
    evr_col = next((c for c in df.columns if "variance" in c.lower() and "ratio" in c.lower()), None)
if evr_col is None:
    st.error("Não encontrei coluna com 'explained_variance_ratio'.")
    st.dataframe(df.head(), use_container_width=True)
    st.stop()

dfp = df[[comp_col, evr_col]].dropna().copy()
dfp.columns = ["component","explained_variance_ratio"]
try:
    dfp = dfp.sort_values("component")
except Exception:
    pass
dfp["cumulative"] = dfp["explained_variance_ratio"].cumsum()

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(dfp, x="component", y="explained_variance_ratio", title="Scree — Variância explicada por componente")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig2 = px.line(dfp, x="component", y="cumulative", markers=True, title="Variância explicada acumulada")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Tabela PCA")
st.dataframe(dfp, use_container_width=True)
