import io
import json
import os
import itertools
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from scipy import stats

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(
    page_title="MODELO DE REDE NEURAL ARTIFICIAL — Clusters SP",
    page_icon="🧠",
    layout="wide",
)
TITLE = "MODELO DE REDE NEURAL ARTIFICIAL PARA MAPEAMENTO DE CLUSTERS DE INTELIGÊNCIA E SUA APLICAÇÃO NO MUNICÍPIO DE SÃO PAULO"
st.title(TITLE)

# =============================================================================
# GITHUB I/O HELPERS (monolítico)
# =============================================================================
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
    token = _secret(["github","token"], None)
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def normalize_repo(owner_repo: str) -> str:
    s = (owner_repo or "").strip()
    s = s.replace("https://github.com/", "").replace("http://github.com/", "")
    s = s.strip("/")
    parts = [p for p in s.split("/") if p]
    if len(parts) < 2:
        raise RuntimeError("Informe o repositório no formato 'owner/repo'. Ex.: 'emiliobneto/UrbanTechCluster'")
    return f"{parts[0]}/{parts[1]}"

@st.cache_data(show_spinner=True)
def github_repo_info(owner_repo: str) -> dict:
    owner_repo = normalize_repo(owner_repo)
    url = f"{API_BASE}/repos/{owner_repo}"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Falha lendo repo {owner_repo}: {r.status_code} {r.text}")
    return r.json()

def resolve_branch(owner_repo: str, user_branch: str) -> str:
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
        return []  # devolve vazio para permitir fallbacks
    return r.json()

@st.cache_data(show_spinner=True)
def github_get_contents(owner_repo: str, path: str, branch: str) -> dict:
    owner_repo = normalize_repo(owner_repo)
    url = f"{API_BASE}/repos/{owner_repo}/contents/{path}?ref={branch}"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Falha listando {path}: {r.status_code} {r.text}")
    return r.json()

@st.cache_data(show_spinner=True)
def github_fetch_bytes(owner_repo: str, path: str, branch: str) -> bytes:
    """Baixa binário via Contents API (funciona com LFS/privado)."""
    meta = github_get_contents(owner_repo, path, branch)
    download_url = meta.get("download_url") or build_raw_url(owner_repo, path, branch)
    r = requests.get(download_url, headers=_gh_headers(), timeout=180)
    if r.status_code != 200:
        ct = r.headers.get("Content-Type", "")
        raise RuntimeError(f"Download falhou ({r.status_code}, Content-Type={ct}). Verifique token/privacidade.")
    data = r.content
    # ponteiro LFS?
    if data.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError("Recebi um ponteiro Git LFS em vez do binário. Use token em st.secrets['github']['token'].")
    head = data[:200].strip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise RuntimeError("Recebi HTML em vez do arquivo binário. Provável rate limit/privado. Defina token.")
    return data

@st.cache_data(show_spinner=True)
def load_gpkg(owner_repo: str, path: str, branch: str, layer: str | None = None):
    try:
        import geopandas as gpd  # type: ignore
    except Exception as e:
        raise RuntimeError("geopandas/pyogrio são necessários para ler GPKG.") from e
    blob = github_fetch_bytes(owner_repo, path, branch)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
        tmp.write(blob)
        tmp.flush()
        tmp_path = tmp.name
    try:
        # tenta engine pyogrio (mais estável)
        return gpd.read_file(tmp_path, layer=layer, engine="pyogrio")  # type: ignore
    except Exception:
        # fallback p/ engine padrão (Fiona)
        return gpd.read_file(tmp_path, layer=layer)  # type: ignore
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass

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
def github_branch_info(owner_repo: str, branch: str) -> dict:
    owner_repo = normalize_repo(owner_repo)
    url = f"{API_BASE}/repos/{owner_repo}/branches/{branch}"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Falha lendo branch {branch}: {r.status_code} {r.text}")
    return r.json()

@st.cache_data(show_spinner=True)
def github_tree_paths(owner_repo: str, branch: str) -> list[str]:
    """Lista todos os blobs do repo/branch (recursivo)."""
    info = github_branch_info(owner_repo, branch)
    tree_sha = info["commit"]["commit"]["tree"]["sha"]
    url = f"{API_BASE}/repos/{normalize_repo(owner_repo)}/git/trees/{tree_sha}?recursive=1"
    r = requests.get(url, headers=_gh_headers(), timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Falha lendo tree: {r.status_code} {r.text}")
    tree = r.json().get("tree", [])
    return [ent["path"] for ent in tree if ent.get("type") == "blob"]

def pick_existing_dir(owner_repo: str, branch: str, candidates: list[str]) -> str:
    """Retorna o primeiro diretório existente (case-insensitive helper)."""
    for cand in candidates:
        items = github_listdir(owner_repo, cand, branch)
        if items:  # existe e tem algo
            return cand
    # último recurso: tenta achar um candidato por substrings
    all_paths = github_tree_paths(owner_repo, branch)
    for cand in candidates:
        key = cand.strip("/").lower()
        for p in all_paths:
            if p.lower().startswith(key):
                # retorna só o prefixo de diretório do primeiro arquivo achado
                return "/".join(p.split("/")[:len(key.split("/"))])
    return candidates[0]

# =============================================================================
# COLOR/CLASSIFY/MAPS/STATS HELPERS (monolítico)
# =============================================================================
def hex_to_rgba(hex_color: str):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return [r, g, b, 180]

SEQUENTIAL = {
    4: ['#fee8d8','#fdbb84','#fc8d59','#d7301f'],
    5: ['#feedde','#fdbe85','#fd8d3c','#e6550d','#a63603'],
    6: ['#feedde','#fdd0a2','#fdae6b','#fd8d3c','#e6550d','#a63603'],
    7: ['#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#e6550d','#a63603'],
    8: ['#fff5f0','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#99000d'],
}
CATEGORICAL = [
    '#7c3aed','#d946ef','#fb7185','#f97316','#f59e0b','#facc15','#fde047',
    '#a16207','#9a3412','#b91c1c','#ea580c','#be185d','#9333ea','#6b21a8',
    '#a21caf','#c026d3','#db2777','#e11d48','#eab308','#f43f5e'
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
    if series.dtype.kind in ("O","b","M","m","U","S"):
        return True
    return series.dropna().nunique() <= 12

def jenks_breaks(values: pd.Series, k: int):
    import mapclassify as mc
    vals = values.dropna().astype(float).values
    uniq = np.unique(vals)
    if len(uniq) < max(4, k):
        k = min(len(uniq), max(2, k))
    nb = mc.NaturalBreaks(vals, k=k, initial=200)
    bins = [-float("inf")] + list(nb.bins)
    binned = pd.cut(values, bins=bins, labels=False, include_lowest=True)
    return bins, binned

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

def chi2_between(df: pd.DataFrame, col_a: str, col_b: str):
    tbl = pd.crosstab(df[col_a], df[col_b])
    chi2, p, dof, expected = stats.chi2_contingency(tbl, correction=False)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof), "table": tbl}

def corr_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    return num.corr(method=method).replace([np.inf, -np.inf], np.nan)

# =============================================================================
# SIDEBAR
# =============================================================================
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

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Principal", "🧬 Clusterização", "📊 Univariadas", "🧠 ML → PCA"]
)

# -----------------------------------------------------------------------------
# ABA 1 — Principal
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Quadras e camadas adicionais (GPKG)")
    colA, colB = st.columns([2,1], gap="large")
    with colA:
        st.caption("Carrega `Data/mapa/quadras.gpkg` e sobrepõe camadas auxiliares (linhas/estações/água).")
    with colB:
        basemap = st.radio("Plano de fundo", ["OpenStreetMap", "Satélite (Mapbox)"], index=0)

    # --- Carregamento robusto das quadras (cache + fallback por Tree API)
    quadras_path_default = "Data/mapa/quadras.gpkg"
    gdf_quadras = st.session_state.get("gdf_quadras_cached")

    if gdf_quadras is None:
        first_err = None
        try:
            gdf_quadras = load_gpkg(repo, quadras_path_default, branch)
        except Exception as e:
            first_err = e
            # Procura qualquer arquivo 'quadras.gpkg' no repo
            all_paths = github_tree_paths(repo, branch)
            candidates = [p for p in all_paths if p.lower().endswith("quadras.gpkg")]
            candidates = sorted(
                candidates,
                key=lambda p: ("/data/" not in p.lower(), "/mapa/" not in p.lower(), len(p))
            )
            if not candidates:
                st.error(
                    f"Não encontrei 'quadras.gpkg' no repositório ({repo}@{branch}). "
                    f"Erro ao tentar '{quadras_path_default}': {first_err}"
                )
                st.stop()
            sel_quadras = st.selectbox(
                "Selecione o arquivo de quadras (.gpkg) detectado no repositório:",
                candidates, index=0, key="quadras_tab1"
            )
            gdf_quadras = load_gpkg(repo, sel_quadras, branch)
            st.success(f"Carregado: {sel_quadras}")

        st.session_state["gdf_quadras_cached"] = gdf_quadras

    # --- Camadas auxiliares (se existirem)
    try:
        mapa_dir = pick_existing_dir(repo, branch, ["Data/mapa", "data/mapa", "Data/Mapa"])
        mapa_files = list_files(repo, mapa_dir, branch, (".gpkg",))
        other_layers = [f for f in mapa_files if f["name"].lower() != "quadras.gpkg"]
        layer_names = [f["name"] for f in other_layers]
        sel_layers = st.multiselect("Camadas auxiliares (opcional)", layer_names, default=[])
        loaded_layers = []
        for nm in sel_layers:
            fobj = next(x for x in other_layers if x["name"] == nm)
            g = load_gpkg(repo, fobj["path"], branch)
            loaded_layers.append((nm, g))
    except Exception as e:
        st.warning(f"Não foi possível listar/ler camadas em Data/mapa: {e}")
        loaded_layers = []

    # -------- Dados por SQ
    st.subheader("Dados por `SQ` para espacialização")
    col1, col2, col3 = st.columns([1.6,1,1.2], gap="large")

    with col1:
        src_label = st.radio("Origem dos dados", ["originais", "winsorize"], index=0, horizontal=True)
        base_dir = pick_existing_dir(
            repo, branch,
            [f"Data/dados/{src_label}",
             f"Data/Dados/{src_label}",
             f"Data/dados/{'Originais' if src_label=='originais' else 'winsorizados'}",
             f"Data/dados/{'originais' if src_label=='originais' else 'winsorizados'}"]
        )
        parquet_files = list_files(repo, base_dir, branch, (".parquet",))
        if not parquet_files:
            st.warning(f"Sem .parquet em {base_dir}.")
            st.stop()
        sel_file = st.selectbox("Arquivo .parquet com variáveis", [f["name"] for f in parquet_files])
        fobj = next(x for x in parquet_files if x["name"] == sel_file)
        df_vars = load_parquet(repo, fobj["path"], branch)

    with col2:
        join_col = next((c for c in df_vars.columns if c.upper()=="SQ" or c=="SQ"), None)
        if join_col is None:
            st.error("Dataset selecionado não possui coluna 'SQ'.")
            st.stop()
        years = sorted([int(y) for y in df_vars["Ano"].dropna().unique()]) if "Ano" in df_vars.columns else []
        year = st.select_slider("Ano", options=years, value=years[-1]) if years else None
        if year:
            df_vars = df_vars[df_vars["Ano"]==year]

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
    for feat in geojson.get("features", []):
        val = feat.get("properties", {}).get("value", None)
        hexc = cmap.get(val, "#999999")
        feat.setdefault("properties", {})["fill_color"] = hex_to_rgba(hexc)

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

    # -------- Recortes
    st.subheader("Recortes espaciais (GPKG)")
    st.caption("Seleciona GPKGs em `Data/mapa/recortes` (com fallback por busca).")
    try:
        rec_dir = pick_existing_dir(repo, branch, ["Data/mapa/recortes", "Data/Mapa/recortes", "data/mapa/recortes"])
        recorte_files = list_files(repo, rec_dir, branch, (".gpkg",))
        if not recorte_files:
            # fallback por tree
            all_paths = github_tree_paths(repo, branch)
            rec_paths = [p for p in all_paths if p.lower().endswith(".gpkg") and "/recortes/" in p.lower()]
            if not rec_paths:
                st.info("Nenhum GPKG de recorte encontrado.")
            else:
                rec_sel = st.selectbox("Arquivo de recorte", rec_paths, index=0)
                gdf_rec = load_gpkg(repo, rec_sel, branch)
                layers_rec = [render_geojson_layer(make_geojson(gdf_rec), name="recorte")]
                st.markdown("#### Mapa — Recorte selecionado")
                deck(layers_rec, satellite=basemap.startswith("Satélite")) if basemap.startswith("Satélite") else osm_basemap_deck(layers_rec)
        else:
            rec_sel = st.selectbox("Arquivo de recorte", [f["path"] for f in recorte_files], index=0)
            gdf_rec = load_gpkg(repo, rec_sel, branch)
            layers_rec = [render_geojson_layer(make_geojson(gdf_rec), name="recorte")]
            st.markdown("#### Mapa — Recorte selecionado")
            deck(layers_rec, satellite=basemap.startswith("Satélite")) if basemap.startswith("Satélite") else osm_basemap_deck(layers_rec)
    except Exception as e:
        st.warning(f"Não foi possível listar/ler recortes: {e}")

# -----------------------------------------------------------------------------
# ABA 2 — Clusterização
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Mapa — EstagioClusterizacao")
    base_ori = pick_existing_dir(
        repo, branch,
        ["Data/dados/Originais", "Data/dados/originais", "Data/Dados/Originais"]
    )
    try:
        files_estagio = list_files(repo, base_ori, branch, (".parquet",))
        estagio_candidates = [f for f in files_estagio if "estagioclusterizacao" in f["name"].lower() or f["name"].lower().startswith("estagio")]
        if not estagio_candidates and files_estagio:
            estagio_candidates = files_estagio
        if not estagio_candidates:
            st.error(f"Não encontrei parquet com EstagioClusterizacao em {base_ori}.")
            st.stop()
        sel_estagio = st.selectbox("Arquivo EstagioClusterizacao", [f["name"] for f in estagio_candidates])
        est_file = next(x for x in estagio_candidates if x["name"] == sel_estagio)
        df_est = load_parquet(repo, est_file["path"], branch)
    except Exception as e:
        st.error(f"Erro carregando EstagioClusterizacao: {e}")
        st.stop()

    years = sorted([int(y) for y in df_est["Ano"].dropna().unique()]) if "Ano" in df_est.columns else []
    year = st.select_slider("Ano", options=years, value=years[-1]) if years else None
    if year:
        df_est = df_est[df_est["Ano"] == year]

    gdf_quadras = st.session_state.get("gdf_quadras_cached")
    if gdf_quadras is None:
        st.warning("As quadras ainda não foram carregadas (abra a aba 'Principal').")
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

    gdfc = gdf_quadras.merge(df_est[[join_col_est, cluster_col]], left_on=join_col_quad, right_on=join_col_est, how="left")

    cats = [c for c in gdfc[cluster_col].dropna().unique()]
    palette = pick_categorical(len(cats))
    cmap = {cat: palette[i] for i,cat in enumerate(cats)}

    gdfc["value"] = gdfc[cluster_col]
    gj = make_geojson(gdfc)
    for feat in gj.get("features", []):
        val = feat.get("properties", {}).get("value", None)
        hexc = cmap.get(val, "#999999")
        feat.setdefault("properties", {})["fill_color"] = hex_to_rgba(hexc)

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

    st.subheader("Métricas por cluster/ano")
    opt_vers = st.radio("Versão dos dados", ["originais", "winsorizados"], index=0, horizontal=True)
    base_metrics = pick_existing_dir(
        repo, branch,
        ["Data/analises/original", "Data/analises/Original", "Data/Analises/original"]
        if opt_vers == "originais" else
        ["Data/analises/winsorizados", "Data/analises/Winsorizados"]
    )

    try:
        files_metrics_csv = list_files(repo, base_metrics, branch, (".csv",))
        files_metrics_parq = list_files(repo, base_metrics, branch, (".parquet",))
        main_candidates = [f for f in (files_metrics_parq + files_metrics_csv)
                           if "metrica" in f["name"].lower() or "metrics" in f["name"].lower()]
        files_all = main_candidates if main_candidates else (files_metrics_parq + files_metrics_csv)
        if not files_all:
            st.info(f"Sem arquivos de métricas em {base_metrics}.")
        else:
            sel_met = st.selectbox("Arquivo de métricas", [f["name"] for f in files_all])
            met_obj = next(x for x in files_all if x["name"] == sel_met)
            dfm = load_parquet(repo, met_obj["path"], branch) if met_obj["name"].endswith(".parquet") else load_csv(repo, met_obj["path"], branch)

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

    st.subheader("Associações (Spearman) e Testes t par-a-par")
    try:
        spearman_dir = pick_existing_dir(repo, branch, ["Data/analises/original", "Data/analises/Original"])
        spearman_candidates = [f for f in list_files(repo, spearman_dir, branch, (".csv",))
                               if "spearman_pairs" in f["name"].lower()]
        if spearman_candidates:
            sel_sp = st.selectbox("Arquivo Spearman (original)", [f["name"] for f in spearman_candidates])
            sp_obj = next(x for x in spearman_candidates if x["name"] == sel_sp)
            df_spear = load_csv(repo, sp_obj["path"], branch)
            st.dataframe(df_spear, use_container_width=True)
        else:
            st.info("Nenhum arquivo 'spearman_pairs' encontrado.")
    except Exception as e:
        st.info(f"Spearman pairs não encontrado ou erro ao ler: {e}")

    st.markdown("**Teste t par-a-par entre clusters**")
    try:
        st.caption("Selecione um `.parquet` (originais/winsorize) e uma variável numérica; junta com clusters e calcula t-test entre pares.")
        src_type = st.radio("Origem", ["originais", "winsorize"], horizontal=True, index=0, key="tt_src")
        var_dir = pick_existing_dir(
            repo, branch,
            [f"Data/dados/{src_type}",
             f"Data/dados/{'Originais' if src_type=='originais' else 'winsorizados'}"]
        )
        files_vars = list_files(repo, var_dir, branch, (".parquet",))
        if files_vars:
            sel_vf = st.selectbox("Arquivo com variáveis", [f["name"] for f in files_vars], key="tt_file")
            vf_obj = next(x for x in files_vars if x["name"] == sel_vf)
            dfv = load_parquet(repo, vf_obj["path"], branch)
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
            st.info(f"Sem arquivos em {var_dir}.")
    except Exception as e:
        st.warning(f"Erro nos testes t: {e}")

# -----------------------------------------------------------------------------
# ABA 3 — Univariadas
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Seleção de dataset")
    src_type = st.radio("Origem", ["originais", "winsorizados"], index=0, horizontal=True, key="uni_src")
    base_dir = pick_existing_dir(
        repo, branch,
        [f"Data/dados/{src_type}",
         f"Data/dados/{'Originais' if src_type=='originais' else 'Winsorizados'}",
         f"Data/dados/{'originais' if src_type=='originais' else 'winsorize'}"]
    )

    files_pq = list_files(repo, base_dir, branch, (".parquet",))
    files_csv = list_files(repo, base_dir, branch, (".csv",))
    files_all = files_pq + files_csv
    if not files_all:
        st.error(f"Sem arquivos em {base_dir}.")
        st.stop()

    sel_file = st.selectbox("Arquivo de dados", [f["name"] for f in files_all])
    fobj = next(x for x in files_all if x["name"] == sel_file)
    df = load_parquet(repo, fobj["path"], branch) if fobj["name"].endswith(".parquet") else load_csv(repo, fobj["path"], branch)

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

# -----------------------------------------------------------------------------
# ABA 4 — ML → PCA
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Variância explicada (PCA)")
    base = pick_existing_dir(repo, branch, ["Data/analises/PCA", "Data/Analises/PCA"])
    files_pq = list_files(repo, base, branch, (".parquet",))
    files_csv = list_files(repo, base, branch, (".csv",))
    files_all = files_pq + files_csv
    if not files_all:
        st.info(f"Sem arquivos em {base}.")
        st.stop()

    st.caption("Selecione um arquivo PCA com colunas como `component` e `explained_variance_ratio` (ou similares).")
    sel = st.selectbox("Arquivo PCA", [f["name"] for f in files_all])
    fobj = next(x for x in files_all if x["name"] == sel)
    dfpca = load_parquet(repo, fobj["path"], branch) if fobj["name"].endswith(".parquet") else load_csv(repo, fobj["path"], branch)

    comp_col = next((c for c in dfpca.columns if c.lower().startswith("comp")), dfpca.columns[0])
    evr_col = next((c for c in dfpca.columns if "explained" in c.lower() and "ratio" in c.lower()), None)
    if evr_col is None:
        evr_col = next((c for c in dfpca.columns if "variance" in c.lower() and "ratio" in c.lower()), None)
    if evr_col is None:
        st.error("Não encontrei coluna com 'explained_variance_ratio'.")
        st.dataframe(dfpca.head(), use_container_width=True)
        st.stop()

    dfp = dfpca[[comp_col, evr_col]].dropna().copy()
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
