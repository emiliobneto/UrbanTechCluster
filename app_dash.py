# -*- coding: utf-8 -*-
# APP PRINCIPAL — abas 1, 2, 3 e chamada da aba 4 (PCA)
import io
import os
import json
import re
import itertools
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

# ==========================
# CONFIG GERAL
# ==========================
st.set_page_config(
    page_title="MODELO DE REDE NEURAL ARTIFICIAL — Clusters SP",
    page_icon="🧠",
    layout="wide",
)
TITLE = "MODELO DE REDE NEURAL ARTIFICIAL PARA MAPEAMENTO DE CLUSTERS DE INTELIGÊNCIA E SUA APLICAÇÃO NO MUNICÍPIO DE SÃO PAULO"
st.title(TITLE)

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
    token = _secret(["github","token"], None)
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def normalize_repo(owner_repo):
    s = (owner_repo or "").strip()
    s = s.replace("https://github.com/", "").replace("http://github.com/", "")
    s = s.strip("/")
    parts = [p for p in s.split("/") if p]
    if len(parts) < 2:
        raise RuntimeError("Informe o repositório no formato 'owner/repo'. Ex.: 'emiliobneto/UrbanTechCluster'")
    return f"{parts[0]}/{parts[1]}"

@st.cache_data(show_spinner=True)
def github_repo_info(owner_repo):
    owner_repo = normalize_repo(owner_repo)
    url = f"{API_BASE}/repos/{owner_repo}"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Falha lendo repo {owner_repo}: {r.status_code} {r.text}")
    return r.json()

def resolve_branch(owner_repo, user_branch):
    owner_repo = normalize_repo(owner_repo)
    b = (user_branch or "").strip()
    if b:
        url = f"{API_BASE}/repos/{owner_repo}/branches/{b}"
        r = requests.get(url, headers=_gh_headers(), timeout=60)
        if r.status_code == 200:
            return b
    info = github_repo_info(owner_repo)
    return info.get("default_branch", "main")

def build_raw_url(owner_repo, path, branch):
    owner_repo = normalize_repo(owner_repo).strip("/")
    path = path.lstrip("/")
    return f"{RAW_BASE}/{owner_repo}/{branch}/{path}"

@st.cache_data(show_spinner=False)
def github_listdir(owner_repo, path, branch):
    owner_repo = normalize_repo(owner_repo)
    url = f"{API_BASE}/repos/{owner_repo}/contents/{path}?ref={branch}"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    if r.status_code != 200:
        return []
    return r.json()

@st.cache_data(show_spinner=True)
def github_get_contents(owner_repo, path, branch):
    owner_repo = normalize_repo(owner_repo)
    url = f"{API_BASE}/repos/{owner_repo}/contents/{path}?ref={branch}"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Falha listando {path}: {r.status_code} {r.text}")
    return r.json()

@st.cache_data(show_spinner=True)
def github_fetch_bytes(owner_repo, path, branch):
    meta = github_get_contents(owner_repo, path, branch)
    download_url = meta.get("download_url") or build_raw_url(owner_repo, path, branch)
    r = requests.get(download_url, headers=_gh_headers(), timeout=180)
    if r.status_code != 200:
        ct = r.headers.get("Content-Type", "")
        raise RuntimeError(f"Download falhou ({r.status_code}, Content-Type={ct}). Verifique token/privacidade.")
    data = r.content
    # Ponteiro Git LFS?
    if data.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError("Arquivo está em LFS (ponteiro). Defina token em st.secrets['github']['token'].")
    # HTML/JSON?
    head = data[:200].strip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise RuntimeError("Recebi HTML em vez do arquivo. Provável rate limit/privado. Defina token.")
    return data

@st.cache_data(show_spinner=True)
def load_gpkg(owner_repo, path, branch, layer=None):
    try:
        import geopandas as gpd
    except Exception as e:
        raise RuntimeError("geopandas/pyogrio são necessários para ler GPKG.") from e
    blob = github_fetch_bytes(owner_repo, path, branch)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
        tmp.write(blob); tmp.flush()
        tmp_path = tmp.name
    try:
        return gpd.read_file(tmp_path, layer=layer, engine="pyogrio")
    except Exception:
        return gpd.read_file(tmp_path, layer=layer)
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass

@st.cache_data(show_spinner=True)
def load_parquet(owner_repo, path, branch):
    blob = github_fetch_bytes(owner_repo, path, branch)
    return pd.read_parquet(io.BytesIO(blob), engine="pyarrow")

@st.cache_data(show_spinner=True)
def load_csv(owner_repo, path, branch):
    blob = github_fetch_bytes(owner_repo, path, branch)
    return pd.read_csv(io.BytesIO(blob))

def list_files(owner_repo, path, branch, exts=(".parquet", ".csv", ".gpkg")):
    items = github_listdir(owner_repo, path, branch)
    out = []
    for it in items:
        if isinstance(it, dict) and it.get("type") == "file":
            nm = it["name"]
            if any(nm.lower().endswith(e) for e in exts):
                out.append({"path": f"{path.rstrip('/')}/{nm}", "name": nm})
    return out

@st.cache_data(show_spinner=True)
def github_branch_info(owner_repo, branch):
    owner_repo = normalize_repo(owner_repo)
    url = f"{API_BASE}/repos/{owner_repo}/branches/{branch}"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Falha lendo branch {branch}: {r.status_code} {r.text}")
    return r.json()

@st.cache_data(show_spinner=True)
def github_tree_paths(owner_repo, branch):
    info = github_branch_info(owner_repo, branch)
    tree_sha = info["commit"]["commit"]["tree"]["sha"]
    url = f"{API_BASE}/repos/{normalize_repo(owner_repo)}/git/trees/{tree_sha}?recursive=1"
    r = requests.get(url, headers=_gh_headers(), timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Falha lendo tree: {r.status_code} {r.text}")
    tree = r.json().get("tree", [])
    return [ent["path"] for ent in tree if ent.get("type") == "blob"]

def pick_existing_dir(owner_repo, branch, candidates):
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
                # volta com prefixo original do candidato
                parts = p.split("/")
                return "/".join(parts[:len(key.split("/"))])
    return candidates[0]

# ==========================
# CORES / CLASSIF / MAPAS / LEGENDAS
# ==========================
def hex_to_rgba(hex_color):
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
def pick_sequential(n):
    n = max(4, min(8, n))
    return SEQUENTIAL.get(n, SEQUENTIAL[6])
def pick_categorical(k):
    if k <= len(CATEGORICAL): return CATEGORICAL[:k]
    reps = (k // len(CATEGORICAL)) + 1
    return (CATEGORICAL * reps)[:k]

def is_categorical(series):
    if series.dtype.kind in ("O","b","M","m","U","S"): return True
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
        get_line_color=[100,100,100],
        get_line_width=0.5,
        auto_highlight=True
    )

def render_line_layer(geojson_obj, name="Lines"):
    return pdk.Layer(
        "GeoJsonLayer",
        geojson_obj,
        pickable=True,
        stroked=True,
        filled=False,
        get_line_color=[30,30,30],
        get_line_width=2
    )

def render_point_layer(geojson_obj, name="Points"):
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
    tile = pdk.Layer("TileLayer", data="https://a.tile.openstreetmap.org/{z}/{x}/{y}.png")
    r = pdk.Deck(
        layers=[tile] + layers,
        initial_view_state=initial_view_state or pdk.ViewState(latitude=-23.55, longitude=-46.63, zoom=10),
        map_style=None,
    )
    st.pydeck_chart(r, use_container_width=True)

# ---------- LEGENDAS ----------
def _legend_row(hex_color, label):
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
           <div style="width:14px;height:14px;border-radius:3px;border:1px solid #00000022;background:{hex_color};"></div>
           <div style="font-size:0.9rem;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_legend_categorical(cmap, title="Legenda"):
    st.markdown(f"**{title}**")
    for k in sorted(cmap.keys(), key=lambda x: str(x)):
        _legend_row(cmap[k], str(k))

def _fmt_num(x):
    try:
        if x == -float("inf"): return "-∞"
        if x == float("inf"): return "+∞"
        return f"{float(x):.3g}"
    except Exception:
        return str(x)

def render_legend_numeric(bins, palette, title="Legenda"):
    st.markdown(f"**{title}**")
    k = len(palette)
    for i in range(k):
        left = bins[i]
        right = bins[i+1] if i+1 < len(bins) else float("inf")
        if left == -float("inf"):
            label = f"≤ {_fmt_num(right)}"
        elif right == float("inf"):
            label = f"> {_fmt_num(left)}"
        else:
            label = f"({_fmt_num(left)} – {_fmt_num(right)}]"
        _legend_row(palette[i], label)

# ==========================
# SIDEBAR
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

def pairs_to_matrix(df_pairs: pd.DataFrame, i_col: str, j_col: str, val_col: str, sym_max=True):
    """Converte tabela longa (i,j,valor) em matriz pivot."""
    if not set([i_col, j_col, val_col]).issubset(df_pairs.columns):
        return None
    m = df_pairs.pivot_table(index=i_col, columns=j_col, values=val_col, aggfunc="mean")
    # completa simetria, se desejado
    if sym_max:
        m2 = m.copy()
        m2 = m2.combine_first(m2.T)
        m2 = np.maximum(m2, m2.T)
        m = m2
    return m

# ==========================
# IMPORTA A ABA 4 (PCA)
# ==========================
# --- Import da aba PCA de forma resiliente ---
def _import_render_pca_tab_strong():
    import importlib, importlib.util, sys, os
    here = os.path.dirname(__file__)
    parent = os.path.dirname(here)

    # 1) tentativas de import "normais"
    last_err = None
    for modname in ("ml_pca_tab", "urbantechcluster.ml_pca_tab"):
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, "render_pca_tab"):
                return mod.render_pca_tab
        except Exception as e:
            last_err = e  # pode ser ImportError, ModuleNotFoundError etc.

    # 2) tentativas por caminho (mesmo dir e diretório pai)
    candidates = [
        os.path.join(here, "ml_pca_tab.py"),
        os.path.join(parent, "ml_pca_tab.py"),
    ]
    for path in candidates:
        try:
            if os.path.exists(path):
                spec = importlib.util.spec_from_file_location("ml_pca_tab", path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules["ml_pca_tab"] = mod
                spec.loader.exec_module(mod)
                if hasattr(mod, "render_pca_tab"):
                    return mod.render_pca_tab
        except Exception as e:
            last_err = e

    # 3) se não achou, retorna None e deixamos a aba PCA desabilitada
    return None, last_err

_render = _import_render_pca_tab_strong()
if isinstance(_render, tuple):
    render_pca_tab, _render_err = _render
else:
    render_pca_tab, _render_err = _render, None

# feedback no sidebar e fallback para não quebrar a app
if render_pca_tab is None:
    st.sidebar.error(
        "Aba 4 (PCA) desabilitada: não encontrei `ml_pca_tab.py` "
        "no mesmo diretório nem no diretório pai.\n"
        f"Detalhe: {repr(_render_err)}"
    )
    def render_pca_tab(**kwargs):
        st.error(
            "A aba de PCA está desabilitada porque o arquivo `ml_pca_tab.py` "
            "não foi encontrado/importado. Coloque `ml_pca_tab.py` no mesmo "
            "diretório do `app_dash.py` (ou dentro do pacote `urbantechcluster/`) "
            "e recarregue o app."
        )

import ast

def _find_pca_base_dir(repo, branch, pick_existing_dir):
    # tenta variações mais comuns do caminho
    return pick_existing_dir(
        repo, branch,
        ["Data/analises/PCA", "Data/Analises/PCA", "data/analises/PCA", "data/Analises/PCA"]
    )

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
    # tentativa: dividir por vírgula
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

def _classify_pca_file(df: pd.DataFrame):
    """
    Classifica rapidamente um arquivo de PCA em:
      - 'evr'         → contém variância explicada (colunas variancia_explicada/var_exp etc.)
      - 'pipeline'    → tabela de pipeline (linhas 'pca','imputer','cols','k' e colunas por grupo)
      - 'generic'     → mostrar preview
    """
    cols_l = [c.lower() for c in df.columns]
    # sinais de EVR
    if any(("variancia" in c and ("explic" in c or "acumul" in c)) or ("var_exp" in c) for c in cols_l):
        return "evr"
    # sinais de pipeline (essas tabelas costumam ter a 1ª coluna com rótulos: pca, imputer, scaler, cols, k)
    first_col = df.columns[0] if len(df.columns) else None
    if first_col and df[first_col].astype(str).str.lower().head(5).isin(
        ["pca", "imputer", "scaler", "cols", "k"]
    ).any():
        return "pipeline"
    # também pode vir como colunas com nomes dos grupos e linhas com 'pca','imputer', etc.
    if any("pca(" in str(x).lower() for x in df.head(5).to_numpy().reshape(-1)):
        return "pipeline"
    return "generic"

def _render_variancia_file(df: pd.DataFrame):
    """
    Renderiza variância explicada e acumulada.
    Lê colunas prováveis: grupo | variancia_explicada | variancia_acumulada | var_exp | var_exp_acumulada
    Gera scree-bar e linha acumulada. Sem recálculo.
    """
    cols = {c.lower(): c for c in df.columns}
    # tenta identificar nomes
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

    # se houver agrupamento por 'grupo', permita selecionar
    if col_group and col_group in df_use.columns:
        grupos = df_use[col_group].dropna().astype(str).unique().tolist()
        if grupos:
            g_sel = st.selectbox("Grupo (quando aplicável)", grupos, index=0)
            df_use = df_use[df_use[col_group].astype(str) == g_sel]

    # normalmente cada linha tem uma lista como string; se houver várias linhas, pegue a primeira
    if len(df_use) > 1 and (col_evr in df_use.columns):
        df_use = df_use.head(1)

    evr_list = _safe_literal_list(df_use.iloc[0][col_evr])
    if col_evr_cum and col_evr_cum in df_use.columns:
        evr_cum_list = _safe_literal_list(df_use.iloc[0][col_evr_cum])
    else:
        # se acumulada não existir, compute acumulada a partir da lista (apenas para visualização)
        total = 0.0
        evr_cum_list = []
        for v in evr_list:
            total += float(v)
            evr_cum_list.append(total)

    df_plot = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(evr_list))],
        "explained_variance_ratio": evr_list,
        "cumulative": evr_cum_list
    })

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(df_plot, x="component", y="explained_variance_ratio",
                     title="Scree — Variância explicada por componente")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.line(df_plot, x="component", y="cumulative", markers=True,
                       title="Variância explicada acumulada")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Tabela — Variância")
    st.dataframe(df_plot, use_container_width=True)

def _render_pipeline_file(df: pd.DataFrame):
    """
    Renderiza a tabela de pipeline (PCA, imputer, cols, k, etc.) como quadro legível.
    Esses arquivos que você enviou (p.ex.: PCA_*.csv) têm:
      linhas: pca | imputer | scaler | cols | k
      colunas: nomes dos grupos (Startups, UsoeOcupacao, ...)
    """
    # se a primeira coluna for rotuladora de linhas, transforme em índice
    first_col = df.columns[0]
    if df[first_col].astype(str).str.lower().head(5).isin(["pca","imputer","scaler","cols","k"]).any():
        df2 = df.set_index(first_col)
    else:
        df2 = df.copy()

    st.subheader("Tabela — Pipeline PCA por grupo")
    st.dataframe(df2, use_container_width=True)

    # destaque simples para o 'k' (n_components)
    possible_k_index = next((idx for idx in df2.index.astype(str).str.lower() if idx.strip()=="k"), None)
    if possible_k_index and "k" in df2.index.str.lower().tolist():
        try:
            k_row = df2.loc[[c for c in df2.index if str(c).lower()=="k"][0]]
            st.caption("Componentes (k) por grupo:")
            st.write(k_row.to_frame("k").T)
        except Exception:
            pass

def render_pca_tab_inline(repo, branch, pick_existing_dir, list_files, load_parquet, load_csv):
    st.subheader("Arquivos de PCA (sem recálculo)")
    base_dir = _find_pca_base_dir(repo, branch, pick_existing_dir)
    st.caption(f"Diretório PCA: `{base_dir}`")

    files_all = list_files(repo, base_dir, branch, (".csv", ".parquet"))
    if not files_all:
        st.info("Nenhum arquivo encontrado em `Data/analises/PCA` (ou variações).")
        return

    # Separa por padrão de nome (apenas para ajudar o usuário a escolher rapidamente)
    nomes = [f["name"] for f in files_all]
    evr_default = [n for n in nomes if "variancia" in n.lower() or "var_exp" in n.lower()]
    pipe_default = [n for n in nomes if n.lower().startswith("pca")]

    st.markdown("### 1) Variância explicada")
    if evr_default:
        evr_sel = st.selectbox("Selecione arquivo de variância explicada", evr_default, index=0)
    else:
        evr_sel = st.selectbox("Selecione arquivo de variância explicada", nomes, index=0)
    evr_obj = next(x for x in files_all if x["name"] == evr_sel)
    df_evr = load_parquet(repo, evr_obj["path"], branch) if evr_obj["name"].endswith(".parquet") else load_csv(repo, evr_obj["path"], branch)

    kind_evr = _classify_pca_file(df_evr)
    if kind_evr == "evr":
        _render_variancia_file(df_evr)
    else:
        st.warning("Este arquivo não parece conter variância explicada. Exibindo preview:")
        st.dataframe(df_evr.head(), use_container_width=True)

    st.divider()

    st.markdown("### 2) Pipeline / Modelo (opcional)")
    pipe_choices = pipe_default or nomes
    pipe_sel = st.selectbox("Selecione arquivo de pipeline/modelo", pipe_choices, index=0)
    pipe_obj = next(x for x in files_all if x["name"] == pipe_sel)
    df_pipe = load_parquet(repo, pipe_obj["path"], branch) if pipe_obj["name"].endswith(".parquet") else load_csv(repo, pipe_obj["path"], branch)

    kind_pipe = _classify_pca_file(df_pipe)
    if kind_pipe == "pipeline":
        _render_pipeline_file(df_pipe)
    else:
        st.info("Arquivo não reconhecido como pipeline. Exibindo preview:")
        st.dataframe(df_pipe.head(), use_container_width=True)
# ==========================
# TABS
# ==========================
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Principal", "🧬 Clusterização", "📊 Univariadas", "🧠 ML → PCA"])

# -----------------------------------------------------------------------------
# ABA 1 — Principal (mapa + dados por SQ + recortes) — mantém igual (gera cor por Jenks)
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Quadras e camadas adicionais (GPKG)")
    colA, colB = st.columns([2,1], gap="large")
    with colA:
        st.caption("Carrega `Data/mapa/quadras.gpkg` e camadas auxiliares.")
    with colB:
        basemap = st.radio("Plano de fundo", ["OpenStreetMap", "Satélite (Mapbox)"], index=0)

    # carregar quadras (com fallback)
    quadras_path_default = "Data/mapa/quadras.gpkg"
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
            sel_quadras = st.selectbox("Selecione o arquivo de quadras:", candidates, index=0, key="quadras_tab1")
            gdf_quadras = load_gpkg(repo, sel_quadras, branch)
            st.success(f"Carregado: {sel_quadras}")
        st.session_state["gdf_quadras_cached"] = gdf_quadras

    # camadas auxiliares
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

    # dados por SQ (apenas visual – a paleta é calculada para colorir)
    st.subheader("Dados por `SQ` para espacialização")
    col1, col2, col3 = st.columns([1.6,1,1.2], gap="large")
    with col1:
        src_label = st.radio("Origem dos dados", ["originais", "winsorize"], index=0, horizontal=True)
        base_dir = pick_existing_dir(
            repo, branch,
            [f"Data/dados/{src_label}",
             f"Data/dados/{'Originais' if src_label=='originais' else 'winsorizados'}",
             f"Data/dados/{'originais' if src_label=='originais' else 'winsorize'}"]
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
            st.error("Dataset selecionado não possui coluna 'SQ'."); st.stop()
        years = sorted([int(y) for y in df_vars["Ano"].dropna().unique()]) if "Ano" in df_vars.columns else []
        year = st.select_slider("Ano", options=years, value=years[-1]) if years else None
        if year: df_vars = df_vars[df_vars["Ano"]==year]

    with col3:
        var_options = [c for c in df_vars.columns if c not in (join_col, "Ano")]
        if not var_options:
            st.error("Nenhuma variável disponível além de SQ/Ano."); st.stop()
        var_sel = st.selectbox("Variável a mapear", var_options)
        n_classes = st.slider("Quebras (Jenks)", min_value=4, max_value=8, value=6)

    # merge com quadras
    sq_col_quadras = "SQ" if "SQ" in gdf_quadras.columns else next((c for c in gdf_quadras.columns if c.upper()=="SQ"), None)
    if not sq_col_quadras: st.error("Camada de quadras não possui coluna 'SQ'."); st.stop()
    gdf = gdf_quadras.merge(df_vars[[join_col, var_sel]], left_on=sq_col_quadras, right_on=join_col, how="left")

    # classificação + legenda
    series = gdf[var_sel]
    if is_categorical(series):
        cats = [c for c in series.dropna().unique()]
        palette = pick_categorical(len(cats))
        try: cats_sorted = sorted(cats, key=lambda x: float(x))
        except Exception: cats_sorted = sorted(cats, key=lambda x: str(x))
        cmap = {cat: palette[i] for i,cat in enumerate(cats_sorted)}
        gdf["value"] = series
        legend_kind = "categorical"; legend_info = cmap
    else:
        # quebras Jenks só para colorir (não há cálculo estatístico pesado)
        import mapclassify as mc
        vals = series.dropna().astype(float).values
        uniq = np.unique(vals)
        k = max(4, min(8, n_classes))
        if len(uniq) < max(4, k):
            k = min(len(uniq), max(2, k))
        nb = mc.NaturalBreaks(vals, k=k, initial=200)
        bins = [-float("inf")] + list(nb.bins)
        binned = pd.cut(series, bins=bins, labels=False, include_lowest=True)
        gdf["value"] = binned
        palette = pick_sequential(k)
        cmap = {i: palette[i] for i in range(len(palette))}
        legend_kind = "numeric"; legend_info = (bins, palette)

    geojson = make_geojson(gdf)
    for feat in geojson.get("features", []):
        val = feat.get("properties", {}).get("value", None)
        hexc = cmap.get(val, "#999999") if legend_kind=="numeric" else legend_info.get(val, "#999999")
        feat.setdefault("properties", {})["fill_color"] = hex_to_rgba(hexc)

    layers = [render_geojson_layer(geojson, name="quadras")]
    for nm, g in loaded_layers:
        gj = make_geojson(g)
        try: geoms = set(g.geometry.geom_type.unique())
        except Exception: geoms = {"Polygon"}
        if geoms <= {"LineString","MultiLineString"}:
            layers.append(render_line_layer(gj, nm))
        elif geoms <= {"Point","MultiPoint"}:
            layers.append(render_point_layer(gj, nm))
        else:
            layers.append(render_geojson_layer(gj, nm))

    st.markdown("#### Mapa — Quadras + Camadas auxiliares")
    map_col, legend_col = st.columns([4, 1], gap="large")
    with map_col:
        if basemap.startswith("Satélite"): deck(layers, satellite=True)
        else: osm_basemap_deck(layers)
    with legend_col:
        if legend_kind == "categorical":
            render_legend_categorical(legend_info, title=f"Legenda — {var_sel}")
        elif legend_kind == "numeric":
            bins, palette = legend_info
            render_legend_numeric(bins, palette, title=f"Legenda — {var_sel}")
        else:
            st.caption("Sem legenda.")

    # Recortes
    st.subheader("Recortes espaciais (GPKG)")
    st.caption("Seleciona GPKGs em `Data/mapa/recortes` (com fallback por busca).")
    try:
        rec_dir = pick_existing_dir(repo, branch, ["Data/mapa/recortes", "Data/Mapa/recortes", "data/mapa/recortes"])
        recorte_files = list_files(repo, rec_dir, branch, (".gpkg",))
        if not recorte_files:
            st.info("Nenhum GPKG de recorte encontrado.")
        else:
            rec_sel = st.selectbox("Arquivo de recorte", [f["path"] for f in recorte_files], index=0)
            gdf_rec = load_gpkg(repo, rec_sel, branch)
            layers_rec = [render_line_layer(make_geojson(gdf_rec), name="recorte")]
            st.markdown("#### Mapa — Recorte selecionado")
            col_m, col_l = st.columns([4,1], gap="large")
            with col_m:
                deck(layers_rec, satellite=basemap.startswith("Satélite"))
            with col_l:
                st.markdown("**Legenda — Recorte**"); _legend_row("#444444", "Contorno do recorte")
    except Exception as e:
        st.warning(f"Não foi possível listar/ler recortes: {e}")

# -----------------------------------------------------------------------------
# ABA 2 — Clusterização (somente leitura/exibição)
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Mapa — EstagioClusterizacao")
    base_ori = pick_existing_dir(repo, branch, ["Data/dados/Originais", "Data/dados/originais", "Data/Dados/Originais"])
    try:
        files_estagio = list_files(repo, base_ori, branch, (".parquet",))
        estagio_candidates = [f for f in files_estagio if "estagioclusterizacao" in f["name"].lower() or f["name"].lower().startswith("estagio")]
        if not estagio_candidates: estagio_candidates = files_estagio
        if not estagio_candidates: st.error(f"Não encontrei EstagioClusterizacao em {base_ori}."); st.stop()
        sel_estagio = st.selectbox("Arquivo EstagioClusterizacao", [f["name"] for f in estagio_candidates])
        est_file = next(x for x in estagio_candidates if x["name"] == sel_estagio)
        df_est = load_parquet(repo, est_file["path"], branch)
    except Exception as e:
        st.error(f"Erro carregando EstagioClusterizacao: {e}"); st.stop()

    years = sorted([int(y) for y in df_est["Ano"].dropna().unique()]) if "Ano" in df_est.columns else []
    year = st.select_slider("Ano", options=years, value=years[-1]) if years else None
    if year: df_est = df_est[df_est["Ano"] == year]

    gdf_quadras = st.session_state.get("gdf_quadras_cached")
    if gdf_quadras is None:
        st.warning("As quadras ainda não foram carregadas (abra a aba 'Principal')."); st.stop()

    join_col_est = "SQ" if "SQ" in df_est.columns else None
    join_col_quad = "SQ" if "SQ" in gdf_quadras.columns else None
    if not (join_col_est and join_col_quad):
        st.error("É necessário que quadras e tabela de clusters possuam coluna 'SQ'."); st.stop()

    cluster_cols = [c for c in df_est.columns if "cluster" in c.lower() or "estagio" in c.lower() or "label" in c.lower()]
    if not cluster_cols: st.error("Não encontrei coluna de cluster."); st.stop()
    cluster_col = st.selectbox("Coluna de cluster", cluster_cols, index=0)

    gdfc = gdf_quadras.merge(df_est[[join_col_est, cluster_col]], left_on=join_col_quad, right_on=join_col_est, how="left")
    cats = [c for c in gdfc[cluster_col].dropna().unique()]
    palette = pick_categorical(len(cats))
    try: cats_sorted = sorted(cats, key=lambda x: float(x))
    except Exception: cats_sorted = sorted(cats, key=lambda x: str(x))
    cmap = {cat: palette[i] for i,cat in enumerate(cats_sorted)}

    gdfc["value"] = gdfc[cluster_col]
    gj = make_geojson(gdfc)
    for feat in gj.get("features", []):
        val = feat.get("properties", {}).get("value", None)
        hexc = cmap.get(val, "#999999")
        feat.setdefault("properties", {})["fill_color"] = hex_to_rgba(hexc)

    colA, colB = st.columns([4,1], gap="large")
    with colA:
        st.markdown("#### Mapa — Clusters")
        base = st.radio("Plano de fundo", ["OpenStreetMap", "Satélite (Mapbox)"], index=0, horizontal=True)
        if base.startswith("Satélite"): deck([render_geojson_layer(gj, name="clusters")], satellite=True)
        else: osm_basemap_deck([render_geojson_layer(gj, name="clusters")])
    with colB:
        render_legend_categorical(cmap, title=f"Legenda — {cluster_col}")

    st.subheader("Métricas por cluster/ano (lidas do disco)")
    versao = st.radio("Versão", ["originais", "winsorizados"], index=0, horizontal=True)
    base_metrics = pick_existing_dir(
        repo, branch,
        ["Data/analises/original", "Data/analises/Original", "Data/Analises/original"]
        if versao == "originais" else
        ["Data/analises/winsorizados", "Data/analises/Winsorizados"]
    )
    # arquivos de métricas gerais
    metrics_files = find_files_by_patterns(
        repo, branch, [base_metrics],
        patterns=(r"metrica", r"metrics")
    )
    if metrics_files:
        sel_met = st.selectbox("Arquivo de métricas", [f["name"] for f in metrics_files])
        met_obj = next(x for x in metrics_files if x["name"] == sel_met)
        dfm = load_tabular(repo, met_obj["path"], branch)
        # filtros opcionais por ano/cluster, se existirem
        if "Ano" in dfm.columns:
            years_m = sorted([int(y) for y in dfm["Ano"].dropna().unique()])
            year_m = st.select_slider("Ano (tabela)", options=years_m, value=years_m[-1])
            dfm = dfm[dfm["Ano"]==year_m]
        cl_cols = [c for c in dfm.columns if "cluster" in c.lower() or "estagio" in c.lower() or c.lower()=="label"]
        if cl_cols:
            cl_sel = st.multiselect("Clusters", sorted(dfm[cl_cols[0]].dropna().unique().tolist()))
            if cl_sel: dfm = dfm[dfm[cl_cols[0]].isin(cl_sel)]
        st.dataframe(dfm, use_container_width=True)
    else:
        st.info(f"Sem arquivos de métricas em {base_metrics}.")

    st.subheader("Spearman (pares) — leitura de arquivo e heatmap")
    spearman_files = find_files_by_patterns(
        repo, branch, [base_metrics],
        patterns=(r"spearman", r"pairs")
    )
    if spearman_files:
        sel_sp = st.selectbox("Arquivo Spearman", [f["name"] for f in spearman_files])
        sp_obj = next(x for x in spearman_files if x["name"] == sel_sp)
        df_sp = load_tabular(repo, sp_obj["path"], branch)
        st.markdown("**Tabela (longa) — Spearman**")
        st.dataframe(df_sp, use_container_width=True)
        # tenta montar heatmap se houver colunas var_i/var_j/rho
        cand_i = next((c for c in df_sp.columns if re.search(r"(var|col|a)$", c.lower())), None)
        cand_j = next((c for c in df_sp.columns if re.search(r"(var|col|b)$", c.lower())), None)
        cand_r = next((c for c in df_sp.columns if any(k in c.lower() for k in ["rho","spearman","corr","coef"])), None)
        if cand_i and cand_j and cand_r:
            M = pairs_to_matrix(df_sp, cand_i, cand_j, cand_r, sym_max=True)
            if M is not None:
                fig = px.imshow(M, text_auto=False, color_continuous_scale="Inferno", title="Heatmap — Spearman (dos pares)")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhum arquivo de Spearman encontrado nesta versão de análise.")

    st.subheader("t-test/par-a-par entre clusters — leitura de arquivo")
    # procura arquivos de t-test já calculados
    ttest_files = find_files_by_patterns(
        repo, branch, [base_metrics],
        patterns=(r"ttest", r"t-test", r"pairwise.*t", r"t_par")
    )
    if ttest_files:
        sel_tt = st.selectbox("Arquivo t-test", [f["name"] for f in ttest_files])
        tt_obj = next(x for x in ttest_files if x["name"] == sel_tt)
        df_tt = load_tabular(repo, tt_obj["path"], branch)
        st.dataframe(df_tt, use_container_width=True)
        # caso tenha colunas grupo_a, grupo_b, p_value — cria matriz de p-valor
        col_a = next((c for c in df_tt.columns if c.lower() in ("grupo_a","cluster_a","a","grupo1")), None)
        col_b = next((c for c in df_tt.columns if c.lower() in ("grupo_b","cluster_b","b","grupo2")), None)
        col_p = next((c for c in df_tt.columns if "p" in c.lower()), None)
        if col_a and col_b and col_p:
            P = pairs_to_matrix(df_tt, col_a, col_b, col_p, sym_max=True)
            if P is not None:
                figp = px.imshow(P, text_auto=False, color_continuous_scale="Viridis", title="Matriz de p-valores (t-test)")
                st.plotly_chart(figp, use_container_width=True)
    else:
        st.info("Nenhum arquivo de t-test encontrado nesta versão de análise.")

# -----------------------------------------------------------------------------
# ABA 3 — Univariadas (somente leitura/exibição)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Seleção de versão e tipo de análise")
    versao_u = st.radio("Versão", ["originais", "winsorizados"], index=0, horizontal=True, key="uni_ver")
    base_u = pick_existing_dir(
        repo, branch,
        ["Data/analises/original", "Data/analises/Original"]
        if versao_u=="originais" else
        ["Data/analises/winsorizados", "Data/analises/Winsorizados"]
    )
    analise_tipo = st.selectbox(
        "Tipo de análise",
        ["chi2", "spearman", "pearson", "ttest", "pairwise", "univariadas", "correlacao_matriz"]
    )

    padroes = {
        "chi2":        (r"chi", r"chi2"),
        "spearman":    (r"spearman",),
        "pearson":     (r"pearson", r"correl"),
        "ttest":       (r"ttest", r"t-test"),
        "pairwise":    (r"pairwise",),
        "univariadas": (r"univariad", r"descri", r"summary"),
        "correlacao_matriz": (r"corr(_|.*)matrix", r"correlation(_|.*)matrix", r"correlacao.*matriz", r"pearson.*matrix", r"spearman.*matrix"),
    }
    found = find_files_by_patterns(repo, branch, [base_u], patterns=padroes.get(analise_tipo, ()))
    if not found:
        st.info(f"Nenhum arquivo encontrado em `{base_u}` para {analise_tipo}.")
    else:
        sel_file = st.selectbox("Arquivo", [f["name"] for f in found])
        fobj = next(x for x in found if x["name"] == sel_file)
        df_any = load_tabular(repo, fobj["path"], branch)
        st.dataframe(df_any, use_container_width=True)

        # se for correlação (pares → matriz) desenha heatmap
        if analise_tipo in ("spearman","pearson","correlacao_matriz"):
            # tenta detectar formato: matriz já pronta vs pares
            if (df_any.shape[0] == df_any.shape[1]) and (set(df_any.columns) == set(df_any.index.astype(str))):
                fig = px.imshow(df_any, text_auto=False, color_continuous_scale="Inferno",
                                title=f"Heatmap — {analise_tipo}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # tenta (var_a, var_b, coef)
                cand_i = next((c for c in df_any.columns if re.search(r"(var|col).*a", c.lower())), None)
                cand_j = next((c for c in df_any.columns if re.search(r"(var|col).*b", c.lower())), None)
                cand_r = next((c for c in df_any.columns if any(k in c.lower() for k in ["rho","pearson","spearman","corr","coef"])), None)
                if cand_i and cand_j and cand_r:
                    M = pairs_to_matrix(df_any, cand_i, cand_j, cand_r, sym_max=True)
                    if M is not None:
                        fig = px.imshow(M, text_auto=False, color_continuous_scale="Inferno",
                                        title=f"Heatmap — {analise_tipo} (pares → matriz)")
                        st.plotly_chart(fig, use_container_width=True)
                # caso contrário, apenas exibe a tabela mesmo

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
        load_csv=load_csv
    )
    
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
        id_like = any(c in cols for c in ["sq","id","codigo","code"])
        has_pcs = any(c.startswith("pc") for c in cols)
        if has_pcs:
            return "scores" if id_like else "scores_no_id"
        return "unknown"
    
    def _list_candidate_files(repo, branch, base_dir, list_files, load_parquet, load_csv):
        files_all = list_files(repo, base_dir, branch, (".parquet",".csv"))
        candidates = {"evr": [], "loadings": [], "scores": [], "unknown": []}
        for f in files_all:
            try:
                df = load_parquet(repo, f["path"], branch) if f["name"].endswith(".parquet") else load_csv(repo, f["path"], branch)
                kind = _classify_pca_file(df)
            except Exception:
                kind = "unknown"
            if kind == "evr":
                candidates["evr"].append((f, "evr"))
            elif kind in ("loadings_long","loadings_wide"):
                candidates["loadings"].append((f, kind))
            elif kind in ("scores","scores_no_id"):
                candidates["scores"].append((f, kind))
            else:
                candidates["unknown"].append((f, "unknown"))
        return candidates
    
    def _tidy_loadings(df: pd.DataFrame):
        cols_lower = {c: c.lower() for c in df.columns}
        if "component" in cols_lower.values() and any(x in cols_lower.values() for x in ["loading","valor","carga"]):
            comp_col = next(k for k,v in cols_lower.items() if v=="component")
            load_col = next(k for k,v in cols_lower.items() if v in ("loading","valor","carga"))
            var_col = next((k for k,v in cols_lower.items() if v in ("variable","feature","variavel","atributo")), None)
            if var_col is None:
                non_num = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != comp_col]
                var_col = non_num[0] if non_num else comp_col
            out = df[[var_col, comp_col, load_col]].copy()
            out.columns = ["variable","component","loading"]
            return out
        pc_cols = [c for c in df.columns if c.lower().startswith("pc") or c.lower().startswith("component")]
        if pc_cols:
            var_candidates = [c for c in df.columns if c not in pc_cols]
            if len(var_candidates) == 0:
                df = df.copy(); df["variable"] = df.index.astype(str)
                var_col = "variable"
            else:
                var_col = var_candidates[0]
            long = df.melt(id_vars=[var_col], value_vars=pc_cols, var_name="component", value_name="loading")
            long.columns = ["variable","component","loading"]
            return long
        return pd.DataFrame(columns=["variable","component","loading"])
    
    def _prep_scores(df: pd.DataFrame):
        cols = {c.lower(): c for c in df.columns}
        pc_cols = [c for c in df.columns if c.lower().startswith("pc")]
        id_col = cols.get("sq") or cols.get("id") or cols.get("codigo") or cols.get("code")
        ano_col = cols.get("ano")
        return pc_cols, id_col, ano_col
    
    def _render_evr_section(df_evr: pd.DataFrame):
        cols = {c.lower(): c for c in df_evr.columns}
        if "explained_variance_ratio" in cols:
            evr_col = cols["explained_variance_ratio"]; comp_col = None
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
        df = df[["component","explained_variance_ratio"]].dropna()
        try: df = df.sort_values("component")
        except Exception: pass
        df["cumulative"] = df["explained_variance_ratio"].cumsum()
    
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(df, x="component", y="explained_variance_ratio", title="Scree — Variância explicada por componente")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.line(df, x="component", y="cumulative", markers=True, title="Variância explicada acumulada")
            st.plotly_chart(fig2, use_container_width=True)
    
        st.subheader("Tabela — Variância explicada")
        st.dataframe(df, use_container_width=True)
    
    def _render_loadings_section(df_load: pd.DataFrame):
        long = _tidy_loadings(df_load)
        if long.empty:
            st.warning("Não foi possível identificar a estrutura de *loadings* deste arquivo.")
            st.dataframe(df_load.head(), use_container_width=True)
            return
        comps = sorted(long["component"].astype(str).unique(), key=lambda x: (len(x), x))
        c1, c2 = st.columns([2,1])
        with c1:
            comp_sel = st.selectbox("Componente", comps, index=0)
        with c2:
            topn = st.slider("Top |loading|", 5, 30, 15)
    
        sub = long[long["component"].astype(str)==str(comp_sel)].copy()
        sub["abs_loading"] = sub["loading"].abs()
        sub = sub.sort_values("abs_loading", ascending=False).head(topn)
        fig = px.bar(sub.sort_values("abs_loading"), x="abs_loading", y="variable", orientation="h",
                     title=f"Maiores |loadings| — {comp_sel}")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Tabela — Loadings")
        st.dataframe(sub.drop(columns=["abs_loading"]), use_container_width=True)
    
    def _render_scores_section(df_scores: pd.DataFrame, repo, branch, pick_existing_dir, list_files, load_parquet, load_csv):
        pc_cols, id_col, ano_col = _prep_scores(df_scores)
        if not pc_cols:
            st.warning("Arquivo de *scores* sem colunas de PCs identificáveis.")
            st.dataframe(df_scores.head(), use_container_width=True)
            return
    
        # filtro opcional por ano
        if ano_col:
            anos = sorted([int(x) for x in df_scores[ano_col].dropna().unique()])
            ano_sel = st.select_slider("Ano (scores)", options=anos, value=anos[-1])
            df_scores = df_scores[df_scores[ano_col]==ano_sel]
    
        pc_x = st.selectbox("PC eixo X", pc_cols, index=0)
        pc_y = st.selectbox("PC eixo Y", pc_cols, index=1 if len(pc_cols) > 1 else 0)
        hover_cols = [pc_x, pc_y]
        if id_col: hover_cols.insert(0, id_col)
    
        fig = px.scatter(df_scores, x=pc_x, y=pc_y, hover_data=hover_cols, title=f"Biplot (scores) — {pc_x} × {pc_y}")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Tabela — Scores (colunas selecionadas)")
        st.dataframe(df_scores[hover_cols].dropna(how="all"), use_container_width=True)
    
    def render_pca_tab(repo, branch, pick_existing_dir, list_files, load_parquet, load_csv, github_tree_paths):
        st.subheader("Arquivos de PCA (somente leitura)")
        base_dir = _find_pca_base_dir(repo, branch, pick_existing_dir)
    
        with st.spinner("Procurando arquivos de PCA..."):
            cands = _list_candidate_files(repo, branch, base_dir, list_files, load_parquet, load_csv)
    
        st.caption(f"Diretório PCA: `{base_dir}`")
    
        # 1) Variância explicada
        st.markdown("### 1) Variância explicada (Scree + cumulativa)")
        if cands["evr"]:
            sel_evr_name = st.selectbox("Arquivo de variância explicada", [f["name"] for f,_ in cands["evr"]], index=0)
            evr_obj, _ = next(x for x in cands["evr"] if x[0]["name"] == sel_evr_name)
            df_evr = load_parquet(repo, evr_obj["path"], branch) if evr_obj["name"].endswith(".parquet") else load_csv(repo, evr_obj["path"], branch)
            _render_evr_section(df_evr)
        else:
            st.info("Nenhum arquivo claramente identificado como 'explained_variance_ratio'.")
    
        st.divider()
    
        # 2) Loadings
        st.markdown("### 2) Cargas (loadings) por componente")
        if cands["loadings"]:
            sel_load_name = st.selectbox("Arquivo de loadings", [f["name"] for f,_ in cands["loadings"]], index=0)
            load_obj, kind = next(x for x in cands["loadings"] if x[0]["name"] == sel_load_name)
            df_load = load_parquet(repo, load_obj["path"], branch) if load_obj["name"].endswith(".parquet") else load_csv(repo, load_obj["path"], branch)
            _render_loadings_section(df_load)
        else:
            st.info("Nenhum arquivo de *loadings* identificado.")
    
        st.divider()
    
        # 3) Scores
        st.markdown("### 3) Scores / Projeções (Biplot PC1×PC2)")
        if cands["scores"]:
            sel_scores_name = st.selectbox("Arquivo de scores", [f["name"] for f,_ in cands["scores"]], index=0)
            sc_obj, kind = next(x for x in cands["scores"] if x[0]["name"] == sel_scores_name)
            df_scores = load_parquet(repo, sc_obj["path"], branch) if sc_obj["name"].endswith(".parquet") else load_csv(repo, sc_obj["path"], branch)
            _render_scores_section(df_scores, repo, branch, pick_existing_dir, list_files, load_parquet, load_csv)
        else:
            st.info("Nenhum arquivo de *scores* identificado.")




