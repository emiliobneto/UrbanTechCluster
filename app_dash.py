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

try:
    from scipy.stats import spearmanr as _spearman_fn, shapiro as _shapiro_fn, \
         ttest_ind as _ttest_fn, mannwhitneyu as _mw_fn, \
         f_oneway as _anova_fn, kruskal as _kruskal_fn
except Exception:
    _spearman_fn = _shapiro_fn = _ttest_fn = _mw_fn = _anova_fn = _kruskal_fn = None
    
# ——— CONFIGURAÇÃO GERAL (deve ser a 1ª chamada Streamlit) ———
st.set_page_config(
    page_title="MODELO DE REDE NEURAL ARTIFICIAL — Clusters SP",
    page_icon="🧠",
    layout="wide",
)

# ✅ CSS vem antes do título e com seletores mais seguros
st.markdown(
    """
    <style>
      /* Compat: versões antigas e novas */
      :root .block-container, :root .stMainBlockContainer {
        max-width: 1600px;
        padding-top: 1.25rem; /* mais espaço pro título */
      }
      /* Tabs: limite o escopo aos botões das tabs */
      div[data-testid="stTabs"] button p {
        margin: 0 !important;
        font-size: 15px !important;
        color: rgba(17,17,17,1) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

TITLE = (
    "MODELO DE REDE NEURAL ARTIFICIAL PARA MAPEAMENTO DE CLUSTERS DE INTELIGÊNCIA "
    "E SUA APLICAÇÃO NO MUNICÍPIO DE SÃO PAULO"
)
st.title(TITLE)

# --- Definição precoce de repo/branch (antes de QUALQUER uso) ---
with st.sidebar:
    st.header("🔗 Fonte dos Dados (GitHub)")
    repo_input = st.text_input("owner/repo", value="emiliobneto/UrbanTechCluster", key="repo_input")
    branch_input = st.text_input("branch (vazio = auto)", value="", key="branch_input")

try:
    repo = normalize_repo(repo_input)     # <- define repo aqui
    branch = resolve_branch(repo, branch_input)  # <- e o branch aqui
    st.session_state["repo"] = repo
    st.session_state["branch"] = branch
    st.sidebar.caption(f"Usando: **{repo}@{branch}**")
except Exception as e:
    st.error(f"Configuração inválida: {e}")
    st.stop()

if not repo or not branch:
    st.stop()

# ==========================
# GITHUB I/O HELPERS
# ==========================
API_BASE = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"

def _load_quadras_min(repo, branch):
    gdf = st.session_state.get("gdf_quadras_cached")
    if gdf is None or gdf.empty:
        gdf = load_gpkg(repo, "Data/mapa/quadras.gpkg", branch)
        st.session_state["gdf_quadras_cached"] = gdf
    sq_col = next((c for c in gdf.columns if str(c).upper() == "SQ"), None)
    if not sq_col:
        raise RuntimeError("Camada de quadras não possui coluna 'SQ'.")
    gmin = gdf[[sq_col, gdf.geometry.name]].copy()
    gmin["_SQ_norm"] = _norm_sq_series(gmin[sq_col])
    try:
        gmin = ensure_wgs84(gmin)
        gmin["_centroid"] = gmin.geometry.centroid
    except Exception:
        gmin["_centroid"] = gmin.geometry
    return gmin, sq_col

# uso:
try:
    gdfq_min, sq_col_quadras = _load_quadras_min(repo, branch)
except Exception as e:
    st.error(f"Falha ao carregar quadras: {e}")
    st.stop()

    # ==========
    # ARQUIVO DE VALORES (por SQ) + filtro de ano coerente
    # ==========
    ver_val = st.radio("Versão dos dados (valores por SQ)", ["originais", "winsorizados"], horizontal=True, key="t2_vals_ver")
    base_vals = pick_existing_dir(
        repo, branch,
        [f"Data/dados/{'originais' if ver_val=='originais' else 'winsorizados'}",
         f"Data/dados/{'Originais' if ver_val=='originais' else 'Winsorizados'}",
         f"Data/dados/{'winsorize' if ver_val!='originais' else 'originais'}"]
    )
    vals_all = list_files(repo, base_vals, branch, (".parquet", ".csv"))
    incl_pred = st.checkbox("Incluir arquivos pred_*", value=False, key="t2_vals_incl_pred")
    vals_files = [
        f for f in vals_all
        if (incl_pred or not str(f["name"]).lower().startswith("pred_"))
        and not re.search(r"(?i)est[aá]gio.*cluster", str(f["name"]))
    ]
    if not vals_files:
        st.info(f"Nenhum arquivo elegível em `{base_vals}` (excluí EstagioClusterizacao.* e, opcionalmente, pred_*).")
        st.stop()
    sel_vals = st.selectbox("Arquivo de valores (por SQ)", [f["name"] for f in vals_files], index=0, key="t2_vals_file")
    vals_obj = next(x for x in vals_files if x["name"] == sel_vals)
    df_vals_raw = load_parquet(repo, vals_obj["path"], branch) if str(vals_obj["name"]).endswith(".parquet") else load_csv(repo, vals_obj["path"], branch)
    
    sq_col_vals = next((c for c in df_vals_raw.columns if str(c).upper() == "SQ"), None)
    if sq_col_vals is None:
        st.error("O arquivo de valores precisa ter a coluna 'SQ'.")
        st.stop()
    ano_col_vals = next((c for c in df_vals_raw.columns if str(c).lower() in ("ano", "year")), None)
    if ano_col_vals and year_sel is not None:
        df_vals_raw = df_vals_raw[pd.to_numeric(df_vals_raw[ano_col_vals], errors="coerce").astype("Int64") == year_sel].copy()
    
    # ==========
    # PRÉ-CARREGAMENTO (cache): métricas por cluster × ano — FILTRADO PELO ANO
    # ==========
    # Reduz clusters ao ano selecionado para o cálculo (mais leve)
    df_est_for_pre = df_est_raw
    if ano_col_est and (year_sel is not None):
        df_est_for_pre = df_est_raw[pd.to_numeric(df_est_raw[ano_col_est], errors="coerce").astype("Int64") == year_sel].copy()
    
    metrics_key = f"t2_metrics_{repo}@{branch}|{vals_obj['path']}|{source_label}|{cluster_col}|{year_sel}"
    df_metrics_all = st.session_state.get(metrics_key)
    
    if preload_toggle and (df_metrics_all is None or df_metrics_all.empty):
        try:
            with st.spinner("Pré-carregando métricas por cluster×ano..."):
                df_metrics_all = _preload_cluster_metrics_by_year(
                    df_vals_raw, df_est_for_pre, cluster_col, chunk_size=max_vars
                )
            st.session_state[metrics_key] = df_metrics_all
        except MemoryError:
            st.warning("Pré-cálculo ficou pesado; desative o pré-carregamento ou reduza o número de variáveis.")
            df_metrics_all = pd.DataFrame()
    
    # ======================
    # MAPA DE CLUSTERIZAÇÃO
    # ======================
    st.markdown("### 🗺️ Mapa de clusterização")
    base_map_t2 = st.radio("Plano de fundo", ["OpenStreetMap", "Satélite (Mapbox)"], index=0, horizontal=True, key="t2_base")
    view_mode = st.radio("Visualização do mapa", ["Mapa geral", "Recorte(s)"], index=0, horizontal=True, key="t2_view")
    
    gdf_map = gdfq_min.merge(df_est_clean, on="_SQ_norm", how="inner").copy()
    if gdf_map.empty:
        st.info("Não há feições para mapear após o JOIN de quadras × clusters.")
    else:
        # amostragem para acelerar
        if len(gdf_map) > max_feat:
            gdf_map = gdf_map.sample(n=max_feat, random_state=42)
    
        # simplificação opcional (só polígonos)
        if (not fast_map) and simplify_tol and simplify_tol > 0:
            try:
                gdf_map = gdf_map.copy()
                gdf_map[gdf_map.geometry.name] = gdf_map.geometry.simplify(simplify_tol, preserve_topology=True)
            except Exception:
                pass
    
        palette = pick_categorical(4)
    
        def _geojson_colored(gdf_in, use_centroid: bool):
            import geopandas as gpd
            geom_col = "_centroid" if use_centroid else gdf_in.geometry.name
            gg = gpd.GeoDataFrame(
                gdf_in[[geom_col, "_cl_code"]].rename(columns={geom_col: "geometry"}),
                geometry="geometry",
                crs=getattr(gdf_in, "crs", 4326)
            )
            gj = make_geojson(gg)
            for feat in gj.get("features", []):
                cl_raw = feat.get("properties", {}).get("_cl_code", None)
                cl = _safe_int(cl_raw, {0, 1, 2, 3})
                hexc = palette[cl] if cl is not None else "#999999"
                feat.setdefault("properties", {})
                feat["properties"]["fill_color"] = hex_to_rgba(hexc, 180 if use_centroid else 150)
                feat["properties"]["name"] = f"Cluster {cl}" if cl is not None else "Cluster indef."
                feat["properties"]["value"] = label_map.get(cl, str(cl_raw))
            return gj
    
        def _draw_map(layers):
            if base_map_t2.startswith("Satélite"):
                deck(layers, satellite=True)
            else:
                osm_basemap_deck(layers)
    
        if view_mode == "Mapa geral":
            colL, colC, colR = st.columns([0.08, 0.84, 0.08])
            with colC:
                gj = _geojson_colored(gdf_map, use_centroid=fast_map)
                lyr = render_point_layer(gj, "clusters (centróides)") if fast_map else render_geojson_layer(gj, "clusters")
                _draw_map([lyr])
            st.markdown("**Legenda — clusters**")
            for c in [0, 1, 2, 3]:
                _legend_row(palette[c], label_map[c])
    
        else:
            # Recortes lado a lado
            rec_dir = pick_existing_dir(repo, branch, ["Data/mapa/recortes", "data/mapa/recortes", "Data/Mapa/recortes"])
            rec_files = list_files(repo, rec_dir, branch, (".gpkg",))
            if not rec_files:
                st.info("Nenhum GPKG de recorte encontrado em `Data/mapa/recortes`.")
            else:
                rec_name = st.selectbox("Recorte (.gpkg)", [f["name"] for f in rec_files], index=0, key="t2rec_file")
                rec_obj = next(x for x in rec_files if x["name"] == rec_name)
                gdfrec = ensure_wgs84(load_gpkg(repo, rec_obj["path"], branch))
    
                # interseção: pega só o que cai no recorte
                try:
                    import geopandas as gpd
                    gq = gpd.GeoDataFrame(gdf_map[[gdf_map.geometry.name]], geometry=gdf_map.geometry.name, crs=getattr(gdf_map, "crs", 4326))
                    gq = ensure_wgs84(gq)
                    gr = ensure_wgs84(gdfrec)[["geometry"]]
                    sel_idx = gpd.sjoin(gq, gr, predicate="intersects", how="inner").index.unique()
                    gdf_sub = gdf_map.loc[sel_idx].copy()
                except Exception:
                    bbox = gdfrec.total_bounds
                    gdf_sub = gdf_map.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].copy()
    
                gj_sub = _geojson_colored(gdf_sub, use_centroid=fast_map)
                lyr_sub = render_point_layer(gj_sub, "recorte") if fast_map else render_geojson_layer(gj_sub, "recorte")
                lyr_brd = render_line_layer(make_geojson(gdfrec), "borda recorte")
    
                c1, c2 = st.columns([1.6, 1], gap="large")
                with c1:
                    _draw_map([lyr_sub, lyr_brd])
                with c2:
                    st.markdown("**Legenda — clusters**")
                    for c in [0, 1, 2, 3]:
                        _legend_row(palette[c], label_map[c])
                    st.metric("Feições no recorte", len(gdf_sub))

def _load_clusters(repo, branch) -> tuple[pd.DataFrame, str] | tuple[None, str]:
    try:
        clusters_dir = pick_existing_dir(
            repo, branch, ["Data/dados/Originais", "Data/dados/originais", "data/dados/originais"]
        )
        all_in_dir = list_files(repo, clusters_dir, branch, (".csv", ".parquet"))
        # preferir nome exato
        cand = [f for f in all_in_dir if re.fullmatch(r"(?i)EstagioClusterizacao\.(csv|parquet)", str(f["name"]))]
        if not cand:
            cand = [f for f in all_in_dir if re.search(r"(?i)est[aá]gio", str(f["name"])) and re.search(r"(?i)cluster", str(f["name"]))]
        if not cand:
            return None, "Não encontrei `EstagioClusterizacao.{csv|parquet}` em Data/dados/Originais."
        est_file = cand[0]
        df_est = (
            load_parquet(repo, est_file["path"], branch)
            if str(est_file["name"]).lower().endswith(".parquet")
            else load_csv(repo, est_file["path"], branch)
        )
        source_label = f"{clusters_dir}/{est_file['name']}"
        return df_est, source_label
    except Exception as e:
        return None, f"Falha ao ler arquivo de clusters: {e}"

# uso:
df_est_raw, source_label = (None, "")
if up is not None:
    try:
        df_est_raw = (pd.read_parquet(up) if up.name.lower().endswith(".parquet") else pd.read_csv(up))
        source_label = f"(upload) {up.name}"
    except Exception as e:
        st.error(f"Falha ao ler upload: {e}")
else:
    df_est_raw, source_label = _load_clusters(repo, branch)

if not isinstance(df_est_raw, pd.DataFrame) or df_est_raw.empty:
    st.error(f"Clusters indisponíveis. {source_label or ''}")
    st.stop()

    # Ano + coluna de cluster (robustos a tipos)
    ano_col_est = next((c for c in df_est_raw.columns if str(c).lower() == "ano"), None)
    anos_ok = None
    if ano_col_est:
        anos_vals = pd.to_numeric(df_est_raw[ano_col_est], errors="coerce")
        anos_ok = sorted(anos_vals.dropna().astype(int).unique().tolist()) or None
    year_sel = st.select_slider("Ano (clusters)", options=anos_ok or [None], value=(anos_ok[-1] if anos_ok else None), key="t2_year_sel")

    cluster_cols = [c for c in df_est_raw.columns if re.search(r"(?i)(cluster|est[aá]gio|label)", str(c))]
    if not cluster_cols:
        st.error("Não encontrei coluna de cluster (ex.: EstagioClusterizacao, Cluster, Label).")
        st.stop()
    preferred = next((c for c in cluster_cols if str(c).lower() == "estagioclusterizacao"), cluster_cols[0])
    cluster_col = st.selectbox("Coluna de cluster", cluster_cols, index=cluster_cols.index(preferred), key="t2_cluster_col")

    # Normaliza/filtra clusters p/ JOIN
    df_est_clean = df_est_raw.copy()
    sq_est_col = next((c for c in df_est_clean.columns if str(c).upper() == "SQ"), None)
    if sq_est_col is None:
        st.error("Arquivo de clusters precisa ter coluna 'SQ'.")
        st.stop()
    df_est_clean["_SQ_norm"] = _norm_sq_series(df_est_clean[sq_est_col])
    if (ano_col_est is not None) and (year_sel is not None):
        df_est_clean = df_est_clean[pd.to_numeric(df_est_clean[ano_col_est], errors="coerce").astype("Int64") == year_sel].copy()
    df_est_clean["_cl_code"] = df_est_clean[cluster_col].map(_to_int_code).astype("Int64")
    df_est_clean = (
        df_est_clean.sort_values(["_SQ_norm", "_cl_code"])
        .drop_duplicates("_SQ_norm", keep="last")
        .dropna(subset=["_SQ_norm", "_cl_code"])
        [["_SQ_norm", "_cl_code"]]
    )

def _norm_sq_6(x):
            s = re.sub(r"\D", "", str(x)) if x is not None else ""
            if s == "": return None
            if len(s) > 6: s = s[-6:]
            return s.zfill(6)

dfp = df_pred[[sq_col, est_col, pred_col]].copy()
dfp.columns = ["SQ_raw", "estagio", "predicted"]
dfp["_SQ_norm"] = dfp["SQ_raw"].apply(_norm_sq_6)

gdf_tmp = gdf_quadras[[sq_geo_col, geom_name]].copy()
gdf_tmp["_SQ_norm"] = gdf_tmp[sq_geo_col].apply(_norm_sq_6)

gdf = gdf_tmp.merge(dfp[["_SQ_norm", "estagio", "predicted"]], on="_SQ_norm", how="left")
gdf = ensure_wgs84(gdf)

# Paletas
cmap_est = {str(v): c for v, c in zip(sorted(gdf["estagio"].dropna().astype(str).unique()), pick_categorical(len(gdf["estagio"].dropna().astype(str).unique())))}
cmap_pred = {str(v): c for v, c in zip(sorted(gdf["predicted"].dropna().astype(str).unique()), pick_categorical(len(gdf["predicted"].dropna().astype(str).unique())))}

# ===== Hotfix: utilitários globais para testes/efeitos =====
def _bh_fdr(pvals):
    """Benjamini–Hochberg FDR (q-values). Aceita array/Series."""
    s = pd.Series(pvals, copy=False).astype(float)
    p = s.to_numpy()
    idx = np.isfinite(p)
    if not idx.any():
        return pd.Series(np.nan, index=s.index)
    p_ok = p[idx]
    order = np.argsort(p_ok)
    ranks = np.arange(1, len(p_ok) + 1, dtype=float)
    q_sorted = (p_ok[order] * len(p_ok)) / ranks
    for i in range(len(q_sorted) - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q = np.full_like(p, np.nan, dtype=float)
    q[idx] = np.clip(q_sorted, 0, 1)
    return pd.Series(q, index=s.index)

def _eta_squared(groups):
    """Eta² de um fator (ANOVA one-way) a partir de listas/arrays já separados por grupo."""
    arrs = [np.asarray(g, float)[np.isfinite(g)] for g in groups if len(g)]
    if not arrs:
        return np.nan
    allv = np.concatenate(arrs)
    if allv.size < 2:
        return np.nan
    grand = float(np.mean(allv))
    ss_between = sum(len(g) * (float(np.mean(g)) - grand) ** 2 for g in arrs if len(g))
    ss_total = float(np.sum((allv - grand) ** 2))
    return float(ss_between / ss_total) if ss_total > 0 else np.nan

def _cohens_d(a, b):
    """d de Cohen (amostras independentes)."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    return float((np.mean(a) - np.mean(b)) / np.sqrt(sp)) if sp > 0 else np.nan

def _cliffs_delta(a, b):
    """Cliff’s delta (amostras independentes)."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan
    # versão simples O(n*m)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (n1 * n2))

def _gini_corr(x, y):
    """Placeholder seguro (não usado por padrão)."""
    return np.nan


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
    safe_key = re.sub(r"[^a-z0-9_]+", "_", str(base_name).lower())
    st.download_button(
        "📥 Baixar CSV",
        csv,
        file_name=f"{base_name}.csv",
        mime="text/csv",
        key=f"dl_{safe_key}",
    )

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
        raise RuntimeError("Informe o repositório no formato 'owner/repo'. Ex.: 'emiliobneto/UrbanTechCluster'")
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

def build_raw_url(ownerrepo: str, path: str, branch: str) -> str:
    ownerrepo = normalizerepo(ownerrepo).strip("/")
    path = path.lstrip("/")
    return f"{RAW_BASE}/{ownerrepo}/{branch}/{path}"


@st.cache_data(show_spinner=False)
def github_listdir(ownerrepo: str, path: str, branch: str):
    ownerrepo = normalizerepo(ownerrepo)
    url = f"{API_BASE}/repos/{ownerrepo}/contents/{path}?ref={branch}"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    if r.status_code != 200:
        return []
    return r.json()


@st.cache_data(show_spinner=True)
def github_get_contents(ownerrepo: str, path: str, branch: str):
    ownerrepo = normalizerepo(ownerrepo)
    url = f"{API_BASE}/repos/{ownerrepo}/contents/{path}?ref={branch}"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Falha listando {path}: {r.status_code} {r.text}")
    return r.json()


@st.cache_data(show_spinner=True)
def github_fetch_bytes(ownerrepo: str, path: str, branch: str) -> bytes:
    meta = github_get_contents(ownerrepo, path, branch)
    download_url = meta.get("download_url") or build_raw_url(ownerrepo, path, branch)
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
def load_gpkg(ownerrepo: str, path: str, branch: str, layer: str | None = None):
    try:
        import geopandas as gpd
    except Exception as e:
        raise RuntimeError("geopandas/pyogrio são necessários para ler GPKG.") from e
    blob = github_fetch_bytes(ownerrepo, path, branch)
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
def load_parquet(ownerrepo: str, path: str, branch: str) -> pd.DataFrame:
    blob = github_fetch_bytes(ownerrepo, path, branch)
    return pd.read_parquet(io.BytesIO(blob), engine="pyarrow")


@st.cache_data(show_spinner=True)
def load_csv(ownerrepo, path, branch) -> pd.DataFrame:
    blob = github_fetch_bytes(ownerrepo, path, branch)
    return pd.read_csv(io.BytesIO(blob), usecols=lambda c: not str(c).startswith("Unnamed"))

def list_files(ownerrepo: str, path: str, branch: str, exts=(".parquet", ".csv", ".gpkg")):
    items = github_listdir(ownerrepo, path, branch)
    out = []
    for it in items:
        if isinstance(it, dict) and it.get("type") == "file":
            nm = it["name"]
            if any(nm.lower().endswith(e) for e in exts):
                out.append({"path": f"{path.rstrip('/')}/{nm}", "name": nm})
    return out


@st.cache_data(show_spinner=True)
def github_branch_info(ownerrepo: str, branch: str):
    ownerrepo = normalizerepo(ownerrepo)
    url = f"{API_BASE}/repos/{ownerrepo}/branches/{branch}"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Falha lendo branch {branch}: {r.status_code} {r.text}")
    return r.json()


@st.cache_data(show_spinner=True)
def github_tree_paths(ownerrepo: str, branch: str):
    info = github_branch_info(ownerrepo, branch)
    tree_sha = info["commit"]["commit"]["tree"]["sha"]
    url = f"{API_BASE}/repos/{normalizerepo(ownerrepo)}/git/trees/{tree_sha}?recursive=1"
    r = requests.get(url, headers=_gh_headers(), timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Falha lendo tree: {r.status_code} {r.text}")
    tree = r.json().get("tree", [])
    return [ent["path"] for ent in tree if ent.get("type") == "blob"]


def pick_existing_dir(ownerrepo: str, branch: str, candidates: list[str]) -> str:
    """Tenta encontrar diretório existente (case-insensitive / alternativas)."""
    for cand in candidates:
        items = github_listdir(ownerrepo, cand, branch)
        if items:
            return cand
    all_paths = github_tree_paths(ownerrepo, branch)
    for cand in candidates:
        key = cand.strip("/").lower()
        for p in all_paths:
            if p.lower().startswith(key):
                parts = p.split("/")
                return "/".join(parts[: len(key.split("/"))])
    return candidates[0]

@st.cache_data(show_spinner=True, ttl=3600, max_entries=6)
def _preload_cluster_metrics_by_year(
    df_vals_raw: pd.DataFrame,
    df_est_raw: pd.DataFrame,
    cluster_col: str,
    *,
    chunk_size: int = 60,
) -> pd.DataFrame:
    # --- detectar colunas chave ---
    sq_vals = next((c for c in df_vals_raw.columns if str(c).upper() == "SQ"), None)
    ano_vals = next((c for c in df_vals_raw.columns if str(c).lower() in ("ano", "year")), None)
    if sq_vals is None:
        raise RuntimeError("Arquivo de valores não possui coluna 'SQ'.")
    sq_est  = next((c for c in df_est_raw.columns  if str(c).upper() == "SQ"), None)
    ano_est = next((c for c in df_est_raw.columns  if str(c).lower() in ("ano", "year")), None)
    if sq_est is None:
        raise RuntimeError("Arquivo de clusters não possui coluna 'SQ'.")

    # normaliza SQ
    def _norm_sq_series(s: pd.Series, digits: int = 6) -> pd.Series:
        s = s.astype("string").str.replace(r"\D", "", regex=True).fillna("")
        s = s.str[-digits:].str.zfill(digits)
        return s.mask(s.eq(""))

    vals = df_vals_raw.copy()
    vals["_SQ_norm"] = _norm_sq_series(vals[sq_vals])

    est  = df_est_raw.copy()
    est["_SQ_norm"] = _norm_sq_series(est[sq_est])

    # cluster como código 0..3 quando possível
    import re as _re
    def _to_int_code(x):
        try:
            v = float(str(x).strip())
            if np.isfinite(v) and abs(v - int(v)) < 1e-9:
                return int(v)
        except Exception:
            pass
        m = _re.search(r"\d+", str(x))
        return int(m.group(0)) if m else None

    est["_cl_code"] = est[cluster_col].map(_to_int_code).astype("Int64")

    # interseção de anos (ou None)
    if ano_vals and ano_est:
        anos = sorted(
            set(pd.to_numeric(vals[ano_vals], errors="coerce").dropna().astype(int))
            & set(pd.to_numeric(est[ano_est],  errors="coerce").dropna().astype(int))
        )
    elif ano_vals:
        anos = sorted(pd.to_numeric(vals[ano_vals], errors="coerce").dropna().astype(int).unique().tolist())
    elif ano_est:
        anos = sorted(pd.to_numeric(est[ano_est], errors="coerce").dropna().astype(int).unique().tolist())
    else:
        anos = [None]  # sem ano

    # selecionar variáveis numéricas válidas
    id_like   = {c for c in vals.columns if str(c).lower() in {"sq","id","codigo","code","_sq_norm"}}
    time_like = {c for c in vals.columns if str(c).lower() in {"ano","year"}}
    num_cols  = [c for c in vals.columns if pd.api.types.is_numeric_dtype(vals[c])]
    var_all   = [c for c in num_cols if c not in (id_like | time_like)]

    out_frames = []

    for ano in anos:
        # filtra ano em ambos os lados (quando existir)
        v = vals if (ano is None or not ano_vals) else vals[pd.to_numeric(vals[ano_vals], errors="coerce").astype("Int64") == ano]
        e = est  if (ano is None or not ano_est) else est [pd.to_numeric(est [ano_est ], errors="coerce").astype("Int64") == ano]
        # pega a melhor linha por SQ (caso tenha repetição por merges)
        e = e.sort_values(["_SQ_norm","_cl_code"]).drop_duplicates("_SQ_norm", keep="last")

        # mapa SQ->cluster
        mapper = e.set_index("_SQ_norm")["_cl_code"]

        # processa em blocos de variáveis para economizar memória
        for i in range(0, len(var_all), chunk_size):
            chunk = var_all[i:i+chunk_size]
            d = v[["_SQ_norm"] + chunk].copy()
            d[chunk] = d[chunk].apply(pd.to_numeric, errors="coerce")
            d["_cl_code"] = d["_SQ_norm"].map(mapper).astype("Int64")
            d = d[d["_cl_code"].isin([0,1,2,3])]  # clusters conhecidos

            if d.empty:
                continue

            g = d.groupby("_cl_code", observed=True)

            n_df     = g[chunk].count()
            miss_df  = pd.DataFrame(g.size().values[:,None] - n_df.values, index=n_df.index, columns=chunk)
            mean_df  = g[chunk].mean()
            med_df   = g[chunk].median()
            std_df   = g[chunk].std(ddof=1)
            min_df   = g[chunk].min()
            max_df   = g[chunk].max()
            q_df     = g[chunk].quantile([0.25, 0.75])
            p25_df   = q_df.xs(0.25, level=1)
            p75_df   = q_df.xs(0.75, level=1)
            cv_df    = std_df/mean_df

            def _to_long(name, df_):
                return (
                    df_.stack()
                       .rename(name)
                       .reset_index()
                       .rename(columns={"_cl_code":"cluster","level_1":"variavel"})
                )

            parts = [
                _to_long("n",             n_df),
                _to_long("missings",      miss_df),
                _to_long("media",         mean_df),
                _to_long("mediana",       med_df),
                _to_long("desvio_padrao", std_df),
                _to_long("p25",           p25_df),
                _to_long("p75",           p75_df),
                _to_long("minimo",        min_df),
                _to_long("maximo",        max_df),
                _to_long("coef_var",      cv_df),
            ]
            merged = parts[0]
            for p in parts[1:]:
                merged = merged.merge(p, on=["cluster","variavel"], how="left")
            merged.insert(0, "ano", ano)
            out_frames.append(merged)

    if not out_frames:
        return pd.DataFrame(columns=["ano","cluster","variavel","n","missings","media","mediana","desvio_padrao","p25","p75","minimo","maximo","coef_var"])

    out = pd.concat(out_frames, ignore_index=True)
    out["cluster_label"] = out["cluster"].map({
        0: "0 – Ausência de clusterização",
        1: "1 – Cluster em estágio inicial",
        2: "2 – Cluster em formação",
        3: "3 – Clusterizado",
    })
    return out



# ---------- DESCRITIVA ----------
@st.cache_data(show_spinner=True, max_entries=12)
def _compute_descritiva_fast(
    df_join: pd.DataFrame,
    vars_list: tuple[str, ...],
    use_shapiro: bool,
    shapiro_max_n: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Descritiva por cluster 0–3, vetorizada (sem loops), com Shapiro opcional."""
    vars_list = list(vars_list)
    d = df_join[["_cl_code"] + vars_list].copy()
    for v in vars_list:
        d[v] = pd.to_numeric(d[v], errors="coerce")

    g = d.groupby("_cl_code", observed=True)

    n_total = g.size()
    count_df = g[vars_list].count()
    miss_df = pd.DataFrame(
        n_total.values[:, None] - count_df.values,
        index=count_df.index,
        columns=vars_list,
    )

    mean_df   = g[vars_list].mean()
    median_df = g[vars_list].median()
    std_df    = g[vars_list].std(ddof=1)
    min_df    = g[vars_list].min()
    max_df    = g[vars_list].max()
    q_df      = g[vars_list].quantile([0.25, 0.75])
    p25_df    = q_df.xs(0.25, level=1)
    p75_df    = q_df.xs(0.75, level=1)
    cv_df     = std_df / mean_df

    def _to_long(name, df_):
        return (
            df_.stack()
               .rename(name)
               .reset_index()
               .rename(columns={"_cl_code": "cluster", "level_1": "variavel"})
        )

    parts = [
        _to_long("n",         count_df),
        _to_long("missings",  miss_df),
        _to_long("media",     mean_df),
        _to_long("mediana",   median_df),
        _to_long("desvio_padrao", std_df),
        _to_long("p25",       p25_df),
        _to_long("p75",       p75_df),
        _to_long("minimo",    min_df),
        _to_long("maximo",    max_df),
        _to_long("coef_var",  cv_df),
    ]
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=["cluster", "variavel"], how="left")

    # Shapiro (opcional, com cap)
    out["shapiro_p"] = np.nan
    out["shapiro_sig"] = ""
    if use_shapiro and _shapiro_fn is not None:
        rng = np.random.default_rng(random_state)
        for cl, sub in d.groupby("_cl_code", observed=True):
            arr = sub[vars_list].to_numpy(dtype=float)
            for j, v in enumerate(vars_list):
                col = arr[:, j]
                col = col[np.isfinite(col)]
                n = col.size
                if n < 3:
                    continue
                if n > shapiro_max_n:
                    idx = rng.choice(n, size=shapiro_max_n, replace=False)
                    col = col[idx]
                try:
                    p = float(_shapiro_fn(col).pvalue)
                except Exception:
                    p = np.nan
                mask = (out["cluster"] == cl) & (out["variavel"] == v)
                out.loc[mask, "shapiro_p"] = p

        def _sig_index(p):
            if not np.isfinite(p): return ""
            return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("•" if p < 0.10 else "ns")))
        out["shapiro_sig"] = out["shapiro_p"].apply(_sig_index)

    out["cluster_label"] = out["cluster"].map({
        0: "0 – Ausência de clusterização",
        1: "1 – Cluster em estágio inicial",
        2: "2 – Cluster em formação",
        3: "3 – Clusterizado",
    })
    return out


@st.cache_data(show_spinner=True, max_entries=12)
def _compute_omnibus_fast(
    df_join: pd.DataFrame,
    vars_list: tuple[str, ...],
    calc_gini: bool = False,
) -> pd.DataFrame:
    """ANOVA/Kruskal por variável + correlações com o código do cluster, em bloco."""
    vars_list = list(vars_list)
    d = df_join[["_cl_code"] + vars_list].copy()
    for v in vars_list:
        d[v] = pd.to_numeric(d[v], errors="coerce")

    code = d["_cl_code"].astype(float)
    X = d[vars_list]

    pearson_r  = X.corrwith(code, method="pearson")
    spearman_r = X.corrwith(code, method="spearman")
    r2_simple  = (pearson_r ** 2).rename("r2_simple")

    # p de Spearman (loop leve)
    if _spearman_fn is not None:
        spearman_p = []
        for v in vars_list:
            x = pd.to_numeric(d[v], errors="coerce")
            try:
                _, p = _spearman_fn(x, code, nan_policy="omit")
                spearman_p.append(float(p))
            except Exception:
                spearman_p.append(np.nan)
    else:
        spearman_p = [np.nan] * len(vars_list)

    # arrays por cluster (reuso)
    groups = {}
    for c in [0, 1, 2, 3]:
        sub = d.loc[d["_cl_code"] == c, vars_list].to_numpy(dtype=float)
        groups[c] = sub

    rows = []
    for j, v in enumerate(vars_list):
        g = [groups[c][:, j] for c in [0, 1, 2, 3]]
        g = [arr[np.isfinite(arr)] for arr in g]
        n_total = int(sum(len(arr) for arr in g))

        if _anova_fn is not None and n_total >= 4:
            try:
                Fv, pA = _anova_fn(*g)
                Fv, pA = float(Fv), float(pA)
            except Exception:
                Fv, pA = np.nan, np.nan
        else:
            Fv, pA = np.nan, np.nan

        if _kruskal_fn is not None:
            non_empty = [arr for arr in g if len(arr)]
            try:
                Hv, pK = _kruskal_fn(*non_empty) if len(non_empty) >= 2 else (np.nan, np.nan)
                Hv, pK = float(Hv), float(pK)
            except Exception:
                Hv, pK = np.nan, np.nan
        else:
            Hv, pK = np.nan, np.nan

        eta2 = _eta_squared(g)

        if calc_gini:
            try:
                gini = _gini_corr(pd.to_numeric(d[v], errors="coerce"), code)
            except Exception:
                gini = np.nan
        else:
            gini = np.nan

        rows.append({
            "variavel": v,
            "n_total": n_total,
            "anova_F": Fv, "anova_p": pA,
            "eta2": eta2,
            "kruskal_H": Hv, "kruskal_p": pK,
            "spearman_rho": float(spearman_r.get(v, np.nan)),
            "spearman_p":  float(spearman_p[j]),
            "gini_corr": gini,
            "r2_simple":  float(r2_simple.get(v, np.nan)),
        })

    out = pd.DataFrame(rows)
    def _sig_index(p):
        if not np.isfinite(p): return ""
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("•" if p < 0.10 else "ns")))
    out["anova_sig"]    = out["anova_p"].apply(_sig_index)
    out["kruskal_sig"]  = out["kruskal_p"].apply(_sig_index)
    out["spearman_sig"] = out["spearman_p"].apply(_sig_index)
    return out


@st.cache_data(show_spinner=True, max_entries=12)
def _compute_pairs_fast(
    df_join: pd.DataFrame,
    vars_list: tuple[str, ...],
) -> pd.DataFrame:
    """Comparações par-a-par (Welch t-test, Mann–Whitney, d de Cohen, Cliff’s Δ) com pré-split por cluster."""
    vars_list = list(vars_list)
    d = df_join[["_cl_code"] + vars_list].copy()
    for v in vars_list:
        d[v] = pd.to_numeric(d[v], errors="coerce")

    arrays = {c: d.loc[d["_cl_code"] == c, vars_list].to_numpy(dtype=float) for c in [0, 1, 2, 3]}
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

    rows = []
    for a, b in pairs:
        A = arrays[a]; B = arrays[b]
        for j, v in enumerate(vars_list):
            xa = A[:, j]; xa = xa[np.isfinite(xa)]
            xb = B[:, j]; xb = xb[np.isfinite(xb)]
            nA, nB = xa.size, xb.size
            muA = float(np.mean(xa)) if nA else np.nan
            muB = float(np.mean(xb)) if nB else np.nan
            medA = float(np.median(xa)) if nA else np.nan
            medB = float(np.median(xb)) if nB else np.nan

            if _ttest_fn is not None and nA >= 2 and nB >= 2:
                try:
                    t_stat, p_t = _ttest_fn(xa, xb, equal_var=False)
                    t_stat, p_t = float(t_stat), float(p_t)
                except Exception:
                    t_stat, p_t = np.nan, np.nan
            else:
                t_stat, p_t = np.nan, np.nan

            if _mw_fn is not None and nA >= 1 and nB >= 1:
                try:
                    U, p_mw = _mw_fn(xa, xb, alternative="two-sided")
                    U, p_mw = float(U), float(p_mw)
                except Exception:
                    U, p_mw = np.nan, np.nan
            else:
                U, p_mw = np.nan, np.nan

            d_eff = _cohens_d(xa, xb)
            cd    = _cliffs_delta(xa, xb)

            rows.append({
                "variavel": v,
                "par": f"{a} vs {b}",
                "cluster_A": a, "cluster_B": b,
                "n_A": nA, "n_B": nB,
                "media_A": muA, "media_B": muB,
                "mediana_A": medA, "mediana_B": medB,
                "t_stat": t_stat, "p_t": p_t,
                "U": U, "p_mw": p_mw,
                "cohens_d": d_eff, "cliffs_delta": cd,
            })

    out = pd.DataFrame(rows)
    if "p_t" in out.columns:
        out["p_t_fdr_bh"] = _bh_fdr(out["p_t"])
        out["t_sig"]      = out["p_t"].apply(lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("•" if p < 0.10 else "ns"))))
        out["t_sig_fdr"]  = out["p_t_fdr_bh"].apply(lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("•" if p < 0.10 else "ns"))))
    if "p_mw" in out.columns:
        out["p_mw_fdr_bh"] = _bh_fdr(out["p_mw"])
        out["mw_sig"]      = out["p_mw"].apply(lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("•" if p < 0.10 else "ns"))))
        out["mw_sig_fdr"]  = out["p_mw_fdr_bh"].apply(lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("•" if p < 0.10 else "ns"))))
    out["cluster_A_label"] = out["cluster_A"].map({
        0: "0 – Ausência de clusterização",
        1: "1 – Cluster em estágio inicial",
        2: "2 – Cluster em formação",
        3: "3 – Clusterizado",
    })
    out["cluster_B_label"] = out["cluster_B"].map({
        0: "0 – Ausência de clusterização",
        1: "1 – Cluster em estágio inicial",
        2: "2 – Cluster em formação",
        3: "3 – Clusterizado",
    })
    return out
# ==========================
# CORES / CLASSIF / MAPAS / LEGENDAS
# ==========================

def hex_to_rgba(hex_color, alpha: int = 180):
    """Converte #RRGGBB ou #RGB em [r,g,b,alpha]. Tolera NaN/None/strings inválidas."""
    try:
        if not isinstance(hex_color, str):
            return [153, 153, 153, alpha]  # cinza padrão
        h = hex_color.strip().lstrip("#")
        if len(h) == 3:  # #abc -> #aabbcc
            h = "".join(ch * 2 for ch in h)
        if len(h) != 6:
            return [153, 153, 153, alpha]
        r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
        return [r, g, b, alpha]
    except Exception:
        return [153, 153, 153, alpha]



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
    try:
        import geopandas as gpd
    except Exception:
        raise RuntimeError("geopandas é necessário para montar GeoJSON.")

    # Se não for GeoDataFrame, mas tiver coluna 'geometry', converte
    if not isinstance(gdf, gpd.GeoDataFrame):
        if "geometry" in getattr(gdf, "columns", []):
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=getattr(gdf, "crs", 4326))
        else:
            raise RuntimeError("Objeto sem coluna 'geometry' para gerar GeoJSON.")
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
        get_fill_color="properties.fill_color",
        get_point_radius=60,          # <- em GeoJsonLayer é get_point_radius
        auto_highlight=True,
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

def find_files_by_patterns(ownerrepo, branch, base_dirs, patterns=(), exts=(".csv", ".parquet")):
    """Procura arquivos dentro de múltiplos diretórios candidatos filtrando por padrões (substring/regex simples)."""
    found = []
    for base in base_dirs:
        base_dir = pick_existing_dir(ownerrepo, branch, [base])
        for f in list_files(ownerrepo, base_dir, branch, exts):
            name_low = f["name"].lower()
            ok = True if not patterns else any(re.search(p, name_low) for p in patterns)
            if ok:
                found.append({"path": f["path"], "name": f["name"], "base": base_dir})
    return found


def load_tabular(ownerrepo, path, branch):
    if path.lower().endswith(".parquet"):
        return load_parquet(ownerrepo, path, branch)
    return load_csv(ownerrepo, path, branch)


def pairs_to_matrix(df_pairs, i_col, j_col, val_col, sym_max=True):
    m = df_pairs.pivot(index=i_col, columns=j_col, values=val_col)
    if sym_max:
        m = m.combine_first(m.T)
        m = pd.DataFrame(np.maximum(m.values, m.T.values), index=m.index, columns=m.columns)
    return m

import io
import json
import unicodedata
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------------------
# Tab 4 — ANN: métricas, relatórios e mapas (Data/ANN)
# --------------------------------------------------------------------------------------
# Esta função DEPENDE dos helpers já existentes no seu app principal.
# Para evitar import circular, passamos os helpers como parâmetros.
# Veja no final deste arquivo o snippet de como chamá-la no main.
# --------------------------------------------------------------------------------------


# --------------------------- Utilidades locais ----------------------------------------

def _norm_text(x: str) -> str:
    """Normaliza strings: remove acentos e baixa caixa."""
    s = unicodedata.normalize("NFKD", str(x))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def _pick_dir_ann(repo, branch, pick_existing_dir):
    return pick_existing_dir(
        repo,
        branch,
        ["Data/ANN", "data/ANN", "data/ann", "Data/Ann", "ANN"],
    )


def _load_if_exists(repo, branch, list_files, load_csv, filename_globs: list[str]):
    """Procura 1º arquivo que case com os padrões (case-insensitive) dentro de Data/ANN."""
    base_dir = _pick_dir_ann(repo, branch, pick_existing_dir)
    files = list_files(repo, base_dir, branch, exts=(".csv", ".txt", ".json", ".parquet"))
    # tenta match por nome exato primeiro
    for pat in filename_globs:
        for f in files:
            if _norm_text(f["name"]) == _norm_text(pat):
                p = f["path"]
                if p.lower().endswith(".parquet"):
                    try:
                        return pd.read_parquet(io.BytesIO(load_parquet(repo, p, branch).to_parquet()))
                    except Exception:
                        pass
                if p.lower().endswith(".csv"):
                    return load_csv(repo, p, branch)
                if p.lower().endswith(".txt"):
                    # devolve bytes/texto cru dentro de DataFrame com col=__raw__
                    try:
                        from io import BytesIO
                        b = github_fetch_bytes(repo, p, branch)
                        return pd.DataFrame({"__raw__": [b.decode("utf-8", errors="replace")]})
                    except Exception:
                        return None
                if p.lower().endswith(".json"):
                    try:
                        b = github_fetch_bytes(repo, p, branch)
                        return pd.DataFrame({"__raw_json__": [json.loads(b.decode("utf-8", errors="replace"))]})
                    except Exception:
                        return None
    # senão tenta por "contains"
    for pat in filename_globs:
        for f in files:
            if _norm_text(pat) in _norm_text(f["name"]):
                p = f["path"]
                if p.lower().endswith(".parquet"):
                    try:
                        return pd.read_parquet(io.BytesIO(load_parquet(repo, p, branch).to_parquet()))
                    except Exception:
                        pass
                if p.lower().endswith(".csv"):
                    return load_csv(repo, p, branch)
                if p.lower().endswith(".txt"):
                    try:
                        b = github_fetch_bytes(repo, p, branch)
                        return pd.DataFrame({"__raw__": [b.decode("utf-8", errors="replace")]})
                    except Exception:
                        return None
                if p.lower().endswith(".json"):
                    try:
                        b = github_fetch_bytes(repo, p, branch)
                        return pd.DataFrame({"__raw_json__": [json.loads(b.decode("utf-8", errors="replace"))]})
                    except Exception:
                        return None
    return None


def _maybe_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan


def _parse_classificationreport_text(text: str) -> pd.DataFrame:
    """Extrai tabela do classificationreport (texto sklearn)."""
    lines = [l for l in text.splitlines() if l.strip()]
    rows = []
    for ln in lines:
        # Ex.: "classe    0.82    0.79    0.80    1234"
        parts = [p for p in ln.strip().split() if p]
        if len(parts) >= 5 and parts[0].lower() not in {"accuracy"}:
            label = " ".join(parts[:-4])
            prec, rec, f1, sup = parts[-4:]
            rows.append({
                "label": label,
                "precision": _maybe_float(prec),
                "recall": _maybe_float(rec),
                "f1": _maybe_float(f1),
                "support": _maybe_float(sup),
            })
        elif ln.lower().startswith("accuracy"):
            # linha "accuracy" costuma ser: accuracy 0.84 X 12345
            tokens = ln.split()
            try:
                acc = float(tokens[1])
                rows.append({"label": "accuracy", "precision": np.nan, "recall": np.nan, "f1": acc, "support": np.nan})
            except Exception:
                pass
    return pd.DataFrame(rows)


def _cols_detect_epoch_metrics(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    epoch = cols.get("epoch") or next((c for c in df.columns if _norm_text(c) in {"epoca", "epocas"}), None)
    # mapeia métricas comuns
    metrics = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["loss", "accuracy", "acc", "auc", "precision", "recall", "f1", "mae", "mse", "rmse"]):
            metrics.append(c)
    return epoch, sorted(set(metrics))


def _cat_palette_from_values(values, pick_categorical):
    cats = [str(v) for v in pd.Series(values).dropna().unique().tolist()]
    palette = pick_categorical(len(cats))
    return {cats[i]: palette[i] for i in range(len(cats))}


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


def render_variancia_file(df: pd.DataFrame):
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


def render_pipeline_file(df: pd.DataFrame):
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


def render_evr_section(df_evr: pd.DataFrame):
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
        download_plotly_png(fig,  "pca_evr_bar")
    with c2:
        fig2 = px.line(
            df, x="component", y="cumulative", markers=True, title="Variância explicada acumulada"
        )
        st.plotly_chart(fig2, use_container_width=True)
        download_plotly_png(fig2, "pca_evr_cumulative")

    st.subheader("Tabela — Variância explicada")
    st.dataframe(df, use_container_width=True)


def render_loadings_section(df_load: pd.DataFrame):
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


def render_scores_section(df_scores: pd.DataFrame, repo, branch, pick_existing_dir, list_files, load_parquet, load_csv):
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
        render_variancia_file(df_evr)
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
        render_pipeline_file(df_pipe)
    else:
        st.info("Arquivo não reconhecido como pipeline. Exibindo preview:")
        st.dataframe(df_pipe.head(), use_container_width=True)

# --------------------------- Função principal -----------------------------------------

# ==========================
# PATCH — Aba 4 (ANN) com seletor de pasta em Data/ANN
# ==========================

def _load_if_exists_in(repo, branch, list_files, load_csv, base_dir: str, filename_globs: list[str]):
    """
    Igual ao _load_if_exists original, mas RECEBE explicitamente o base_dir
    (ex.: Data/ANN/04_ANN_sem_padronizacao_force90).
    """
    files = list_files(repo, base_dir, branch, exts=(".csv", ".txt", ".json", ".parquet"))

    # 1) nome exato
    for pat in filename_globs:
        for f in files:
            if _norm_text(f["name"]) == _norm_text(pat):
                p = f["path"]
                if p.lower().endswith(".parquet"):
                    try:
                        return load_parquet(repo, p, branch)
                    except Exception:
                        return None
                if p.lower().endswith(".csv"):
                    return load_csv(repo, p, branch)
                if p.lower().endswith(".txt"):
                    try:
                        b = github_fetch_bytes(repo, p, branch)
                        return pd.DataFrame({"__raw__": [b.decode("utf-8", errors="replace")]})
                    except Exception:
                        return None
                if p.lower().endswith(".json"):
                    try:
                        b = github_fetch_bytes(repo, p, branch)
                        return pd.DataFrame({"__raw_json__": [json.loads(b.decode("utf-8", errors="replace"))]})
                    except Exception:
                        return None
    # 2) contains
    for pat in filename_globs:
        for f in files:
            if _norm_text(pat) in _norm_text(f["name"]):
                p = f["path"]
                if p.lower().endswith(".parquet"):
                    try:
                        return load_parquet(repo, p, branch)
                    except Exception:
                        return None
                if p.lower().endswith(".csv"):
                    return load_csv(repo, p, branch)
                if p.lower().endswith(".txt"):
                    try:
                        b = github_fetch_bytes(repo, p, branch)
                        return pd.DataFrame({"__raw__": [b.decode("utf-8", errors="replace")]})
                    except Exception:
                        return None
                if p.lower().endswith(".json"):
                    try:
                        b = github_fetch_bytes(repo, p, branch)
                        return pd.DataFrame({"__raw_json__": [json.loads(b.decode("utf-8", errors="replace"))]})
                    except Exception:
                        return None
    return None


def render_ann_tab(
    *,
    repo,
    branch,
    # helpers obrigatórios vindos do app principal
    pick_existing_dir,
    list_files,
    load_parquet,
    load_csv,
    load_gpkg,
    github_fetch_bytes,
    make_geojson,
    ensure_wgs84,
    hex_to_rgba,
    pick_categorical,
    render_geojson_layer,
    render_line_layer,
    render_point_layer,
    osm_basemap_deck,
    deck,
):
    """Renderiza toda a **Aba 4 — ANN** lendo de `Data/ANN/<EXECUCAO>` escolhido na UI."""
    st.subheader("🧠 Rede Neural — Métricas e Resultados (Data/ANN)")

    # ---------------- Seletor de pasta de execução dentro de Data/ANN ----------------
    ann_root = pick_existing_dir(repo, branch, ["Data/ANN", "data/ANN", "data/ann", "Data/Ann", "ANN"])
    try:
        items = github_listdir(repo, ann_root, branch)
        run_dirs = [it["name"] for it in items if isinstance(it, dict) and it.get("type") == "dir"]
    except Exception:
        run_dirs = []
    if not run_dirs:
        st.warning(f"Não encontrei subpastas dentro de `{ann_root}`. Usando o próprio diretório.")
    run_sel = st.selectbox(
        "📁 Execução (pasta dentro de Data/ANN)",
        options=(["(raiz)"] + run_dirs) if run_dirs else ["(raiz)"],
        index=0,
        key="ann_run_dir",
        help="Escolha a pasta da execução (ex.: 04_ANN_sem_padronizacao_force90).",
    )
    ann_base = ann_root if run_sel == "(raiz)" else f"{ann_root}/{run_sel}"
    st.caption(f"Lendo arquivos de: `{ann_base}`")

    # ==================================================================================
    # 1) Histórico por época (val_metrics_per_epoch.csv / keras_history.csv / metrics_over_epochs.csv)
    # ==================================================================================
    st.markdown("### 📈 Evolução por época")
    
    def _is_loss_like(name: str) -> bool:
        n = str(name).lower()
        return any(k in n for k in ["loss", "mae", "mse", "rmse"])
    
    history_candidates = [
        "val_metrics_per_epoch.csv",
        "metrics_over_epochs.csv",
        "keras_history.csv",
    ]
    
    any_history = False
    hist_summary_rows = []   # ← resumo para tabela ao final
    
    for name in history_candidates:
        df_hist = _load_if_exists_in(repo, branch, list_files, load_csv, ann_base, [name])
        if not (isinstance(df_hist, pd.DataFrame) and not df_hist.empty and "__raw__" not in df_hist.columns):
            continue
    
        any_history = True
        st.markdown(f"**Arquivo:** `{name}`")
        epoch_col, metric_cols = _cols_detect_epoch_metrics(df_hist)
    
        if not metric_cols:
            st.info("Nenhuma métrica reconhecida.")
            st.divider()
            continue
    
        # chave base única para evitar StreamlitDuplicateElementId
        keybase = f"{run_sel or 'root'}_{name}"
    
        for m in metric_cols:
            # tenta achar a versão 'val_' da mesma métrica
            val_col = None
            for alt in [f"val_{m}", f"val-{m}", f"val{m}"]:
                if alt in df_hist.columns:
                    val_col = alt
                    break
    
            ycols = [m] + ([val_col] if val_col else [])
            fig = px.line(
                df_hist,
                x=epoch_col or df_hist.index,
                y=ycols,
                markers=True,
                title=f"{m} por época",
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"plt_hist_{keybase}_{m}".replace(" ", "_")
            )
    
            # ---------- coleta de resumo ----------
            dir_min = _is_loss_like(m)  # True → minimizar; False → maximizar
            ep = (df_hist[epoch_col] if epoch_col else pd.Series(np.arange(len(df_hist))))
            # treino
            s_tr = pd.to_numeric(df_hist[m], errors="coerce")
            if s_tr.notna().any():
                idx_best_tr = int((s_tr.idxmin() if dir_min else s_tr.idxmax()))
                hist_summary_rows.append({
                    "arquivo": name,
                    "metrica": m,
                    "epocas": int(len(df_hist)),
                    "treino_ultimo": float(s_tr.iloc[-1]),
                    "treino_melhor": float(s_tr.min() if dir_min else s_tr.max()),
                    "treino_ep_melhor": int(ep.iloc[idx_best_tr]),
                })
            # validação (se houver)
            if val_col:
                s_val = pd.to_numeric(df_hist[val_col], errors="coerce")
                if s_val.notna().any():
                    idx_best_val = int((s_val.idxmin() if dir_min else s_val.idxmax()))
                    last_row = next((r for r in reversed(hist_summary_rows) if r["metrica"] == m and r["arquivo"] == name), None)
                    if last_row is not None:
                        last_row.update({
                            "val_ultimo": float(s_val.iloc[-1]),
                            "val_melhor": float(s_val.min() if dir_min else s_val.max()),
                            "val_ep_melhor": int(ep.iloc[idx_best_val]),
                        })
    
        st.divider()
    
    if not any_history:
        st.info("Não encontrei arquivos de histórico por época nesta execução.")
    else:
        # ---------- Tabela de resumo após os gráficos ----------
        df_hist_sum = pd.DataFrame(hist_summary_rows)
        if not df_hist_sum.empty:
            # ordena por arquivo/metric
            df_hist_sum = df_hist_sum.sort_values(["arquivo", "metrica"]).reset_index(drop=True)
            st.markdown("**Resumo — métricas por época (último vs. melhor)**")
            st.dataframe(df_hist_sum, use_container_width=True)
            download_df(df_hist_sum, f"history_summary_{(run_sel or 'root')}")

    # ==================================================================================
    # 2) AUC por classe
    # ==================================================================================
    st.markdown("### 📊 AUC por classe")
    df_auc = _load_if_exists_in(repo, branch, list_files, load_csv, ann_base, ["auc_summary.csv", "roc_auc.csv"])
    if isinstance(df_auc, pd.DataFrame) and not df_auc.empty and "__raw__" not in df_auc.columns:
        cols = {c.lower(): c for c in df_auc.columns}
        class_col = cols.get("class") or cols.get("label") or cols.get("classe") or list(df_auc.columns)[0]
        auc_col = cols.get("auc") or cols.get("roc_auc") or list(df_auc.columns)[1]
        fig = px.bar(df_auc, x=class_col, y=auc_col, title="AUC por classe")
        st.plotly_chart(fig, use_container_width=True, key=f"plt_auc_{run_sel or 'root'}")
        st.dataframe(df_auc, use_container_width=True)
    else:
        st.info("auc_summary.csv não encontrado nesta execução.")

    # ==================================================================================
    # 3) Classification report
    # ==================================================================================
    st.markdown("### 🧾 Classification report")
    df_cr_raw = _load_if_exists_in(
        repo, branch, list_files, load_csv, ann_base, ["classificationreport.txt", "classificationreport.json"]
    )
    if isinstance(df_cr_raw, pd.DataFrame) and not df_cr_raw.empty:
        if "__raw_json__" in df_cr_raw.columns:
            data = df_cr_raw["__raw_json__"].iloc[0]
            try:
                df_cr = pd.DataFrame(data).T.reset_index().rename(columns={"index": "label"})
            except Exception:
                df_cr = pd.json_normalize(data)
        elif "__raw__" in df_cr_raw.columns:
            text = df_cr_raw["__raw__"].iloc[0]
            df_cr = _parse_classificationreport_text(text)
        else:
            df_cr = df_cr_raw.copy()
        if not df_cr.empty and "label" in df_cr.columns:
            st.dataframe(df_cr, use_container_width=True)
            if "f1" in df_cr.columns:
                fig = px.bar(df_cr[df_cr["label"].str.lower() != "accuracy"], x="label", y="f1", title="F1-score por classe")
                st.plotly_chart(fig, use_container_width=True, key=f"plt_cls_{run_sel or 'root'}")
        else:
            st.info("Não consegui extrair a tabela do classificationreport.")
    else:
        st.info("classificationreport (txt/json) não encontrado nesta execução.")

    # ==================================================================================
    # 4) Testes de hipótese
    # ==================================================================================
    st.markdown("### 🧪 Testes de hipótese")
    df_ht = _load_if_exists_in(repo, branch, list_files, load_csv, ann_base, ["hypothesis_tests.csv"])
    if isinstance(df_ht, pd.DataFrame) and not df_ht.empty and "__raw__" not in df_ht.columns:
        cols = {c.lower(): c for c in df_ht.columns}
        name_col = cols.get("metric") or cols.get("name") or cols.get("teste") or list(df_ht.columns)[0]
        p_col = cols.get("p") or cols.get("pvalue") or cols.get("p_value") or cols.get("p-val")
        if p_col:
            df_ht["mlog10"] = -np.log10(pd.to_numeric(df_ht[p_col], errors="coerce").clip(lower=1e-300))
            fig = px.bar(df_ht.sort_values("mlog10", ascending=False), x=name_col, y="mlog10", title="-log10(p)")
            fig.add_hline(y=-np.log10(0.05))
            st.plotly_chart(fig, use_container_width=True, key=f"plt_ht_{run_sel or 'root'}")
        st.dataframe(df_ht, use_container_width=True)
    else:
        st.info("hypothesis_tests.csv não encontrado nesta execução.")

    # ==================================================================================
    # 5) Resumo de métricas
    # ==================================================================================
    st.markdown("### 🧮 Resumo de métricas")
    df_ms = _load_if_exists_in(repo, branch, list_files, load_csv, ann_base, ["metrics_summary.csv"])
    if isinstance(df_ms, pd.DataFrame) and not df_ms.empty and "__raw__" not in df_ms.columns:
        st.dataframe(df_ms, use_container_width=True)
        num_cols = [c for c in df_ms.columns if pd.api.types.is_numeric_dtype(df_ms[c])]
        if len(num_cols) >= 1:
            melted = df_ms.melt(id_vars=[c for c in df_ms.columns if c not in num_cols],
                                value_vars=num_cols, var_name="metric", value_name="value")
            if "metric" in melted.columns and "value" in melted.columns:
                fig = px.bar(melted, x="metric", y="value", title="Métricas (agregadas)")
                st.plotly_chart(fig, use_container_width=True, key=f"plt_ms_{run_sel or 'root'}")
    else:
        st.info("metrics_summary.csv não encontrado nesta execução.")

    # ==================================================================================
    # 6) Configs e scaler
    # ==================================================================================
    st.markdown("### ⚙️ Configurações e scaler")
    for fname in ["inference_config.json", "run_config.json", "scaler_params.json"]:
        df_cfg = _load_if_exists_in(repo, branch, list_files, load_csv, ann_base, [fname])
        if isinstance(df_cfg, pd.DataFrame) and not df_cfg.empty and "__raw_json__" in df_cfg.columns:
            st.markdown(f"**{fname}**")
            st.json(df_cfg["__raw_json__"].iloc[0])
            if fname == "scaler_params.json":
                params = df_cfg["__raw_json__"].iloc[0]
                for key in ["mean_", "scale_"]:
                    if key in params and isinstance(params[key], (list, tuple)) and len(params[key]) > 0:
                        dfp_ = pd.DataFrame({"feature": list(range(len(params[key]))), key: params[key]})
                        fig = px.bar(dfp_, x="feature", y=key, title=f"Scaler: {key}")
                        st.plotly_chart(fig, use_container_width=True, key=f"plt_scaler_{run_sel or 'root'}_{key}")
        else:
            st.info(f"{fname} não encontrado nesta execução.")

    # ==================================================================================
    # 7) Mapas lado a lado — Estágio vs Predicted (usa df25_com_previsoes.csv da execução)
    # ==================================================================================
    st.markdown("### 🗺️ Mapas — Estágio de clusterização × Predicted class")
    df_pred = _load_if_exists_in(repo, branch, list_files, load_csv, ann_base, ["df25_com_previsoes.csv"])
    if isinstance(df_pred, pd.DataFrame) and not df_pred.empty and "__raw__" not in df_pred.columns:
        # Detecta colunas
        cols = list(df_pred.columns)
        cols_norm = {_norm_text(c): c for c in cols}
        # SQ
        sq_col = next((c for c in cols if _norm_text(c) in {"sq"}), None)
        if sq_col is None:
            sq_col = next((c for c in cols if _norm_text(c) in {"id", "codigo", "code"}), None)
        # Estágio de clusterização
        est_col = next((c for c in cols if ("estagio" in _norm_text(c) and "cluster" in _norm_text(c))), None)
        if est_col is None:
            est_col = cols_norm.get("estagioclusterizacao") or cols_norm.get("estagio de clusterizacao")
        # Predicted class
        pred_col = next((c for c in cols if ("pred" in _norm_text(c) and "class" in _norm_text(c))), None)
        if pred_col is None:
            pred_col = cols_norm.get("predicted") or cols_norm.get("pred_class") or cols_norm.get("predicted class")

        if not sq_col:
            st.error("Não encontrei a coluna 'SQ' (ou equivalente) em df25_com_previsoes.csv.")
            return
        if not est_col or not pred_col:
            st.warning("Não encontrei 'Estágio de clusterização' e/ou 'Predicted class'. Exibindo preview.")
            st.dataframe(df_pred.head(), use_container_width=True)
            return

        # --- quadras com proteção + cache de sessão ---
        gdf_quadras = st.session_state.get("gdf_quadras_cached")
        if gdf_quadras is None or gdf_quadras.empty:
            try:
                gdf_quadras = load_gpkg(repo, "Data/mapa/quadras.gpkg", branch)
                st.session_state["gdf_quadras_cached"] = gdf_quadras
            except Exception as e:
                st.error(f"Falha carregando quadras.gpkg: {e}")
                return

        geom_name = gdf_quadras.geometry.name
        sq_geo_col = next((c for c in gdf_quadras.columns if _norm_text(c) == "sq"), None)
        if not sq_geo_col:
            st.error("A camada de quadras não possui coluna 'SQ'.")
            return

        # --- NORMALIZAÇÃO DO JOIN (evita erro de tipos) ---
        import re as re
        
        # GeoJSONs com cores
        def _geojson_with_fill(gdf_in: pd.DataFrame, value_col: str, cmap: dict):
            gj = make_geojson(gdf_in[[value_col, geom_name]].rename(columns={geom_name: "geometry"}))
            for feat in gj.get("features", []):
                v = feat.get("properties", {}).get(value_col, None)
                hexc = cmap.get(str(v), "#999999")
                feat.setdefault("properties", {})["fill_color"] = hex_to_rgba(hexc)
            return gj

        gj_est = _geojson_with_fill(gdf, "estagio", cmap_est)
        gj_pred = _geojson_with_fill(gdf, "predicted", cmap_pred)

        # Plano de fundo
        base = st.radio(
            "Plano de fundo",
            ["OpenStreetMap", "Satélite (Mapbox)"],
            index=0,
            horizontal=True,
            key="ann_maps_base",
        )
        
        c1, c2 = st.columns(2, gap="large")
        
        with c1:
            st.markdown("**Estágio de clusterização**")
            lyr_est = render_geojson_layer(gj_est, name="estagio")
            if base.startswith("Satélite"):
                deck([lyr_est], satellite=True)              # <- CORRETO: deck aceita 'satellite'
            else:
                osm_basemap_deck([lyr_est])                  # <- CORRETO: sem parâmetro 'satellite'
            st.markdown("**Legenda — Estágio**")
            for k, hexc in cmap_est.items():
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0'>"
                    f"<span style='display:inline-block;width:14px;height:14px;border:1px solid #0003;background:{hexc}'></span>"
                    f"<span>{k}</span></div>",
                    unsafe_allow_html=True,
                )
        
        with c2:
            st.markdown("**Predicted class**")
            lyr_pred = render_geojson_layer(gj_pred, name="predicted")
            if base.startswith("Satélite"):
                deck([lyr_pred], satellite=True)
            else:
                osm_basemap_deck([lyr_pred])
            st.markdown("**Legenda — Predicted**")
            for k, hexc in cmap_pred.items():
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0'>"
                    f"<span style='display:inline-block;width:14px;height:14px;border:1px solid #0003;background:{hexc}'></span>"
                    f"<span>{k}</span></div>",
                    unsafe_allow_html=True,
                )
        
        # =========================
        # Resumos após os "gráficos"
        # =========================
        st.markdown("#### 📋 Resumo — cobertura e frequências")
        
        # Cobertura do join (usa pelo menos uma das colunas não nulas)
        total_quadras = int(len(gdf_quadras))
        com_join = int(((gdf["estagio"].notna()) | (gdf["predicted"].notna())).sum())
        sem_join = total_quadras - com_join
        df_cobertura = pd.DataFrame(
            {
                "Métrica": ["Total de quadras", "SQ com match (join)", "SQ sem match"],
                "Valor": [total_quadras, com_join, sem_join],
            }
        )
        st.dataframe(df_cobertura, use_container_width=True)
        download_df(df_cobertura, "resumo_cobertura_mapas_ann")
        
        # Frequências por categoria (inclui NaN como categoria)
        freq_est = (
            gdf["estagio"].astype("string").fillna("(NaN)").value_counts(dropna=False)
            .rename_axis("estagio").reset_index(name="n")
        )
        freq_pred = (
            gdf["predicted"].astype("string").fillna("(NaN)").value_counts(dropna=False)
            .rename_axis("predicted").reset_index(name="n")
        )
        
        col_fe1, col_fe2 = st.columns(2, gap="large")
        with col_fe1:
            st.markdown("**Frequências — Estágio**")
            st.dataframe(freq_est, use_container_width=True)
            download_df(freq_est, "frequencias_estagio_ann")
        with col_fe2:
            st.markdown("**Frequências — Predicted**")
            st.dataframe(freq_pred, use_container_width=True)
            download_df(freq_pred, "frequencias_predicted_ann")

    else:
        st.info("df25_com_previsoes.csv não encontrado nesta execução.")

        # ==================================================================================
        # 8) (Opcional) Distribuições de score / matriz de confusão
        # ==================================================================================
        with st.expander("Extras — distribuições e matriz de confusão", expanded=False):
    
            # -- helper: encontra o primeiro CSV de predições com meta dentro de Data/ANN (ou subpasta escolhida)
            def _find_predictions_with_meta(repo, branch) -> tuple[pd.DataFrame | None, str | None]:
                cand_names = [
                    "test_predictions_with_meta.csv",
                    "predictions_df25_with_meta.csv",
                    "predictions_with_meta.csv",
                ]
    
                # 1) Se o usuário já selecionou uma subpasta de run antes (item 7), tente ali primeiro
                run_dir = st.session_state.get("ann_run_dir") or st.session_state.get("ann_subdir")  # nomes que podemos ter guardado
                if run_dir:
                    for nm in cand_names:
                        p = f"{run_dir.rstrip('/')}/{nm}"
                        try:
                            df = load_csv(repo, p, branch)
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                return df, p
                        except Exception:
                            pass
    
                # 2) Busca recursiva sob Data/ANN
                base_ann = _pick_dir_ann(repo, branch, pick_existing_dir).strip("/")
                try:
                    tree = github_tree_paths(repo, branch)
                except Exception:
                    tree = []
                for nm in cand_names:
                    for path in tree:
                        if path.lower().startswith(base_ann.lower() + "/") and path.lower().endswith("/" + nm.lower()):
                            try:
                                df = load_csv(repo, path, branch)
                                if isinstance(df, pd.DataFrame) and not df.empty:
                                    # guarda subpasta para uso das outras seções
                                    st.session_state["ann_run_dir"] = "/".join(path.split("/")[:-1])
                                    return df, path
                            except Exception:
                                continue
                return None, None
    
            df_meta, meta_path = _find_predictions_with_meta(repo, branch)
    
            if not (isinstance(df_meta, pd.DataFrame) and not df_meta.empty):
                st.caption("Arquivo de predições com meta não encontrado em `Data/ANN/**` "
                           "(procurei: test_predictions_with_meta.csv, predictions_df25_with_meta.csv, predictions_with_meta.csv).")
            else:
                st.caption(f"Fonte: `{meta_path}`")
    
                # -- detectar colunas
                cols_low = {c.lower(): c for c in df_meta.columns}
                # verdade:
                y_true = (
                    cols_low.get("y_true") or cols_low.get("true") or cols_low.get("label")
                    or cols_low.get("classe_verdade") or cols_low.get("real") or cols_low.get("target")
                )
                # previsto:
                y_pred = (
                    cols_low.get("y_pred") or cols_low.get("pred") or cols_low.get("predicted")
                    or cols_low.get("predicted_class") or cols_low.get("classe_prevista")
                )
                # caso típico do seu dataset:
                if y_pred is None and "Predicted_Class" in df_meta.columns:
                    y_pred = "Predicted_Class"
    
                # probas: prob_*, proba_* ou P(...)
                import re as re
                prob_cols = [c for c in df_meta.columns
                             if c.lower().startswith(("prob_", "proba_")) or re.match(r"^p\(.+\)$", c.lower())]
    
                # ---- Matriz de confusão (se tivermos y_true e y_pred)
                if y_true and y_pred:
                    try:
                        from sklearn.metrics import confusion_matrix
                        labels = sorted(pd.unique(pd.concat([
                            df_meta[y_true].astype(str),
                            df_meta[y_pred].astype(str)
                        ])))
                        cm = confusion_matrix(df_meta[y_true].astype(str),
                                              df_meta[y_pred].astype(str),
                                              labels=labels)
                        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
                        fig_cm = px.imshow(df_cm, text_auto=True, title="Matriz de confusão", aspect="auto")
                        st.plotly_chart(fig_cm, use_container_width=True)
                    except Exception as e:
                        st.info(f"Não foi possível calcular a matriz de confusão ({e}).")
    
                # ---- Histograma das probabilidades (se existirem)
                if prob_cols:
                    mlong = df_meta.melt(value_vars=prob_cols, var_name="classe", value_name="prob")
                    fig_hist = px.histogram(mlong, x="prob", facet_col="classe", nbins=30,
                                            title="Distribuição de probabilidades por classe")
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.caption("Não encontrei colunas de probabilidade (ex.: `prob_*`, `proba_*` ou `P(classe)`).")
    
                # ---- TABELA-RESUMO (sempre que tivermos y_true/y_pred)
                if y_true and y_pred:
                    # acurácia simples
                    acc = float((df_meta[y_true].astype(str) == df_meta[y_pred].astype(str)).mean()) if len(df_meta) else float("nan")
    
                    vc_true = df_meta[y_true].astype(str).value_counts(dropna=False)
                    vc_pred = df_meta[y_pred].astype(str).value_counts(dropna=False)
                    classes = sorted(set(vc_true.index) | set(vc_pred.index), key=str)
    
                    resumo = pd.DataFrame({
                        "classe": classes,
                        "n_true": [int(vc_true.get(c, 0)) for c in classes],
                        "n_pred": [int(vc_pred.get(c, 0)) for c in classes],
                    })
                    resumo.loc["TOTAL"] = ["TOTAL", int(resumo["n_true"].sum()), int(resumo["n_pred"].sum())]
    
                    st.markdown("#### Resumo — contagens e acurácia")
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.metric("Acurácia (global)", f"{acc:.2%}" if pd.notna(acc) else "—")
                    with c2:
                        st.dataframe(resumo, use_container_width=True)
                        download_df(resumo.reset_index(drop=True), "resumo_predictions_with_meta")
                else:
                    st.caption("Para a matriz de confusão e o resumo, preciso identificar as colunas de rótulo real (ex.: `y_true`, `label`) e previsto (ex.: `y_pred`, `Predicted_Class`).")

# ==========================
# SIDEBAR — Repositório e Mapbox
# ==========================
with st.sidebar:
    st.header("🔗 Fonte dos Dados (GitHub)")
    repo_input = st.text_input("owner/repo", value="emiliobneto/UrbanTechCluster")
    branch_input = st.text_input("branch (vazio = auto)", value="")
    try:
        repo = normalizerepo(repo_input)
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
TAB_LABELS = ["🗺️ Principal", "🧬 Clusterização", "📊 Univariadas", "🧠 ML → PCA",  "🤖 Clusterizador"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(TAB_LABELS)


# -----------------------------------------------------------------------------
# ABA 1 — Principal (mapa + dados por SQ + recortes) — AJUSTADA
# -----------------------------------------------------------------------------
with tab1:
    # ---------------- Helpers locais ----------------
    def _t1_download_df(df: pd.DataFrame, base_name: str):
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Baixar CSV", csv, file_name=f"{base_name}.csv", mime="text/csv", key=f"t1_dl_{base_name}")

    def _t1_png_mapa_300dpi(gdf_pol, value_col, cmap_dict, gdf_overlay=None, titulo=None, dpi=300):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
        # plota por classe/categoria
        for k, hexc in cmap_dict.items():
            sub = gdf_pol[gdf_pol[value_col] == k]
            if not sub.empty:
                try:
                    sub.plot(ax=ax, color=hexc, linewidth=0, edgecolor="none")
                except Exception:
                    sub = sub.buffer(0)
                    sub.plot(ax=ax, color=hexc, linewidth=0, edgecolor="none")
        if gdf_overlay is not None and not gdf_overlay.empty:
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
    loaded_layers, other_layers_paths = [], []
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

    # ---------------- Dados por SQ — seleção de fonte/arquivo/ano ----------------
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

    # ---------------- Definição da variável e do modo de coloração (3ª coluna) ----------------
    # Regra:
    # - Se a 3ª coluna for 'str':
    #     * Se tiver 1 único valor (ex.: "predrenda") e existir uma coluna numérica com esse nome => usar essa coluna numérica
    #     * Caso contrário => usar a própria 3ª coluna como classificação (categórica) e montar a legenda a partir dela
    # - Fallback: escolher manualmente uma variável numérica
    third_col = df_vars.columns[2] if len(df_vars.columns) >= 3 else None
    color_mode = "fallback_numeric"   # "classification_str" | "numeric_from_flag" | "fallback_numeric"
    var_label_for_legend = None       # Título/descrição na legenda

    # candidatos numéricos pro fallback
    id_like = {c for c in df_vars.columns if str(c).lower() in {"sq", "id", "codigo", "code"}}
    time_like = {c for c in df_vars.columns if str(c).lower() in {"ano", "year"}}
    ignore_cols = id_like | time_like
    num_cols_all = [c for c in df_vars.columns if pd.api.types.is_numeric_dtype(df_vars[c])]
    var_options_numeric = [c for c in num_cols_all if c not in ignore_cols] or [c for c in df_vars.columns if c not in ignore_cols]

    # tentativa automática
    if third_col is not None and (df_vars[third_col].dtype.kind in ("O", "U", "S")):
        uniq_vals = df_vars[third_col].dropna().unique().tolist()
        if len(uniq_vals) == 1 and str(uniq_vals[0]) in df_vars.columns and pd.api.types.is_numeric_dtype(df_vars[str(uniq_vals[0])]):
            # 3ª coluna contém o nome da variável numérica a usar
            var_sel = str(uniq_vals[0])
            color_mode = "numeric_from_flag"
            var_label_for_legend = var_sel
        else:
            # 3ª coluna é uma classificação por SQ (categórica)
            color_mode = "classification_str"
            var_sel = third_col  # será usada diretamente para pintar
            var_label_for_legend = f"{third_col} (classificação)"
    else:
        # fallback manual
        var_sel = st.selectbox("Variável numérica a mapear (fallback)", var_options_numeric, key="t1_varname_fallback")
        color_mode = "fallback_numeric"
        var_label_for_legend = var_sel

    # Para numeric modes: número de classes (Jenks → fallback quantis)
    n_classes = None
    if color_mode in ("numeric_from_flag", "fallback_numeric"):
        col3.write("")  # só para manter o grid
        n_classes = col3.slider("Quebras (Jenks)", min_value=4, max_value=8, value=6, key="t1_jenks")

    # ---------------- Merge com quadras ----------------
    gdf = gdf_quadras.merge(df_vars[[join_col, var_sel]], left_on=sq_col_quadras, right_on=join_col, how="left")

    # ---------------- Classificação e legenda ----------------
    legend_kind, legend_info, cmap = None, None, None
    if color_mode == "classification_str":
        # Usar diretamente os rótulos da 3ª coluna
        series = gdf[var_sel].astype("string")
        cats = [c for c in series.dropna().unique()]
        try:
            cats_sorted = sorted(cats, key=lambda x: str(x))
        except Exception:
            cats_sorted = cats
        palette = pick_categorical(len(cats_sorted))
        cmap = {cat: palette[i] for i, cat in enumerate(cats_sorted)}
        gdf["value"] = series
        legend_kind = "categorical"
        legend_info = cmap
    else:
        # numeric_from_flag ou fallback_numeric
        series = gdf[var_sel].astype(float)
        vals = series.dropna().values
        uniq = np.unique(vals)
        k = max(4, min(8, n_classes if n_classes else 6))
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
            labels = list(range(k))
            binned, bins = pd.qcut(series, q=k, labels=labels, retbins=True, duplicates="drop")
            binned = pd.Series(binned, index=series.index).astype("float").astype("Int64")
            gdf["value"] = binned
            palette = pick_sequential(len(np.unique(binned.dropna())))
            cmap = {i: palette[i] for i in range(len(palette))}
            legend_kind = "numeric"
            legend_info = (bins, palette)

    # ---------------- GeoJSON + camadas de mapa ----------------
    geojson = make_geojson(gdf)
    for feat in geojson.get("features", []):
        val = feat.get("properties", {}).get("value", None)
        if legend_kind == "numeric":
            hexc = cmap.get(val, "#999999")
        else:
            hexc = cmap.get(val, "#999999")
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
    map_col, legend_col = st.columns([5, 1], gap="large")  # mais espaço pro mapa
    with map_col:
        if basemap.startswith("Satélite"):
            deck(layers, satellite=True)
        else:
            osm_basemap_deck(layers)

        # Export PNG 300 DPI (mapa geral)
        if st.button("🖼️ Gerar PNG 300 DPI (mapa geral)", key="t1_btn_png_geral"):
            try:
                titulo = f"{var_label_for_legend}" + (f" — {year}" if years and year is not None else "")
                png_bytes = _t1_png_mapa_300dpi(
                    gdf[["value", "geometry"]].dropna(subset=["value"]),
                    "value",
                    cmap,
                    gdf_overlay=None,
                    titulo=titulo,
                    dpi=300,
                )
                st.download_button(
                    "Baixar PNG 300 DPI (mapa geral)",
                    png_bytes,
                    file_name=f"mapa_geral_{var_label_for_legend}{'_'+str(year) if years and year is not None else ''}.png",
                    mime="image/png",
                    key="t1_dl_png_geral",
                )
            except Exception as e:
                st.caption(f"Export PNG indisponível ({e})")

        # Export tabela (SQ + value)
        df_export = gdf[[sq_col_quadras, var_sel]].rename(columns={sq_col_quadras: "SQ", var_sel: var_label_for_legend})
        _t1_download_df(df_export, f"dados_mapa_{var_label_for_legend}{'_'+str(year) if years and year is not None else ''}")

    with legend_col:
        st.markdown(f"**Legenda — {var_label_for_legend}**")
        if legend_kind == "categorical":
            for k in sorted(cmap.keys(), key=lambda x: str(x)):
                _legend_row(cmap[k], str(k))
        elif legend_kind == "numeric":
            bins, palette = legend_info
            # mesma função de legenda numérica, mas com título já acima
            k = len(palette)
            for i in range(k):
                left = bins[i]
                right = bins[i + 1] if i + 1 < len(bins) else float("inf")
                def _fmt_num(x):
                    try:
                        if x == -float("inf"): return "-∞"
                        if x == float("inf"): return "+∞"
                        return f"{float(x):.3g}"
                    except Exception:
                        return str(x)
                if left == -float("inf"):
                    label = f"≤ {_fmt_num(right)}"
                elif right == float("inf"):
                    label = f"> {_fmt_num(left)}"
                else:
                    label = f"({_fmt_num(left)} – {_fmt_num(right)}]"
                _legend_row(palette[i], label)

    # ---------------- Recortes: mapa temático preenchido + métricas da área ----------------
    st.subheader("Recortes espaciais (GPKG)")
    st.caption("Selecione um GPKG em `Data/mapa/recortes` para filtrar os SQs e ver **mapa temático** e **métricas** apenas dessa área.")
    rec_dir = None
    recorte_file_path = None
    try:
        rec_dir = pick_existing_dir(repo, branch, ["Data/mapa/recortes", "Data/Mapa/recortes", "data/mapa/recortes"])
        recorte_files = list_files(repo, rec_dir, branch, (".gpkg",))
        if not recorte_files:
            st.info("Nenhum GPKG de recorte encontrado.")
        else:
            colR0, colR1 = st.columns([4, 2], gap="large")

            with colR0:
                rec_sel_name = st.selectbox("Arquivo de recorte (.gpkg)", [f["name"] for f in recorte_files], index=0, key="t1rec_file")
                rec_obj = next(x for x in recorte_files if x["name"] == rec_sel_name)
                recorte_file_path = rec_obj["path"]
                gdfrec = load_gpkg(repo, recorte_file_path, branch)

                # Interseção dos SQs com o recorte
                try:
                    import geopandas as gpd
                    gq = ensure_wgs84(gdf_quadras[[sq_col_quadras, "geometry"]].copy())
                    gr = ensure_wgs84(gdfrec[["geometry"]].copy())
                    try:
                        sq_sel = gpd.sjoin(gq, gr, predicate="intersects", how="inner")[sq_col_quadras].unique().tolist()
                    except Exception:
                        bbox = gr.total_bounds
                        sq_sel = gq.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]][sq_col_quadras].unique().tolist()
                except Exception as e:
                    st.error(f"Falha ao cruzar recorte com SQs: {e}")
                    sq_sel = []

                # Subconjunto colorido para o recorte usando a MESMA variável/legenda do mapa geral
                gdf_colorrec = gdf[gdf[sq_col_quadras].isin(sq_sel)].copy()
                gjrec_fill = make_geojson(gdf_colorrec)
                for feat in gjrec_fill.get("features", []):
                    val = feat.get("properties", {}).get("value", None)
                    hexc = cmap.get(val, "#999999")
                    feat.setdefault("properties", {})["fill_color"] = hex_to_rgba(hexc)

                # Camadas: preenchido + contorno do recorte
                layersrec = [render_geojson_layer(gjrec_fill, name="recorte_fill"),
                              render_line_layer(make_geojson(gdfrec), name="recorte_borda")]

                st.markdown("#### Mapa — Recorte selecionado (preenchido pela variável escolhida)")
                if basemap.startswith("Satélite"):
                    deck(layersrec, satellite=True)
                else:
                    osm_basemap_deck(layersrec)

                # Export PNG 300 DPI do mapa do recorte
                if st.button("🖼️ Gerar PNG 300 DPI (mapa do recorte)", key="t1_btn_pngrec"):
                    try:
                        titulo = f"{var_label_for_legend}" + (f" — {year}" if years and year is not None else "")
                        png_bytes = _t1_png_mapa_300dpi(
                            gdf_colorrec[["value", "geometry"]].dropna(subset=["value"]),
                            "value",
                            cmap,
                            gdf_overlay=gdfrec,
                            titulo=titulo,
                            dpi=300,
                        )
                        base_nome = f"recorte_{os.path.splitext(rec_sel_name)[0]}"
                        st.download_button("Baixar PNG 300 DPI (mapa do recorte)", png_bytes, file_name=f"{base_nome}_{var_label_for_legend}_300dpi.png", mime="image/png", key="t1_dl_pngrec")
                    except Exception as e:
                        st.caption(f"Export PNG indisponível ({e})")

            with colR1:
                st.metric("SQs no recorte", len(sq_sel))
                st.markdown(f"**Legenda — {var_label_for_legend} (recorte)**")
                if legend_kind == "categorical":
                    # mostra apenas categorias presentes no recorte
                    present = gdf_colorrec["value"].dropna().astype(str).unique().tolist()
                    for k in sorted(present, key=lambda x: str(x)):
                        _legend_row(cmap[k], str(k))
                else:
                    # numérica — mantém os mesmos bins da visão geral
                    bins, palette = legend_info
                    k = len(palette)
                    for i in range(k):
                        left = bins[i]
                        right = bins[i + 1] if i + 1 < len(bins) else float("inf")
                        def _fmt_num(x):
                            try:
                                if x == -float("inf"): return "-∞"
                                if x == float("inf"): return "+∞"
                                return f"{float(x):.3g}"
                            except Exception:
                                return str(x)
                        if left == -float("inf"):
                            label = f"≤ {_fmt_num(right)}"
                        elif right == float("inf"):
                            label = f"> {_fmt_num(left)}"
                        else:
                            label = f"({_fmt_num(left)} – {_fmt_num(right)}]"
                        _legend_row(palette[i], label)

                # Dados da área recortada (por SQ e resumo)
                st.markdown("#### Métricas para a área recortada")
                df_varsrec = df_vars[df_vars[join_col].isin(sq_sel)].copy()
                # variáveis numéricas válidas no recorte
                id_like_r = {c for c in df_varsrec.columns if str(c).lower() in {"sq", "id", "codigo", "code"}}
                time_like_r = {c for c in df_varsrec.columns if str(c).lower() in {"ano", "year"}}
                ignore_cols_r = id_like_r | time_like_r
                num_colsrec = [c for c in df_varsrec.columns if pd.api.types.is_numeric_dtype(df_varsrec[c])]
                var_optsrec = [c for c in num_colsrec if c not in ignore_cols_r] or [c for c in df_varsrec.columns if c not in ignore_cols_r]

                modo = st.radio("Exibição", ["Por SQ", "Resumo (estatísticas)"], horizontal=True, index=0, key="t1_modorec")
                if modo == "Por SQ":
                    # mostra prioritariamente a variável usada no mapa (se for numérica)
                    cols_show = [join_col]
                    if color_mode in ("numeric_from_flag", "fallback_numeric") and var_label_for_legend in df_varsrec.columns:
                        cols_show.append(var_label_for_legend)
                    st.dataframe(df_varsrec[cols_show].sort_values(join_col), use_container_width=True)
                    _t1_download_df(df_varsrec[cols_show].sort_values(join_col), f"recorte_porSQ_{os.path.splitext(rec_sel_name)[0]}")
                else:
                    # resumo das numéricas selecionadas
                    vars_escolhidas = st.multiselect(
                        "Variáveis (métricas) a resumir",
                        var_optsrec,
                        default=[var_label_for_legend] if var_label_for_legend in var_optsrec else var_optsrec[: min(5, len(var_optsrec))],
                        key="t1_varsrec"
                    )
                    if vars_escolhidas:
                        desc = df_varsrec[vars_escolhidas].describe().T
                        st.dataframe(desc, use_container_width=True)
                        _t1_download_df(desc.reset_index().rename(columns={"index": "variavel"}), f"recorteresumo_{os.path.splitext(rec_sel_name)[0]}")

    except Exception as e:
        st.warning(f"Não foi possível listar/ler recortes: {e}")

    # ---------------- Debug — caminhos/variáveis usados ----------------
    with st.expander("🔎 Debug — caminhos/variáveis usados (Aba 1)"):
        debug_info = {
            "repo@branch": f"{repo}@{branch}",
            "quadras_path_usado": quadras_path_used,
            "mapa_dir": mapa_dir if 'mapa_dir' in locals() else None,
            "camadas_auxiliares_sel": other_layers_paths,
            "dados_base_dir": base_dir,
            "arquivo_dados_selecionado": data_file_path,
            "coluna_SQ_quadras": sq_col_quadras,
            "coluna_SQ_dados": join_col,
            "coluna_ano": years_col if 'years_col' in locals() else None,
            "ano_selecionado": year if 'year' in locals() else None,
            "third_col": third_col,
            "color_mode": color_mode,
            "var_usada_para_pintar/legenda": var_label_for_legend,
            "recortes_dir": rec_dir,
            "arquivorecorte_sel": recorte_file_path,
            "legend_kind": legend_kind,
        }
        st.code(json.dumps(debug_info, ensure_ascii=False, indent=2), language="json")

# -----------------------------------------------------------------------------
# ABA 2 — Clusterização (mapas, métricas por cluster e testes) — V2 (COMPLETA)
# -----------------------------------------------------------------------------
with tab2:
    import io
    import re
    import json
    import numpy as np
    import pandas as pd
    import streamlit as st

    # Tenta garantir pydeck/geopandas se o mapa for usado
    try:
        import pydeck as pdk
    except Exception:
        pdk = None
    try:
        import geopandas as gpd
    except Exception:
        gpd = None

    st.subheader("🧬 Clusterização — Mapas, Métricas e Testes")

    # =========================================================================
    # PRÉ-CHECAGENS / CONTEXTO
    # =========================================================================
    # repo/branch podem vir do escopo global ou do session_state
    repo = locals().get("repo", st.session_state.get("repo"))
    branch = locals().get("branch", st.session_state.get("branch"))

    if not repo or not branch:
        st.error("Defina `repo` e `branch` antes de abrir a Aba 2 (ex.: em controles na Aba 1).")
        st.stop()

    # Exigiremos alguns helpers que já devem existir no app principal
    required_helpers = [
        "pick_existing_dir", "list_files", "load_parquet", "load_csv",
        "load_gpkg", "ensure_wgs84"
    ]
    missing = [h for h in required_helpers if h not in globals()]
    if missing:
        st.error(
            "Estes helpers precisam existir no app principal e não foram encontrados: "
            + ", ".join(missing)
        )
        st.stop()

    # =========================================================================
    # HELPERS LOCAIS (apenas o que a Aba 2 usa diretamente)
    # =========================================================================
    @st.cache_data(show_spinner=False)
    def _norm_sq_series(s: pd.Series, digits: int = 6) -> pd.Series:
        s = s.astype("string").str.replace(r"\D", "", regex=True).fillna("")
        s = s.str[-digits:].str.zfill(digits)
        return s.mask(s.eq(""))

    @st.cache_data(show_spinner=False)
    def _to_int_code(x):
        try:
            v = float(str(x).strip())
            if np.isfinite(v) and abs(v - int(v)) < 1e-9:
                return int(v)
        except Exception:
            pass
        m = re.search(r"\d+", str(x))
        return int(m.group(0)) if m else None

    def _safe_int(x, allowed: set[int] | None = None) -> int | None:
        try:
            xi = int(float(str(x).strip()))
            if allowed and xi not in allowed:
                return None
            return xi
        except Exception:
            return None

    def download_df(df: pd.DataFrame, base_name: str):
        csv = df.to_csv(index=False).encode("utf-8")
        safe_key = re.sub(r"[^a-z0-9_]+", "_", str(base_name).lower())
        st.download_button(
            "📥 Baixar CSV",
            csv,
            file_name=f"{base_name}.csv",
            mime="text/csv",
            key=f"dl_{safe_key}",
        )

    # Paletas e utilitários simples de mapa (auto-contidos)
    def hex_to_rgba(hex_color, alpha: int = 180):
        try:
            if not isinstance(hex_color, str):
                return [153, 153, 153, alpha]
            h = hex_color.strip().lstrip("#")
            if len(h) == 3:
                h = "".join(ch * 2 for ch in h)
            if len(h) != 6:
                return [153, 153, 153, alpha]
            r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
            return [r, g, b, alpha]
        except Exception:
            return [153, 153, 153, alpha]

    CATEGORICAL = [
        "#7c3aed", "#d946ef", "#fb7185", "#f97316", "#f59e0b", "#facc15",
        "#fde047", "#a16207", "#9a3412", "#b91c1c", "#ea580c", "#be185d",
        "#9333ea", "#6b21a8", "#a21caf", "#c026d3", "#db2777", "#e11d48",
        "#eab308", "#f43f5e",
    ]
    def pick_categorical(k: int):
        if k <= len(CATEGORICAL):
            return CATEGORICAL[:k]
        reps = (k // len(CATEGORICAL)) + 1
        return (CATEGORICAL * reps)[:k]

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

    def _make_geojson_from_gdf(gdf_in: "gpd.GeoDataFrame"):
        """Converte GeoDataFrame WGS84 para GeoJSON (dict)."""
        gdf_w = ensure_wgs84(gdf_in)
        return json.loads(gdf_w.to_json())

    def _draw_geojson_layers(layers, satellite=False):
        if pdk is None:
            st.info("pydeck não está disponível; não foi possível renderizar o mapa.")
            return
        view = pdk.ViewState(latitude=-23.55, longitude=-46.63, zoom=10)
        map_style = None if not satellite else "mapbox://styles/mapbox/satellite-streets-v12"
        if satellite and not st.secrets.get("mapbox", {}).get("token"):
            st.info("Para o estilo Satélite, defina `st.secrets['mapbox']['token']`.")
        deck_obj = pdk.Deck(layers=layers, initial_view_state=view, map_style=map_style)
        st.pydeck_chart(deck_obj, use_container_width=True)

    # rótulos de clusters
    label_map = {
        0: "0 – Ausência de clusterização",
        1: "1 – Cluster em estágio inicial",
        2: "2 – Cluster em formação",
        3: "3 – Clusterizado",
    }
    # remover variáveis com estes padrões
    banre = re.compile(r"(cluster|est[aá]gio|classe|label|pred)", re.I)

    # =========================================================================
    # CONTROLES DE PERFORMANCE
    # =========================================================================
    perf_col1, perf_col2, perf_col3 = st.columns([1.2, 1, 1])
    with perf_col1:
        fast_map = st.toggle(
            "🧪 Mapa leve (centróides/amostragem)",
            value=True, key="t2_fastmap",
            help="Usa centróides em vez de polígonos e pode amostrar feições para acelerar."
        )
    with perf_col2:
        max_feat = st.slider("Máx. feições no mapa", 5_000, 80_000, 20_000, step=5_000, key="t2_maxfeat")
    with perf_col3:
        max_vars = st.slider("Máx. variáveis por rodada (métricas)", 3, 60, 12, step=1, key="t2_maxvars")

    preload_toggle = st.toggle(
        "⚡ Pré-carregar métricas por cluster×ano (cache)",
        value=False,  # padrão DESLIGADO
        key="t2_preload_cache",
        help="Carrega e guarda em cache todas as métricas por cluster×ano logo no início."
    )

    # =========================================================================
    # CARGA DE CLUSTERS (upload ou GitHub) + normalização
    # =========================================================================
    colTopL, colTopR = st.columns([2, 1])
    with colTopL:
        up = st.file_uploader("EstagioClusterizacao (opcional)", type=["csv", "parquet"], key="t2_upl")
    with colTopR:
        simplify_tol = st.slider(
            "Simplificação (°)", 0.0, 0.0008, 0.0002, 0.0001, key="t2_simplify",
            help="Valores maiores = menos vértices = mais rápido (apenas quando desenhando polígonos)."
        )

    # Helper para carregar clusters do repositório (usa seus helpers globais)
    def _load_clusters(ownerrepo: str, branch: str):
        clusters_dir = pick_existing_dir(
            ownerrepo, branch, ["Data/dados/Originais", "Data/dados/originais", "data/dados/originais"]
        )
        all_in_dir = list_files(ownerrepo, clusters_dir, branch, (".csv", ".parquet"))
        cand = [f for f in all_in_dir if re.fullmatch(r"(?i)EstagioClusterizacao\.(csv|parquet)", str(f["name"]))]
        if not cand:
            cand = [f for f in all_in_dir if re.search(r"(?i)est[aá]gio", str(f["name"])) and re.search(r"(?i)cluster", str(f["name"]))]
        if not cand:
            return None, "Não encontrei `EstagioClusterizacao.{csv|parquet}` em Data/dados/Originais."
        est_file = cand[0]
        df_est = load_parquet(ownerrepo, est_file["path"], branch) if str(est_file["name"]).lower().endswith(".parquet") else load_csv(ownerrepo, est_file["path"], branch)
        source_label = f"{clusters_dir}/{est_file['name']}"
        return df_est, source_label

    # Ler clusters (upload tem prioridade)
    df_est_raw, source_label = (None, "")
    if up is not None:
        try:
            df_est_raw = pd.read_parquet(up) if up.name.lower().endswith(".parquet") else pd.read_csv(up)
            source_label = f"(upload) {up.name}"
        except Exception as e:
            st.error(f"Falha ao ler upload: {e}")
    else:
        df_est_raw, source_label = _load_clusters(repo, branch)

    if not isinstance(df_est_raw, pd.DataFrame) or df_est_raw.empty:
        st.error(f"Clusters indisponíveis. {source_label or ''}")
        st.stop()

    # Ano + coluna de cluster
    ano_col_est = next((c for c in df_est_raw.columns if str(c).lower() in ("ano", "year")), None)
    anos_ok = None
    if ano_col_est:
        anos_vals = pd.to_numeric(df_est_raw[ano_col_est], errors="coerce")
        anos_ok = sorted(anos_vals.dropna().astype(int).unique().tolist()) or None
    year_sel = st.select_slider("Ano (clusters)", options=anos_ok or [None], value=(anos_ok[-1] if anos_ok else None), key="t2_year_sel")

    cluster_cols = [c for c in df_est_raw.columns if re.search(r"(?i)(cluster|est[aá]gio|label)", str(c))]
    if not cluster_cols:
        st.error("Não encontrei coluna de cluster (ex.: EstagioClusterizacao, Cluster, Label).")
        st.stop()
    preferred = next((c for c in cluster_cols if str(c).lower() == "estagioclusterizacao"), cluster_cols[0])
    cluster_col = st.selectbox("Coluna de cluster", cluster_cols, index=cluster_cols.index(preferred), key="t2_cluster_col")

    # Normaliza/filtra clusters p/ JOIN
    df_est_clean = df_est_raw.copy()
    sq_est_col = next((c for c in df_est_clean.columns if str(c).upper() == "SQ"), None)
    if sq_est_col is None:
        st.error("Arquivo de clusters precisa ter coluna 'SQ'.")
        st.stop()
    df_est_clean["_SQ_norm"] = _norm_sq_series(df_est_clean[sq_est_col])
    if (ano_col_est is not None) and (year_sel is not None):
        df_est_clean = df_est_clean[pd.to_numeric(df_est_clean[ano_col_est], errors="coerce").astype("Int64") == year_sel].copy()
    df_est_clean["_cl_code"] = df_est_clean[cluster_col].map(_to_int_code).astype("Int64")
    df_est_clean = (
        df_est_clean.sort_values(["_SQ_norm", "_cl_code"])
                    .drop_duplicates("_SQ_norm", keep="last")
                    .dropna(subset=["_SQ_norm", "_cl_code"])
                    [["_SQ_norm", "_cl_code"]]
    )

    # =========================================================================
    # ARQUIVO DE VALORES (por SQ) + filtro de ano coerente
    # =========================================================================
    ver_val = st.radio("Versão dos dados (valores por SQ)", ["originais", "winsorizados"], horizontal=True, key="t2_vals_ver")
    base_vals = pick_existing_dir(
        repo, branch,
        [f"Data/dados/{'originais' if ver_val=='originais' else 'winsorizados'}",
         f"Data/dados/{'Originais' if ver_val=='originais' else 'Winsorizados'}",
         f"Data/dados/{'winsorize' if ver_val!='originais' else 'originais'}"]
    )
    vals_all = list_files(repo, base_vals, branch, (".parquet", ".csv"))
    incl_pred = st.checkbox("Incluir arquivos pred_*", value=False, key="t2_vals_incl_pred")
    vals_files = [
        f for f in vals_all
        if (incl_pred or not str(f["name"]).lower().startswith("pred_"))
        and not re.search(r"(?i)est[aá]gio.*cluster", str(f["name"]))
    ]
    if not vals_files:
        st.info(f"Nenhum arquivo elegível em `{base_vals}` (excluí EstagioClusterizacao.* e, opcionalmente, pred_*).")
        st.stop()

    sel_vals = st.selectbox("Arquivo de valores (por SQ)", [f["name"] for f in vals_files], index=0, key="t2_vals_file")
    vals_obj = next(x for x in vals_files if x["name"] == sel_vals)
    df_vals_raw = load_parquet(repo, vals_obj["path"], branch) if str(vals_obj["name"]).endswith(".parquet") else load_csv(repo, vals_obj["path"], branch)

    sq_col_vals = next((c for c in df_vals_raw.columns if str(c).upper() == "SQ"), None)
    if sq_col_vals is None:
        st.error("O arquivo de valores precisa ter a coluna 'SQ'.")
        st.stop()
    ano_col_vals = next((c for c in df_vals_raw.columns if str(c).lower() in ("ano", "year")), None)
    if ano_col_vals and year_sel is not None:
        df_vals_raw = df_vals_raw[pd.to_numeric(df_vals_raw[ano_col_vals], errors="coerce").astype("Int64") == year_sel].copy()

    # =========================================================================
    # CACHE/PRELOAD: métricas por cluster × ano
    # =========================================================================
    @st.cache_data(show_spinner=True, ttl=3600, max_entries=6)
    def _preload_cluster_metrics_by_year(
        df_vals_raw: pd.DataFrame,
        df_est_raw: pd.DataFrame,
        cluster_col: str,
        *,
        chunk_size: int = 60,
    ) -> pd.DataFrame:
        # --- detectar colunas chave ---
        sq_vals = next((c for c in df_vals_raw.columns if str(c).upper()=="SQ"), None)
        ano_vals = next((c for c in df_vals_raw.columns if str(c).lower() in ("ano","year")), None)
        if sq_vals is None:
            raise RuntimeError("Arquivo de valores não possui coluna 'SQ'.")
        sq_est  = next((c for c in df_est_raw.columns  if str(c).upper()=="SQ"), None)
        ano_est = next((c for c in df_est_raw.columns  if str(c).lower() in ("ano","year")), None)
        if sq_est is None:
            raise RuntimeError("Arquivo de clusters não possui coluna 'SQ'.")

        vals = df_vals_raw.copy()
        vals["_SQ_norm"] = _norm_sq_series(vals[sq_vals])
        est  = df_est_raw.copy()
        est["_SQ_norm"] = _norm_sq_series(est[sq_est])

        # cluster como código 0..3 quando possível
        est["_cl_code"] = est[cluster_col].map(_to_int_code).astype("Int64")

        # interseção de anos (ou None)
        if ano_vals and ano_est:
            anos = sorted(
                set(pd.to_numeric(vals[ano_vals], errors="coerce").dropna().astype(int))
                & set(pd.to_numeric(est[ano_est],  errors="coerce").dropna().astype(int))
            )
        elif ano_vals:
            anos = sorted(pd.to_numeric(vals[ano_vals], errors="coerce").dropna().astype(int).unique().tolist())
        elif ano_est:
            anos = sorted(pd.to_numeric(est[ano_est], errors="coerce").dropna().astype(int).unique().tolist())
        else:
            anos = [None]  # sem ano

        # selecionar variáveis numéricas válidas
        id_like   = {c for c in vals.columns if str(c).lower() in {"sq","id","codigo","code","_sq_norm"}}
        time_like = {c for c in vals.columns if str(c).lower() in {"ano","year"}}
        num_cols  = [c for c in vals.columns if pd.api.types.is_numeric_dtype(vals[c])]
        var_all   = [c for c in num_cols if c not in (id_like | time_like)]

        out_frames = []

        for ano in anos:
            # filtra ano em ambos os lados (quando existir)
            v = vals if (ano is None or not ano_vals) else vals[pd.to_numeric(vals[ano_vals], errors="coerce").astype("Int64")==ano]
            e = est  if (ano is None or not ano_est) else est [pd.to_numeric(est [ano_est ], errors="coerce").astype("Int64")==ano]
            # pega a melhor linha por SQ (caso tenha repetição por merges)
            e = e.sort_values(["_SQ_norm","_cl_code"]).drop_duplicates("_SQ_norm", keep="last")

            # mapa SQ->cluster
            mapper = e.set_index("_SQ_norm")["_cl_code"]

            # processa em blocos de variáveis para economizar memória
            for i in range(0, len(var_all), chunk_size):
                chunk = var_all[i:i+chunk_size]
                d = v[["_SQ_norm"] + chunk].copy()
                d[chunk] = d[chunk].apply(pd.to_numeric, errors="coerce")
                d["_cl_code"] = d["_SQ_norm"].map(mapper).astype("Int64")
                d = d[d["_cl_code"].isin([0,1,2,3])]  # clusters conhecidos

                if d.empty:
                    continue

                g = d.groupby("_cl_code", observed=True)

                n_df     = g[chunk].count()
                miss_df  = pd.DataFrame(g.size().values[:,None] - n_df.values, index=n_df.index, columns=chunk)
                mean_df  = g[chunk].mean()
                med_df   = g[chunk].median()
                std_df   = g[chunk].std(ddof=1)
                min_df   = g[chunk].min()
                max_df   = g[chunk].max()
                q_df     = g[chunk].quantile([0.25, 0.75])
                p25_df   = q_df.xs(0.25, level=1)
                p75_df   = q_df.xs(0.75, level=1)
                cv_df    = std_df/mean_df

                def _to_long(name, df_):
                    return (
                        df_.stack()
                           .rename(name)
                           .reset_index()
                           .rename(columns={"_cl_code":"cluster","level_1":"variavel"})
                    )

                parts = [
                    _to_long("n",             n_df),
                    _to_long("missings",      miss_df),
                    _to_long("media",         mean_df),
                    _to_long("mediana",       med_df),
                    _to_long("desvio_padrao", std_df),
                    _to_long("p25",           p25_df),
                    _to_long("p75",           p75_df),
                    _to_long("minimo",        min_df),
                    _to_long("maximo",        max_df),
                    _to_long("coef_var",      cv_df),
                ]
                merged = parts[0]
                for p in parts[1:]:
                    merged = merged.merge(p, on=["cluster","variavel"], how="left")
                merged.insert(0, "ano", ano)
                out_frames.append(merged)

        if not out_frames:
            return pd.DataFrame(columns=["ano","cluster","variavel","n","missings","media","mediana","desvio_padrao","p25","p75","minimo","maximo","coef_var"])

        out = pd.concat(out_frames, ignore_index=True)
        out["cluster_label"] = out["cluster"].map(label_map)
        return out

    # Para o preload: reduz clusters ao ano selecionado
    df_est_for_pre = df_est_raw
    if ano_col_est and (year_sel is not None):
        df_est_for_pre = df_est_raw[pd.to_numeric(df_est_raw[ano_col_est], errors="coerce").astype("Int64") == year_sel].copy()

    metrics_key = f"t2_metrics_{repo}@{branch}|{vals_obj['path']}|{source_label}|{cluster_col}|{year_sel}"
    df_metrics_all = st.session_state.get(metrics_key)
    if preload_toggle and (df_metrics_all is None or df_metrics_all.empty):
        try:
            with st.spinner("Pré-carregando métricas por cluster×ano..."):
                df_metrics_all = _preload_cluster_metrics_by_year(
                    df_vals_raw, df_est_for_pre, cluster_col, chunk_size=max_vars
                )
            st.session_state[metrics_key] = df_metrics_all
        except MemoryError:
            st.warning("Pré-cálculo ficou pesado; desative o pré-carregamento ou reduza o número de variáveis.")
            df_metrics_all = pd.DataFrame()

    # =========================================================================
    # MAPA DE CLUSTERIZAÇÃO
    # =========================================================================
    st.markdown("### 🗺️ Mapa de clusterização")
    base_map_t2 = st.radio("Plano de fundo", ["OpenStreetMap", "Satélite (Mapbox)"], index=0, horizontal=True, key="t2_base")

    # Carrega quadras (mínimo) do repo (usa seu helper global load_gpkg/ensure_wgs84)
    @st.cache_data(show_spinner=True)
    def _load_quadras_min(ownerrepo, branch):
        gdf = st.session_state.get("gdf_quadras_cached")
        if gdf is None or gdf.empty:
            gdf = load_gpkg(ownerrepo, "Data/mapa/quadras.gpkg", branch)
            st.session_state["gdf_quadras_cached"] = gdf
        sq_col = next((c for c in gdf.columns if str(c).upper() == "SQ"), None)
        if not sq_col:
            raise RuntimeError("Camada de quadras não possui coluna 'SQ'.")
        gmin = gdf[[sq_col, gdf.geometry.name]].copy()
        gmin["_SQ_norm"] = _norm_sq_series(gmin[sq_col])
        try:
            gmin = ensure_wgs84(gmin)
            gmin["_centroid"] = gmin.geometry.centroid
        except Exception:
            gmin["_centroid"] = gmin.geometry
        return gmin, sq_col

    try:
        gdfq_min, sq_col_quadras = _load_quadras_min(repo, branch)
        gdf_map = gdfq_min.merge(df_est_clean, on="_SQ_norm", how="inner").copy()
        if gdf_map.empty:
            st.info("Não há feições para mapear após o JOIN de quadras × clusters.")
        else:
            # amostragem para acelerar
            if len(gdf_map) > max_feat:
                gdf_map = gdf_map.sample(n=max_feat, random_state=42)

            # simplificação opcional (só polígonos)
            if (not fast_map) and simplify_tol and simplify_tol > 0 and gpd is not None:
                try:
                    gdf_map = gdf_map.copy()
                    gdf_map[gdf_map.geometry.name] = gdf_map.geometry.simplify(simplify_tol, preserve_topology=True)
                except Exception:
                    pass

            palette = pick_categorical(4)

            # Cria GeoJSON com cor por feature (usa centróides quando fast_map)
            geom_col = "_centroid" if fast_map else gdf_map.geometry.name
            gg = gpd.GeoDataFrame(
                gdf_map[[geom_col, "_cl_code"]].rename(columns={geom_col: "geometry"}),
                geometry="geometry",
                crs=getattr(gdf_map, "crs", 4326),
            ) if gpd is not None else None

            if gpd is None:
                st.info("geopandas não está disponível; mapa desativado.")
            else:
                gj = _make_geojson_from_gdf(gg)
                for feat in gj.get("features", []):
                    cl_raw = feat.get("properties", {}).get("_cl_code", None)
                    cl = _safe_int(cl_raw, {0, 1, 2, 3})
                    hexc = palette[cl] if cl is not None else "#999999"
                    feat.setdefault("properties", {})
                    feat["properties"]["fill_color"] = hex_to_rgba(hexc, 180 if fast_map else 150)
                    feat["properties"]["name"] = f"Cluster {cl}" if cl is not None else "Cluster indef."
                    feat["properties"]["value"] = label_map.get(cl, str(cl_raw))

                layer = pdk.Layer(
                    "GeoJsonLayer",
                    gj,
                    pickable=True,
                    stroked=(not fast_map),
                    filled=True,
                    extruded=False,
                    get_fill_color="properties.fill_color",
                    get_line_color=[80, 80, 80],
                    get_line_width=0.5,
                    auto_highlight=True,
                ) if pdk is not None else None

                if layer is not None:
                    _draw_geojson_layers([layer], satellite=base_map_t2.startswith("Satélite"))

                st.markdown("**Legenda — clusters**")
                for c in [0, 1, 2, 3]:
                    _legend_row(palette[c], label_map[c])

    except Exception as e:
        st.info(f"Mapa indisponível: {e}")

    st.divider()

    # =========================================================================
    # 2.1) Métricas por cluster — univariadas
    # =========================================================================
    st.subheader("📊 Métricas por cluster — univariadas")

    dfm = st.session_state.get(metrics_key)
    if (dfm is None or dfm.empty):
        with st.spinner("Calculando métricas para o ano selecionado..."):
            try:
                dfm = _preload_cluster_metrics_by_year(df_vals_raw, df_est_for_pre, cluster_col, chunk_size=max_vars)
            except MemoryError:
                dfm = pd.DataFrame()
        st.session_state[metrics_key] = dfm

    if dfm is None or dfm.empty:
        st.warning("Não foi possível calcular as métricas. Verifique se os dois arquivos possuem 'SQ' e variáveis numéricas.")
    else:
        anos_list = sorted([a for a in dfm["ano"].dropna().unique().tolist() if a is not None]) or [None]
        ano_uni = st.select_slider("Ano", options=anos_list, value=(anos_list[-1] if anos_list != [None] else None), key="t2_uni_ano")
        dfm_use = dfm[dfm["ano"] == ano_uni] if ano_uni is not None else dfm.copy()

        estat = st.radio("Estatística", ["Média", "Mediana"], horizontal=True, key="t2_uni_stat")
        stat_col = "media" if estat == "Média" else "mediana"

        var_opts = sorted(dfm_use["variavel"].dropna().astype(str).unique().tolist())
        var_opts = [v for v in var_opts if not banre.search(str(v))]
        if not var_opts:
            st.info("Nenhuma variável numérica elegível encontrada (após remover colunas de cluster/estágio/classe/pred).")
        else:
            vars_sel = st.multiselect("Variáveis", var_opts, default=var_opts[: min(10, len(var_opts))], key="t2_uni_vars")
            if vars_sel:
                sub = dfm_use[dfm_use["variavel"].isin(vars_sel)].copy()
                piv = (
                    sub.pivot_table(index="cluster_label", columns="variavel", values=stat_col, aggfunc="first")
                       .reindex([
                           "0 – Ausência de clusterização",
                           "1 – Cluster em estágio inicial",
                           "2 – Cluster em formação",
                           "3 – Clusterizado",
                       ])
                )
                st.markdown("**Tabela — Valor por cluster (0–3)**")
                st.dataframe(piv, use_container_width=True)
                download_df(
                    piv.reset_index().rename(columns={"index": "Cluster"}),
                    f"univariadas_{stat_col}{'_'+str(ano_uni) if ano_uni is not None else ''}_por_cluster"
                )
            else:
                st.info("Selecione ao menos uma variável.")

    st.divider()

    # =========================================================================
    # 2.2) Métricas avançadas — cluster (0–3)
    # =========================================================================
    st.subheader("🧪 Métricas avançadas — cluster (0–3)")

    use_shapiro = st.checkbox("Calcular Shapiro por cluster (mais lento)", value=False, key="t2_adv_shapiro")
    shapiro_cap = st.slider("Máx. amostras por Shapiro (cap)", 500, 10000, 5000, 500, key="t2_adv_shapiro_cap")

    # lista de variáveis numéricas elegíveis diretamente do df_vals_raw
    id_like = {c for c in df_vals_raw.columns if str(c).lower() in {"sq", "id", "codigo", "code"}}
    time_like = {c for c in df_vals_raw.columns if str(c).lower() in {"ano", "year"}}
    num_vars = [c for c in df_vals_raw.columns if pd.api.types.is_numeric_dtype(df_vals_raw[c])]
    var_opts_base = sorted([c for c in num_vars if (c not in id_like | time_like) and not banre.search(str(c))])

    vars_sel_adv = st.multiselect(
        "Variáveis para testes (0–3)",
        var_opts_base,
        default=var_opts_base[: min(10, len(var_opts_base))],
        key="t2_adv_vars"
    )

    def _compute_descritiva_fallback(df_join: pd.DataFrame, vars_list: list[str], use_shapiro: bool, shapiro_max_n: int = 5000) -> pd.DataFrame:
        d = df_join[["_cl_code"] + vars_list].copy()
        for v in vars_list:
            d[v] = pd.to_numeric(d[v], errors="coerce")
        g = d.groupby("_cl_code", observed=True)
        def _to_long(name, df_):
            return df_.stack().rename(name).reset_index().rename(columns={"_cl_code": "cluster", "level_1": "variavel"})
        count_df = g[vars_list].count()
        n_total = g.size()
        miss_df = pd.DataFrame(n_total.values[:, None] - count_df.values, index=count_df.index, columns=vars_list)
        mean_df   = g[vars_list].mean()
        median_df = g[vars_list].median()
        std_df    = g[vars_list].std(ddof=1)
        min_df    = g[vars_list].min()
        max_df    = g[vars_list].max()
        q_df      = g[vars_list].quantile([0.25, 0.75])
        p25_df    = q_df.xs(0.25, level=1); p75_df = q_df.xs(0.75, level=1)
        cv_df     = std_df / mean_df
        parts = [
            _to_long("n", count_df), _to_long("missings", miss_df), _to_long("media", mean_df),
            _to_long("mediana", median_df), _to_long("desvio_padrao", std_df),
            _to_long("p25", p25_df), _to_long("p75", p75_df),
            _to_long("minimo", min_df), _to_long("maximo", max_df), _to_long("coef_var", cv_df),
        ]
        out = parts[0]
        for p in parts[1:]:
            out = out.merge(p, on=["cluster", "variavel"], how="left")
        out["cluster_label"] = out["cluster"].map(label_map)
        # Shapiro opcional
        out["shapiro_p"] = np.nan; out["shapiro_sig"] = ""
        if use_shapiro:
            try:
                from scipy.stats import shapiro as _shapiro_fn_local
            except Exception:
                _shapiro_fn_local = None
            if _shapiro_fn_local is not None:
                rng = np.random.default_rng(42)
                for cl, sub in d.groupby("_cl_code", observed=True):
                    arr = sub[vars_list].to_numpy(dtype=float)
                    for j, v in enumerate(vars_list):
                        col = arr[:, j]; col = col[np.isfinite(col)]
                        n = col.size
                        if n < 3: continue
                        if n > shapiro_max_n:
                            idx = rng.choice(n, size=shapiro_max_n, replace=False)
                            col = col[idx]
                        try:
                            p = float(_shapiro_fn_local(col).pvalue)
                        except Exception:
                            p = np.nan
                        mask = (out["cluster"] == cl) & (out["variavel"] == v)
                        out.loc[mask, "shapiro_p"] = p
                def _sig_index(p):
                    if not np.isfinite(p): return ""
                    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("•" if p < 0.10 else "ns")))
                out["shapiro_sig"] = out["shapiro_p"].apply(_sig_index)
        return out

    if not vars_sel_adv:
        st.info("Selecione ao menos uma variável para os testes.")
    else:
        if len(vars_sel_adv) > max_vars:
            st.warning(f"Você selecionou {len(vars_sel_adv)} variáveis. Processando apenas as {max_vars} primeiras.")
            vars_sel_adv = vars_sel_adv[:max_vars]

        # JOIN leve valores × clusters
        df_vals_use = df_vals_raw[[sq_col_vals] + vars_sel_adv].copy()
        df_vals_use["_SQ_norm"] = _norm_sq_series(df_vals_use[sq_col_vals])
        df_join = df_vals_use.merge(df_est_clean, on="_SQ_norm", how="inner")
        df_join = df_join[df_join["_cl_code"].isin([0, 1, 2, 3])].copy()

        df_desc = _compute_descritiva_fallback(df_join, list(vars_sel_adv), use_shapiro, shapiro_max_n=shapiro_cap)

        st.markdown("**Descritivas por cluster (0–3)**")
        piv_adv = (
            df_desc.pivot_table(
                index=["cluster_label", "variavel"],
                values=["media", "mediana", "desvio_padrao", "minimo", "p25", "p75", "maximo", "coef_var"],
                aggfunc="first",
            ).sort_index()
        )
        st.dataframe(piv_adv, use_container_width=True)
        download_df(piv_adv.reset_index(), "metricas_avancadas_por_cluster")


# -----------------------------------------------------------------------------
# ABA 3 — Univariadas & Testes (lado a lado com índice de significância)
# -----------------------------------------------------------------------------
def render_tab3():
    st.subheader("📊 Univariadas & Testes — visão lado a lado com significância")

# ---------------- Helpers específicos da aba ----------------
    def _sig_stars(p_or_q: float) -> str:
        try:
            x = float(p_or_q)
        except Exception:
            return "ns"
        if not np.isfinite(x):
            return "ns"
        if x < 0.001: return "****"
        if x < 0.01:  return "***"
        if x < 0.05:  return "**"
        if x < 0.10:  return "•"
        return "ns"

    def _first_col_like(df: pd.DataFrame, patterns: list[str]) -> str | None:
        cols = list(df.columns)
        low = [c.lower() for c in cols]
        for pat in patterns:
            for i, name in enumerate(low):
                if re.search(pat, name):
                    return cols[i]
        return None

    def _to_tidy_pairs(df_any: pd.DataFrame, analise_tipo: str) -> tuple[pd.DataFrame, str, str, str, str | None]:
        """
        Normaliza para formato 'longo' (pares):
        retorna (df_long, col_i, col_j, col_stat, col_pval|None)
        """
        df = df_any.copy()

        # Caso 1: matriz quadrada (correlações)
        is_square = (df.shape[0] == df.shape[1]) and (set(df.columns.astype(str)) == set(df.index.astype(str)))
        if is_square:
            df = df.copy()
            df.index = df.index.astype(str)
            df.columns = df.columns.astype(str)
            df_long = df.reset_index().melt(id_vars=[df.index.name or "index"], var_name="var_b", value_name="stat")
            df_long = df_long.rename(columns={df.index.name or "index": "var_a"})
            df_long = df_long[df_long["var_a"] != df_long["var_b"]]  # remove diagonal
            # dedup (mantém metade superior)
            df_long["__pair_key"] = df_long.apply(lambda r: tuple(sorted([str(r["var_a"]), str(r["var_b"])])), axis=1)
            df_long = df_long.drop_duplicates("__pair_key").drop(columns="__pair_key")
            return df_long, "var_a", "var_b", "stat", None

        # Caso 2: arquivo "pares" (linhas com A,B, estatística e p)
        col_i = _first_col_like(df, [r"(var(_| )?a$)", r"(col(_| )?a$)", r"(^i$)", r"(feature.*a)", r"(var.?1$)", r"(col.?1$)", r"(^x$)"])
        col_j = _first_col_like(df, [r"(var(_| )?b$)", r"(col(_| )?b$)", r"(^j$)", r"(feature.*b)", r"(var.?2$)", r"(col.?2$)", r"(^y$)"])
        if col_i is None or col_j is None:
            obj_cols = [c for c in df.columns if df[c].dtype.kind in ("O","U","S","b")]
            if len(obj_cols) >= 2:
                col_i, col_j = obj_cols[0], obj_cols[1]
            else:
                col_i = col_i or _first_col_like(df, [r"(var.*a|col.*a|i|x)"])
                col_j = col_j or _first_col_like(df, [r"(var.*b|col.*b|j|y)"])

        stat_candidates = [r"(^rho$)", r"(spearman)", r"(pearson)", r"(^r$)", r"(coef)", r"(^t(_| )?stat)", r"(^t$)", r"(chi2)", r"(stat.*)"]
        col_stat = _first_col_like(df, stat_candidates)
        p_candidates = [r"(^p(_?value)?$)", r"(p_?(mw|t|pearson|spearman))", r"(^pval)", r"(p-?value)"]
        col_p = _first_col_like(df, p_candidates)

        rename = {}
        if col_i and col_i != "var_a": rename[col_i] = "var_a"
        if col_j and col_j != "var_b": rename[col_j] = "var_b"
        if col_stat and col_stat != "stat": rename[col_stat] = "stat"
        if col_p and col_p != "pval": rename[col_p] = "pval"
        if rename:
            df = df.rename(columns=rename)

        keep = [c for c in ["var_a","var_b","stat","pval"] if c in df.columns]
        if len(keep) < 3:
            return pd.DataFrame(), "var_a", "var_b", "stat", "pval"
        df_long = df[keep].copy()
        return df_long, "var_a","var_b","stat", ("pval" if "pval" in df_long.columns else None)

    def _make_matrix(df_pairs: pd.DataFrame, col_i: str, col_j: str, col_val: str, sym_max=True):
        try:
            M = pairs_to_matrix(df_pairs, col_i, col_j, col_val, sym_max=sym_max)
            idx = sorted(M.index.astype(str))
            M = M.reindex(index=idx, columns=idx)
            return M
        except Exception:
            return None

    # ---------------- Controles (lado esquerdo) ----------------
    c0, c1 = st.columns([1.2, 1])
    with c0:
        versao_u = st.radio("Versão dos dados", ["originais", "winsorizados"], index=0, horizontal=True, key="uni_ver_v2")
        base_u = pick_existing_dir(
            repo,
            branch,
            ["Data/analises/original", "Data/analises/Original"] if versao_u == "originais"
            else ["Data/analises/winsorizados", "Data/analises/Winsorizados"],
        )
        analise_tipo = st.selectbox(
            "Tipo de análise",
            ["chi2", "spearman", "pearson", "ttest", "pairwise", "univariadas", "correlacao_matriz"],
            key="uni_tipo_v2",
        )
        padroes = {
            "chi2": (r"chi", r"chi2"),
            "spearman": (r"spearman",),
            "pearson": (r"pearson", r"correl"),
            "ttest": (r"ttest", r"t-test"),
            "pairwise": (r"pairwise",),
            "univariadas": (r"univariad", r"descri", r"summary"),
            "correlacao_matriz": (
                r"corr(_|.*)matrix", r"correlation(_|.*)matrix",
                r"correlacao.*matriz", r"pearson.*matrix", r"spearman.*matrix",
            ),
        }
        found = find_files_by_patterns(repo, branch, [base_u], patterns=padroes.get(analise_tipo, ()))
        if not found:
            st.info(f"Nenhum arquivo encontrado em `{base_u}` para {analise_tipo}.")
            return
        sel_file = st.selectbox("Arquivo", [f["name"] for f in found], key="uni_file_v2")
        fobj = next(x for x in found if x["name"] == sel_file)
        df_any = load_tabular(repo, fobj["path"], branch)

    with c1:
        st.markdown("**Ajustes de significância**")
        alpha = st.slider("α (nível)", 0.001, 0.10, 0.05, step=0.001, key="uni_alpha")
        use_fdr = st.checkbox("Ajustar múltiplas comparações (FDR-BH)", value=True, key="uni_use_fdr")
        heat_mode = st.radio("Heatmap colorido por:", ["coeficiente/estatística", "-log10(p) / -log10(q)"], index=0, key="uni_hm_mode")

    st.caption(f"Fonte: `{base_u}/{sel_file}`")

    # ---------------- Normalização e cálculo de significância ----------------
    # 1) Arquivos de 'univariadas' (descritivas por variável)
    if analise_tipo == "univariadas":
        st.markdown("### Visão lado a lado")
        a, b = st.columns([1.6, 2.4])

        with a:
            st.markdown("**Pré-visualização**")
            st.dataframe(df_any.head(50), use_container_width=True)
            download_df(df_any, f"univariadas_{versao_u}")

            # detecta colunas estatísticas usuais
            c_var = _first_col_like(df_any, [r"(^variavel$)", r"(feature|atributo|coluna)"])
            numeric_cols = [c for c in df_any.columns if pd.api.types.is_numeric_dtype(df_any[c])]
            sel_stats = st.multiselect(
                "Colunas numéricas para exibir",
                numeric_cols,
                default=[c for c in numeric_cols if re.search(r"(media|mediana|desvio|min|max|p25|p75)", c.lower())][:6],
                key="uni_u_cols",
            )
        with b:
            if c_var and sel_stats:
                df_show = df_any[[c_var] + sel_stats].copy().rename(columns={c_var: "variavel"})
                st.markdown("**Resumo por variável**")
                st.dataframe(df_show, use_container_width=True)
                download_df(df_show, f"univariadas_{versao_u}resumo")
            else:
                st.info("Selecione ao menos uma coluna numérica.")

        st.info("Este arquivo não traz p-valores; aqui mostramos descritivas. Se quiser significância, use os resultados de **chi2/ttest/spearman/pearson/pairwise**.")
        return  # ← antes era st.stop()

    # 2) Demais testes → converter para pares e calcular estrelas
    df_pairs, col_i, col_j, col_stat, col_p = _to_tidy_pairs(df_any, analise_tipo)
    if df_pairs.empty or col_i not in df_pairs.columns or col_j not in df_pairs.columns or col_stat not in df_pairs.columns:
        st.warning("Não foi possível inferir as colunas de pares/estatística. Exibindo arquivo bruto.")
        st.dataframe(df_any, use_container_width=True)
        download_df(df_any, f"{analise_tipo}_{versao_u}_raw")
        return  # ← antes era st.stop()

    # garante tipos
    df_pairs[col_stat] = pd.to_numeric(df_pairs[col_stat], errors="coerce")
    if col_p and col_p in df_pairs.columns:
        df_pairs[col_p] = pd.to_numeric(df_pairs[col_p], errors="coerce")

    # calcula q-value (FDR) se houver p
    if col_p and df_pairs[col_p].notna().any():
        df_pairs["pval"] = df_pairs[col_p]
        if use_fdr:
            df_pairs["qval"] = _bh_fdr(df_pairs["pval"])
            df_pairs["sigref"] = df_pairs["qval"]
            df_pairs["sig_kind"] = "q (FDR-BH)"
        else:
            df_pairs["qval"] = np.nan
            df_pairs["sigref"] = df_pairs["pval"]
            df_pairs["sig_kind"] = "p"
        df_pairs["sig"] = df_pairs["sigref"].apply(_sig_stars)
        df_pairs["significante"] = (df_pairs["sigref"] <= alpha).astype(int)
        have_p = True
    else:
        df_pairs["sig"] = "—"
        df_pairs["significante"] = np.nan
        df_pairs["sig_kind"] = "—"
        have_p = False

    # ---------------- Layout lado a lado ----------------
    left, mid, right = st.columns([1.1, 2, 2.2], gap="large")

    # (1) Cards/Resumo
    with left:
        st.markdown("#### 🔎 Resumo")
        total_tests = int(len(df_pairs))
        st.metric("Testes (pares)", total_tests)

        if have_p:
            n_sig = int((df_pairs["sigref"] <= alpha).sum())
            perc = (n_sig / total_tests * 100.0) if total_tests else 0.0
            st.metric(f"Significantes (α={alpha:g}, {df_pairs['sig_kind'].iloc[0]})", f"{n_sig} ({perc:.1f}%)")
            # top efeitos (por módulo da estatística)
            topK = min(5, total_tests)
            sub_top = df_pairs.dropna(subset=[col_stat]).copy()
            sub_top["abs_stat"] = sub_top[col_stat].abs()
            cols_top = [col_i, col_j, col_stat] + (["pval","qval"] if "pval" in sub_top.columns else [])
            top_eff = sub_top.sort_values("abs_stat", ascending=False).head(topK)[cols_top]
            st.markdown("**Maiores efeitos (|estatística|)**")
            st.dataframe(top_eff, use_container_width=True)
        else:
            st.caption("Sem p-valores neste arquivo — exibindo apenas estatísticas.")

        st.markdown("**Legenda do índice**")
        st.write("`**** <0.001`, `*** <0.01`, `** <0.05`, `• <0.10`, `ns ≥0.10`")

    # (2) Tabela filtrável
    with mid:
        st.markdown("#### 📋 Tabela — pares com índice de significância")
        vars_all = sorted(set(df_pairs[col_i].astype(str)) | set(df_pairs[col_j].astype(str)))
        sub_i = st.selectbox("Filtrar por variável A (opcional)", ["(todas)"] + vars_all, index=0, key="uni_f_i")
        sub_j = st.selectbox("Filtrar por variável B (opcional)", ["(todas)"] + vars_all, index=0, key="uni_f_j")

        df_show = df_pairs.copy()
        if sub_i != "(todas)":
            df_show = df_show[df_show[col_i].astype(str) == sub_i]
        if sub_j != "(todas)":
            df_show = df_show[df_show[col_j].astype(str) == sub_j]

        cols_out = [col_i, col_j, col_stat, "sig"]
        if "pval" in df_show.columns: cols_out += ["pval"]
        if "qval" in df_show.columns: cols_out += ["qval"]
        if "significante" in df_show.columns: cols_out += ["significante"]

        st.dataframe(df_show[cols_out].sort_values(by=col_stat, ascending=False), use_container_width=True)
        download_df(df_show[cols_out], f"{analise_tipo}_{versao_u}_pares")

    # (3) Heatmap (coeficiente/estatística OU -log10(p/q))
    with right:
        st.markdown("#### 🔥 Heatmap")
        M_stat = _make_matrix(df_pairs[[col_i, col_j, col_stat]].dropna(), col_i, col_j, col_stat, sym_max=True)

        M_sig = None
        if have_p:
            ref_col = "qval" if (use_fdr and "qval" in df_pairs.columns) else "pval"
            df_tmp = df_pairs[[col_i, col_j, ref_col]].dropna().copy()
            df_tmp["mlog10"] = -np.log10(df_tmp[ref_col].clip(lower=1e-300))
            M_sig = _make_matrix(df_tmp[[col_i, col_j, "mlog10"]], col_i, col_j, "mlog10", sym_max=True)

        if heat_mode == "coeficiente/estatística" and M_stat is not None:
            fig = px.imshow(M_stat, color_continuous_scale="Inferno", title="Heatmap — estatística/coeficiente")
            st.plotly_chart(fig, use_container_width=True)
            download_plotly_png(fig, f"heatmap_{analise_tipo}_estat")
        elif heat_mode != "coeficiente/estatística" and (M_sig is not None):
            fig = px.imshow(M_sig, color_continuous_scale="Inferno", title=f"Heatmap — -log10({'q' if use_fdr else 'p'})")
            st.plotly_chart(fig, use_container_width=True)
            download_plotly_png(fig, f"heatmap_{analise_tipo}_log10sig")
        else:
            st.info("Não foi possível montar a matriz para o heatmap com os dados fornecidos.")

    # ---------------- Rodapé: preview do arquivo bruto (debug) ----------------
    with st.expander("🔎 Debug — preview do arquivo bruto"):
        st.dataframe(df_any.head(100), use_container_width=True)
        download_df(df_any, f"{analise_tipo}_{versao_u}_raw_preview")


with tab3:
    render_tab3()
# -----------------------------------------------------------------------------
# ABA 4 — PCA|ML
# -----------------------------------------------------------------------------
with tab4:
    # (opcional) PCA prontos
    render_pca_tab_inline(
        repo, branch,
        pick_existing_dir, list_files,
        load_parquet, load_csv
    )

    st.divider()

    # ANN (como já está)
    render_ann_tab(
        repo=repo, branch=branch,
        pick_existing_dir=pick_existing_dir,
        list_files=list_files,
        load_parquet=load_parquet,
        load_csv=load_csv,
        load_gpkg=load_gpkg,
        github_fetch_bytes=github_fetch_bytes,
        make_geojson=make_geojson,
        ensure_wgs84=ensure_wgs84,
        hex_to_rgba=hex_to_rgba,
        pick_categorical=pick_categorical,
        render_geojson_layer=render_geojson_layer,
        render_line_layer=render_line_layer,
        render_point_layer=render_point_layer,
        osm_basemap_deck=osm_basemap_deck,
        deck=deck,
    )

# -----------------------------------------------------------------------------
# ABA 5 — Clusterizador (Data/clusterizador)
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("🤖 Clusterizador — métricas e comparação")
    st.caption("Lê pastas dentro de `Data/clusterizador` e mostra métricas/tabelas dos experimentos.")

    # ------ helpers locais da aba 5 (chaves únicas prefixadas t5_) ------
    def _list_subdirs(ownerrepo: str, base_dir: str, branch: str) -> list[str]:
        try:
            items = github_listdir(ownerrepo, base_dir, branch)
            return [it["name"] for it in items if isinstance(it, dict) and it.get("type") == "dir"]
        except Exception:
            return []

    def _csv_if_exists(path: str):
        try:
            return load_csv(repo, path, branch)
        except Exception:
            return None

    def _parquet_if_exists(path: str):
        try:
            return load_parquet(repo, path, branch)
        except Exception:
            return None

    # ------ base/descoberta de diretório ------
    base_cluster = pick_existing_dir(
        repo, branch,
        ["Data/clusterizador", "data/clusterizador", "Data/Clusterizador"]
    )
    st.caption(f"Diretório base: `{base_cluster}`")

    subdirs_all = _list_subdirs(repo, base_cluster, branch)
    if not subdirs_all:
        st.error("Nenhuma pasta encontrada em `Data/clusterizador`.")
        st.stop()

    # Por padrão, usa as 5 primeiras pastas (como nas figuras) para seleção única
    default_5 = subdirs_all[:5]
    colL, colR = st.columns([1.4, 1])
    with colL:
        sel_dir = st.selectbox("Pasta de teste", default_5 if default_5 else subdirs_all,
                               index=0, key="t5_sel_dir")
    with colR:
        compare_all = st.toggle("Comparar todas as pastas", value=False, key="t5_compare_all")

    # ------------------------- modo: pasta única -------------------------
    def render_single(folder_name: str):
        st.markdown(f"### 📁 {folder_name}")
        folder_path = f"{base_cluster}/{folder_name}"

        # 1) Métricas de todos os modelos
        df_met = _csv_if_exists(f"{folder_path}/metricas_todos_modelos.csv")
        if isinstance(df_met, pd.DataFrame) and not df_met.empty:
            st.markdown("#### 📊 Métricas — todos os modelos")
            st.dataframe(df_met, use_container_width=True)
            download_df(df_met, f"metricas_todos_modelos_{folder_name}")

            # Detecta coluna do modelo/algoritmo
            cols = {c.lower(): c for c in df_met.columns}
            model_col = cols.get("modelo") or cols.get("model") or cols.get("algoritmo") or list(df_met.columns)[0]
            # Numéricas
            num_cols = [c for c in df_met.columns if pd.api.types.is_numeric_dtype(df_met[c])]
            # Gráficos para as métricas mais comuns
            metric_like_max = [c for c in num_cols if any(k in c.lower() for k in
                                ["silhouette", "calinski", "ari", "nmi", "accuracy", "purity"])]
            metric_like_min = [c for c in num_cols if any(k in c.lower() for k in
                                ["davies", "db", "inertia"])]

            for mc in metric_like_max[:6]:
                fig = px.bar(df_met, x=model_col, y=mc, title=f"{mc} (quanto maior, melhor)")
                st.plotly_chart(fig, use_container_width=True)
            for mc in metric_like_min[:3]:
                fig = px.bar(df_met, x=model_col, y=mc, title=f"{mc} (quanto menor, melhor)")
                st.plotly_chart(fig, use_container_width=True)

            # Resumo (ranking simples)
            df_sum = df_met[[model_col] + metric_like_max + metric_like_min].copy()
            for c in metric_like_max:
                df_sum[f"rank_{c}"] = df_sum[c].rank(ascending=False, method="min")
            for c in metric_like_min:
                df_sum[f"rank_{c}"] = df_sum[c].rank(ascending=True, method="min")
            rank_cols = [c for c in df_sum.columns if c.startswith("rank_")]
            if rank_cols:
                df_sum["rank_medio"] = df_sum[rank_cols].mean(axis=1)
                df_rank = df_sum[[model_col, "rank_medio"] + rank_cols].sort_values("rank_medio")
                st.markdown("#### 🧾 Resumo — ranking médio por modelo")
                st.dataframe(df_rank, use_container_width=True)
                download_df(df_rank, f"resumo_ranking_{folder_name}")

        else:
            st.info("`metricas_todos_modelos.csv` não encontrado nesta pasta.")

        st.divider()

        # 2) Comparação de acurácia (se existir)
        df_acc = _csv_if_exists(f"{folder_path}/comparacao_acuracia.csv")
        if isinstance(df_acc, pd.DataFrame) and not df_acc.empty:
            st.markdown("#### 🎯 Comparação de acurácia")
            st.dataframe(df_acc, use_container_width=True)
            download_df(df_acc, f"comparacao_acuracia_{folder_name}")

            # plota todas as colunas numéricas
            cols = {c.lower(): c for c in df_acc.columns}
            who = cols.get("modelo") or cols.get("model") or cols.get("algoritmo") or list(df_acc.columns)[0]
            nums = [c for c in df_acc.columns if pd.api.types.is_numeric_dtype(df_acc[c])]
            if nums:
                mlong = df_acc.melt(id_vars=[who], value_vars=nums, var_name="métrica", value_name="valor")
                fig = px.bar(mlong, x=who, y="valor", color="métrica", barmode="group",
                             title="Acurácia / métricas por modelo")
                st.plotly_chart(fig, use_container_width=True)

        # 3) Importância de variáveis (RF/KMeans) – opcional
        df_imp = _csv_if_exists(f"{folder_path}/importancia_rf_kmeans.csv")
        if isinstance(df_imp, pd.DataFrame) and not df_imp.empty:
            st.markdown("#### 🌟 Importância de variáveis (RF/KMeans)")
            # detecta 'feature' e 'importance'
            cols = {c.lower(): c for c in df_imp.columns}
            feat = cols.get("feature") or cols.get("variavel") or cols.get("atributo") or list(df_imp.columns)[0]
            imp  = cols.get("importance") or cols.get("importancia") or (list(df_imp.columns)[1] if len(df_imp.columns) > 1 else None)
            if imp:
                top = df_imp.sort_values(imp, ascending=False).head(20)
                fig = px.bar(top.sort_values(imp), x=imp, y=feat, orientation="h", title="Top importâncias")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(top, use_container_width=True)
                download_df(top, f"importancias_top_{folder_name}")

        # 4) Resumo de clusters KMeans (se existir)
        df_cs = _csv_if_exists(f"{folder_path}/cluster_summary_kmeans.csv")
        if isinstance(df_cs, pd.DataFrame) and not df_cs.empty:
            st.markdown("#### 🧩 Resumo de clusters (KMeans)")
            st.dataframe(df_cs, use_container_width=True)
            download_df(df_cs, f"cluster_summary_kmeans_{folder_name}")

            cols = {c.lower(): c for c in df_cs.columns}
            c_cluster = cols.get("cluster") or cols.get("label") or list(df_cs.columns)[0]
            c_count   = cols.get("n") or cols.get("count") or cols.get("tamanho") or cols.get("size")
            if c_count:
                fig = px.bar(df_cs, x=c_cluster, y=c_count, title="Tamanho dos clusters (KMeans)")
                st.plotly_chart(fig, use_container_width=True)

    if not compare_all:
        render_single(sel_dir)
        st.stop()

    # ------------------------- modo: comparar todas -------------------------
    st.markdown("### 📊 Comparação entre pastas")

    # Junta metricas_todos_modelos.csv de todas as pastas
    frames = []
    for d in subdirs_all:
        df = _csv_if_exists(f"{base_cluster}/{d}/metricas_todos_modelos.csv")
        if isinstance(df, pd.DataFrame) and not df.empty:
            tmp = df.copy()
            tmp["pasta"] = d
            frames.append(tmp)
    if not frames:
        st.info("Nenhuma pasta possui `metricas_todos_modelos.csv` para comparação.")
        st.stop()

    df_all = pd.concat(frames, ignore_index=True)
    st.dataframe(df_all, use_container_width=True)
    download_df(df_all, "metricas_todos_modelos_todas_as_pastas")

    # Coluna de modelo
    cols = {c.lower(): c for c in df_all.columns}
    model_col = cols.get("modelo") or cols.get("model") or cols.get("algoritmo") or "modelo"

    # Detecta métricas numéricas
    num_cols = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c])]
    # Silhouette / Calinski / Davies principalmente
    for mc in [c for c in num_cols if any(k in c.lower() for k in ["silhouette", "calinski", "davies"])][:6]:
        fig = px.bar(
            df_all, x="pasta", y=mc, color=model_col, barmode="group",
            title=f"{mc} por pasta × modelo"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Resumo: melhor modelo por métrica/pasta + ranking médio entre pastas
    metric_like_max = [c for c in num_cols if any(k in c.lower() for k in
                        ["silhouette", "calinski", "ari", "nmi", "accuracy", "purity"])]
    metric_like_min = [c for c in num_cols if any(k in c.lower() for k in
                        ["davies", "db", "inertia"])]

    # ranking dentro de cada pasta
    group_cols = ["pasta", model_col]
    df_rank_rows = []
    for pasta, g in df_all.groupby("pasta"):
        for c in metric_like_max:
            if c in g:
                r = g[[model_col, c]].assign(rank=g[c].rank(ascending=False, method="min"))
                for _, row in r.iterrows():
                    df_rank_rows.append({"pasta": pasta, "metric": c, model_col: row[model_col], "rank": row["rank"]})
        for c in metric_like_min:
            if c in g:
                r = g[[model_col, c]].assign(rank=g[c].rank(ascending=True, method="min"))
                for _, row in r.iterrows():
                    df_rank_rows.append({"pasta": pasta, "metric": c, model_col: row[model_col], "rank": row["rank"]})
    if df_rank_rows:
        df_ranks = pd.DataFrame(df_rank_rows)
        resumo = (
            df_ranks.groupby(["metric", model_col])["rank"]
            .mean()
            .reset_index()
            .rename(columns={"rank": "rank_medio_entre_pastas"})
            .sort_values(["metric", "rank_medio_entre_pastas"])
        )
        st.markdown("#### 🧾 Tabela — ranking médio entre pastas")
        st.dataframe(resumo, use_container_width=True)
        download_df(resumo, "ranking_medio_entre_pastas")

        # gráfico por métrica
        for m in resumo["metric"].unique():
            sub = resumo[resumo["metric"] == m]
            fig = px.bar(sub.sort_values("rank_medio_entre_pastas"),
                         x="rank_medio_entre_pastas", y=model_col, orientation="h",
                         title=f"Ranking médio ({m}) — menor é melhor")
            st.plotly_chart(fig, use_container_width=True)





























