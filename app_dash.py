
from __future__ import annotations
import io, os, json, tempfile, re
from pathlib import Path 
from typing import Iterable, Sequence, Any, Dict, List, Optional, Tuple
import requests
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import geopandas as gpd

API_BASE = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"

# =========================
# FASE 0 — CONFIGURAÇÃO
# =========================

# Diretórios base (atenção ao D maiúsculo em Data/)
def _root_dir() -> Path:
    # Funciona no Streamlit Cloud e local
    return Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

ROOT = _root_dir()
DATA = ROOT / "Data"
DADOS = DATA / "dados" / "Originais"
MAPA = DATA / "mapa"
RECORTES = MAPA / "recortes"

# ==== Helpers ====
WINSOR_FLAGS = ("winso", "winsor", "winsoriz", "_wins", "wins_")  # ajuste se usar outro padrão
VALID_EXTS = {".parquet", ".csv"}

def is_winsorized(filename: str) -> bool:
    name = filename.lower()
    return any(flag in name for flag in WINSOR_FLAGS)

def listar_dados(base: Path, winsor: bool) -> list[Path]:
    return sorted(
        p for p in base.iterdir()
        if p.is_file() and p.suffix in VALID_EXTS and (is_winsorized(p.name) == winsor)
    )

@st.cache_data(show_spinner=False)
def ler_tabela(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        # ajuste sep/encoding se necessário
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Extensão não suportada: {path.suffix}")

def escolher_dado(base: Path):
    # Botão de seleção (pode trocar por checkbox abaixo, se preferir)
    versao = st.radio(
        "Versão dos dados para análise:",
        ("Originais", "Winsorizados"),
        horizontal=True,
        index=0
    )
    usar_winsor = (versao == "Winsorizados")

    arquivos = listar_dados(base, winsor=usar_winsor)
    if not arquivos:
        st.error("Nenhum arquivo encontrado para a versão selecionada.")
        st.stop()

    nome_amigavel = [p.name for p in arquivos]
    escolhido = st.selectbox("Escolha o arquivo:", nome_amigavel, index=0)
    caminho = base / escolhido
    df = ler_tabela(caminho)
    return df, caminho, versao

# ==== Uso no app ====
st.subheader("Dados de análise")
df, caminho_escolhido, versao = escolher_dado(DADOS)
st.caption(f"Versão: {versao} • Arquivo: {caminho_escolhido.name}")
st.dataframe(df.head(50))

def must_exist(p: Path) -> Path:
    if not p.exists():
        st.error(f"Arquivo/pasta não encontrado: `{p}`")
        st.stop()
    return p

@st.cache_data(show_spinner=False)
def ler_parquet(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    try:
        return pd.read_parquet(p)
    except Exception as e:
        st.exception(e)
        st.stop()

@st.cache_data(show_spinner=False)
def ler_gpkg(path: str | Path, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    p = Path(path)
    # Preferir pyogrio (sem GDAL) no Streamlit Cloud
    engine = "pyogrio"
    try:
        gdf = gpd.read_file(p, layer=layer, engine=engine)
    except Exception:
        # fallback para engine padrão se pyogrio não estiver disponível
        gdf = gpd.read_file(p, layer=layer)
    # Para visualização no mapa (WGS84)
    try:
        gdf = gdf.to_crs(4326)
    except Exception:
        pass
    return gdf

def listar_layers_gpkg(p: Path) -> List[str]:
    try:
        import pyogrio  # type: ignore
        return [name for name, _, _ in pyogrio.list_layers(p)]
    except Exception:
        # fallback (pode não listar em todos ambientes)
        try:
            import fiona  # type: ignore
            with fiona.Env():
                with fiona.open(p) as src:
                    return [src.name]
        except Exception:
            return []

def sanitize_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """Evita erro 'Unexpected = ...' quando alguma célula começa com '='."""
    df2 = df.copy()
    for col in df2.columns:
        if pd.api.types.is_object_dtype(df2[col]):
            df2[col] = (
                df2[col]
                .astype(str)
                .str.replace(r"^=", "'=", regex=True)  # prefixa ' para não virar função JS
            )
    return df2

def add_lon_lat_from_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "geometry" in gdf:
        # centroid seguro (não altera geometria original)
        cent = gdf.geometry.centroid
        gdf = gdf.copy()
        gdf["lon"] = cent.x
        gdf["lat"] = cent.y
    return gdf

def preview_df(df: pd.DataFrame, caption: str):
    df = sanitize_df_for_streamlit(df)
    st.data_editor(
        df,
        use_container_width=True,
        height=min(500, 100 + 28 * min(12, len(df))),
        disabled=True,
        key=f"preview_{caption}",
    )
    st.caption(caption)

def ver_mapa(gdf: gpd.GeoDataFrame, titulo: str, color_by: Optional[str] = None):
    import pydeck as pdk

    gdf = add_lon_lat_from_geometry(gdf.dropna(subset=["geometry"]))
    if gdf.empty:
        st.warning("Camada vazia após cálculo de centroid.")
        return

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=gdf,
        get_position="[lon, lat]",
        get_radius=25,
        pickable=True,
        get_fill_color="[255, 140, 0]" if color_by is None else None,
    )

    deck = pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=float(gdf["lat"].mean()),
            longitude=float(gdf["lon"].mean()),
            zoom=11,
        ),
        layers=[layer],
        tooltip={"text": "{lon}, {lat}"},
    )
    st.pydeck_chart(deck, use_container_width=True)
    st.caption(titulo)

def painel_debug():
    st.subheader("Debug rápido")
    st.code(f"ROOT     = {ROOT}")
    st.code(f"DATA     = {DATA.exists()} -> {DATA}")
    st.code(f"DADOS    = {DADOS.exists()} -> {DADOS}")
    st.code(f"MAPA     = {MAPA.exists()} -> {MAPA}")
    if MAPA.exists():
        st.write("Arquivos em `Data/mapa`:", [p.name for p in MAPA.glob("*")])

# =========================
# FASE 1 — VARIÁVEIS (PARQUET)
# =========================

def fase_1_carregar_variaveis() -> Dict[str, pd.DataFrame]:
    must_exist(DADOS)
    # Carrega todos os pred_*.parquet automaticamente
    arquivos = sorted(DADOS.glob("pred_*.parquet"))
    if not arquivos:
        st.warning("Nenhum `pred_*.parquet` encontrado em Data/dados/Originais.")
    dfs: Dict[str, pd.DataFrame] = {}
    for p in arquivos:
        dfs[p.stem] = ler_parquet(p)
    return dfs

# =========================
# FASE 2 — MAPAS (GPKG)
# =========================

def fase_2_carregar_mapas() -> Dict[str, gpd.GeoDataFrame]:
    must_exist(MAPA)
    # Lista principal de pacotes .gpkg
    candidatos = [
        MAPA / "quadras.gpkg",
        MAPA / "linhas_trem_e_metro.gpkg",
        MAPA / "estacoes_trem_e_metro.gpkg",
    ]
    # Também pega todos .gpkg dentro de recortes/
    if RECORTES.exists():
        candidatos.extend(sorted(RECORTES.glob("*.gpkg")))

    gdfs: Dict[str, gpd.GeoDataFrame] = {}
    for gpkg in candidatos:
        if gpkg.exists():
            # Se houver várias layers, lê a primeira por padrão
            layers = listar_layers_gpkg(gpkg)
            layer = layers[0] if layers else None
            gdfs[gpkg.stem] = ler_gpkg(gpkg, layer=layer)
    if not gdfs:
        st.warning("Nenhum .gpkg encontrado em Data/mapa.")
    return gdfs

# =========================
# FASE 3 — VISUALIZAÇÃO TABULAR
# =========================

def fase_3_preview_tabelas(dfs: Dict[str, pd.DataFrame], max_rows: int = 500):
    st.header("Pré-visualização de variáveis")
    if not dfs:
        st.info("Nenhum DataFrame carregado.")
        return
    nome = st.selectbox("Escolha uma tabela", list(dfs.keys()))
    df = dfs[nome].head(max_rows)
    preview_df(df, caption=f"{nome} (primeiras {len(df)} linhas)")

# =========================
# FASE 4 — VISUALIZAÇÃO NO MAPA
# =========================

def fase_4_preview_mapa(gdfs: Dict[str, gpd.GeoDataFrame]):
    st.header("Visualização de mapas")
    if not gdfs:
        st.info("Nenhuma camada geográfica carregada.")
        return
    nome = st.selectbox("Escolha uma camada", list(gdfs.keys()))
    gdf = gdfs[nome]
    ver_mapa(gdf, titulo=nome)

# =========================
# FASE 5 — DEBUG
# =========================

def fase_5_debug():
    with st.expander("Abrir painel de debug"):
        painel_debug()

# =========================
# MAIN
# =========================

def main():
    st.set_page_config(page_title="UrbanTechCluster — Visualização", layout="wide")
    st.title("UrbanTechCluster — Variáveis e Mapas")

    # Fase 0: checagens de caminho
    must_exist(DATA)
    must_exist(DADOS)
    must_exist(MAPA)

    # Fase 1: variáveis (.parquet)
    dfs = fase_1_carregar_variaveis()

    # Fase 2: mapas (.gpkg)
    gdfs = fase_2_carregar_mapas()

    col1, col2 = st.columns(2)
    with col1:
        fase_3_preview_tabelas(dfs)
    with col2:
        fase_4_preview_mapa(gdfs)

    # Fase 5: debug
    fase_5_debug()

if __name__ == "__main__":
    main()


# -------------------- secrets / headers --------------------

def read_secret(path: Iterable[str], default=None):
    cur = st.secrets
    try:
        for p in path:
            cur = cur[p]
        return cur
    except Exception:
        return default

def github_headers():
    token = read_secret(["github", "token"], None)
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "UTClean/1.0",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def normalize_repo(owner_repo: str) -> str:
    s = (owner_repo or "").strip()
    s = s.replace("https://github.com/", "").replace("http://github.com/", "").strip("/")
    parts = [p for p in s.split("/") if p]
    if len(parts) < 2:
        raise RuntimeError("Informe no formato owner/repo (ex.: emiliobneto/UrbanTechCluster).")
    return f"{parts[0]}/{parts[1]}"

@st.cache_data(show_spinner=False, ttl=600)
def resolve_branch(owner_repo: str, branch: str | None) -> str:
    """Se branch não vier, tenta descobrir a default no GitHub. Fallback: 'main'."""
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


# -------------------- local helpers --------------------

def _local_bytes(rel_path: str) -> bytes | None:
    try:
        p = rel_path if os.path.isabs(rel_path) else os.path.join(os.getcwd(), rel_path)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                return f.read()
    except Exception:
        pass
    return None


# -------------------- GitHub bytes/listing simples --------------------

def build_raw_url(owner_repo: str, path: str, branch: str) -> str:
    owner_repo = normalize_repo(owner_repo)
    return f"{RAW_BASE}/{owner_repo}/{branch}/{path.lstrip('/')}"

def fetch_bytes(owner_repo: str, path: str, branch: str) -> bytes:
    """Local primeiro; depois tenta API raw (com token) e por fim raw.githubusercontent."""
    # 1) local
    data = _local_bytes(path)
    if data is not None:
        return data

    # 2) API raw (funciona em privado com token)
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

    # sanity: não aceitar HTML/ponteiro LFS
    head = data[:200].strip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise RuntimeError("Recebi HTML em vez do arquivo. Repo privado sem token ou rate limit.")
    if data.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError("Arquivo está em Git LFS (ponteiro). Baixe-o para local ou use token.")

    return data

@st.cache_data(show_spinner=False, ttl=600)
def list_files(owner_repo: str, path: str, branch: str, exts: Sequence[str] = (".csv", ".parquet", ".gpkg")):
    """Lista arquivos no diretório (local se houver; senão GitHub)."""
    out = []

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


# -------------------- loaders (cacheados) --------------------

@st.cache_data(show_spinner=True, ttl=600)
def load_csv(owner_repo: str, path: str, branch: str) -> pd.DataFrame:
    b = fetch_bytes(owner_repo, path, branch)
    return pd.read_csv(io.BytesIO(b), usecols=lambda c: not str(c).startswith("Unnamed"))

@st.cache_data(show_spinner=True, ttl=600)
def load_parquet(owner_repo: str, path: str, branch: str) -> pd.DataFrame:
    b = fetch_bytes(owner_repo, path, branch)
    return pd.read_parquet(io.BytesIO(b), engine="pyarrow")

@st.cache_data(show_spinner=True, ttl=1200)
def load_gpkg(owner_repo: str, path: str, branch: str, layer: str | None = None):
    try:
        import geopandas as gpd  # type: ignore
    except Exception as e:
        raise RuntimeError("Instale geopandas e pyogrio para ler GPKG (pip install geopandas pyogrio).") from e
    b = fetch_bytes(owner_repo, path, branch)
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
        tmp.write(b); tmp.flush()
        tmp_path = tmp.name
    try:
        return gpd.read_file(tmp_path, layer=layer, engine="pyogrio")
    finally:
        try: os.remove(tmp_path)
        except Exception: pass

# paleta simples
CATEGORICAL = [
    "#7c3aed", "#d946ef", "#fb7185", "#f97316", "#f59e0b",
    "#10b981", "#22d3ee", "#60a5fa", "#34d399", "#f43f5e",
]

def pick_categorical(k: int):
    if k <= len(CATEGORICAL): return CATEGORICAL[:k]
    reps = (k // len(CATEGORICAL)) + 1
    return (CATEGORICAL * reps)[:k]

def hex_to_rgba(hex_color: str, alpha: int = 180):
    try:
        h = hex_color.strip().lstrip("#")
        if len(h) == 3: h = "".join(ch*2 for ch in h)
        r, g, b = (int(h[i:i+2], 16) for i in (0,2,4))
        return [r,g,b,alpha]
    except Exception:
        return [153,153,153,alpha]

def ensure_wgs84(gdf):
    try:
        if hasattr(gdf, "crs") and gdf.crs and str(gdf.crs).lower() not in ("epsg:4326","wgs84"):
            return gdf.to_crs(4326)
    except Exception:
        pass
    return gdf

def make_geojson(gdf):
    try:
        import geopandas as gpd  # noqa
    except Exception as e:
        raise RuntimeError("geopandas é necessário para GeoJSON.") from e
    if "geometry" not in gdf.columns:
        raise RuntimeError("GeoDataFrame sem geometry.")
    gdf = ensure_wgs84(gdf)
    return json.loads(gdf.to_json())

# ---------- pydeck wrappers ----------
try:
    import pydeck as pdk
except Exception:
    pdk = None

def _layer_id(prefix: str, name: str) -> str:
    nm = re.sub(r"[^A-Za-z0-9_\-]+", "-", str(name)).strip("-") or "layer"
    return f"{prefix}-{nm}"

def layer_geojson(geojson: Dict[str, Any], name="layer"):
    if pdk is None:
        st.error("pydeck não instalado (pip install pydeck).")
        return None
    return pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        id=_layer_id("geojson", name),
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color="d => (d.properties && d.properties.fill_color) ? d.properties.fill_color : [150,150,150,150]",
        get_line_color=[80,80,80,220],
        get_line_width=1,
        line_width_min_pixels=1,
        auto_highlight=True,
    )

def deck_osm(layers, view_state=None):
    if pdk is None:
        st.error("pydeck não instalado.")
        return
    tile = pdk.Layer("TileLayer", data="https://a.tile.openstreetmap.org/{z}/{x}/{y}.png")
    r = pdk.Deck(
        layers=[tile] + [l for l in layers if l is not None],
        initial_view_state=view_state or pdk.ViewState(latitude=-23.55, longitude=-46.63, zoom=10),
        map_style=None,
        tooltip={"text": "{name}\n{value}"},
    )
    st.pydeck_chart(r, use_container_width=True)

def norm_sq_series(s: pd.Series, digits: int = 6) -> pd.Series:
    s = s.astype("string").str.replace(r"\D", "", regex=True).fillna("")
    s = s.str[-digits:].str.zfill(digits)
    return s.mask(s.eq(""))

def norm_sq_scalar(x, digits: int = 6):
    s = re.sub(r"\D", "", str(x)) if x is not None else ""
    if not s: return None
    if len(s) > digits: s = s[-digits:]
    return s.zfill(digits)

def is_categorical(series: pd.Series) -> bool:
    if series.dtype.kind in ("O","b","M","m","U","S"):
        return True
    return series.dropna().nunique() <= 12

def classify_numeric(series: pd.Series, k: int = 6):
    """Classifica por quantis (sem mapclassify). Retorna (labels_int, bins)."""
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
        # fallback: intervalo uniforme
        vmin, vmax = float(s_no_na.min()), float(s_no_na.max())
        if vmin == vmax:
            cats = pd.Series([0]*len(s_no_na), index=s_no_na.index, dtype="Int64")
            bins = np.array([vmin, vmax])
        else:
            bins = np.linspace(vmin, vmax, num=k+1)
            idx = np.digitize(s_no_na, bins[1:-1], right=True)
            cats = pd.Series(idx, index=s_no_na.index, dtype="Int64")
    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    out.loc[cats.index] = cats
    return out, bins

def render_tab1(repo: str, branch: str):
    st.subheader("🗺️ Mapa — Quadras + Dados por SQ")

    # 1) Quadras
    quadras_path = "Data/mapa/quadras.gpkg"
    try:
        gdf = load_gpkg(repo, quadras_path, branch)
    except Exception as e:
        st.error(f"Não consegui ler `{quadras_path}` ({e}). Se preferir, coloque o arquivo localmente.")
        return

    gdf = ensure_wgs84(gdf)
    sq_geo = next((c for c in gdf.columns if str(c).upper() == "SQ"), None)
    if not sq_geo:
        st.error("Camada de quadras precisa ter coluna 'SQ'.")
        return

    # 2) Dados por SQ (originais / winsorize)
    colA, colB = st.columns([2, 1])
    with colA:
        src = st.radio("Origem dos dados", ["originais", "winsorize"], horizontal=True, index=0, key="t1_src")
        base_dir = f"Data/dados/{src}"
        files = list_files(repo, base_dir, branch, exts=(".parquet", ".csv"))
        files = [f for f in files if f["name"].lower().endswith((".parquet", ".csv"))]
        if not files:
            st.warning(f"Nenhum .parquet/.csv em `{base_dir}` (local ou GitHub).")
            return
        sel_name = st.selectbox("Arquivo de dados", [f["name"] for f in files], key="t1_file")
        fobj = next((x for x in files if x["name"] == sel_name), None)
        if not fobj:
            st.error("Seleção inválida de arquivo.")
            return
        df = load_parquet(repo, fobj["path"], branch) if sel_name.lower().endswith(".parquet") else load_csv(repo, fobj["path"], branch)

    with colB:
        # Ano
        year_col = next((c for c in df.columns if str(c).lower() in ("ano","year")), None)
        if year_col:
            anos = sorted(pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int).unique().tolist())
            if anos:
                ano_sel = st.select_slider("Ano", options=anos, value=anos[-1], key="t1_ano")
                df = df[pd.to_numeric(df[year_col], errors="coerce").astype("Int64") == ano_sel]
        # variável
        id_like = {c for c in df.columns if str(c).lower() in {"sq","id","codigo","code","_sq_norm"}}
        time_like = {c for c in df.columns if str(c).lower() in {"ano","year"}}
        candidates = [c for c in df.columns if c not in (id_like|time_like)]
        if not candidates:
            st.error("Não encontrei variáveis para mapear.")
            return
        var = st.selectbox("Variável a mapear", candidates, key="t1_var")

    # 3) JOIN
    df = df.copy()
    sq_df = next((c for c in df.columns if str(c).upper() == "SQ"), None)
    if not sq_df:
        st.error("Dataset selecionado não possui coluna 'SQ'.")
        return
    df["_SQ_norm"] = norm_sq_series(df[sq_df])
    gdf["_SQ_norm"] = norm_sq_series(gdf[sq_geo])

    gjoin = gdf[[sq_geo, gdf.geometry.name, "_SQ_norm"]].merge(
        df[["_SQ_norm", var]],
        on="_SQ_norm",
        how="left",
    )

    # 4) Pintura (categórica ou numérica — auto)
    palette = None
    legend = []
    props_col = "value"

    if is_categorical(df[var]):
        # categórica
        vals = gjoin[var].astype("string")
        cats = [c for c in vals.dropna().unique()]
        cats_sorted = sorted(cats, key=lambda x: str(x))
        palette = {cats_sorted[i]: pick_categorical(len(cats_sorted))[i] for i in range(len(cats_sorted))}
        gjoin[props_col] = vals
        legend = [("cat", k, palette[k]) for k in cats_sorted]
    else:
        # numérica por quantis (slider)
        k = st.slider("Quebras (quantis)", 3, 9, 6, key="t1_k")
        labels, bins = classify_numeric(gjoin[var], k=k)
        gjoin[props_col] = labels
        pal = px_seq(k)  # sequential Viridis simples
        palette = {i: pal[i] for i in range(len(pal))}
        legend = [("num", i, palette[i]) for i in range(len(pal))]

    # 5) GeoJSON e mapa
    gj = make_geojson(gjoin[[props_col, gjoin.geometry.name]].rename(columns={gjoin.geometry.name: "geometry"}))
    for feat in gj.get("features", []):
        v = feat.get("properties", {}).get(props_col, None)
        color = palette.get(v, "#999999")
        feat.setdefault("properties", {})["fill_color"] = hex_to_rgba(color)

    lyr = layer_geojson(gj, name="quadras")
    deck_osm([lyr])

    # 6) Legenda
    st.markdown("**Legenda**")
    if not legend:
        st.caption("Sem classes definidas.")
    else:
        for kind, k, hexc in legend:
            if kind == "cat":
                st.write(f"▉ {k}")
            else:
                st.write(f"▉ classe {k}")

    # 7) Tabela (SQ + var) e download
    expo = gjoin[[sq_geo, var]].rename(columns={sq_geo: "SQ"})
    st.dataframe(expo.head(200), use_container_width=True)
    csv = expo.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Baixar CSV (SQ + variável)", csv, file_name=f"dados_{var}.csv", mime="text/csv")


# Viridis simples (sem dependência externa no módulo)
def px_seq(k: int):
    from plotly import express as px
    base = px.colors.sequential.Viridis
    if k >= len(base): return base
    import numpy as np
    idxs = np.linspace(0, len(base)-1, k).round().astype(int).tolist()
    return [base[i] for i in idxs]

def render_tab2(repo: str, branch: str):
    st.subheader("🧬 Clusterização — mapa + resumo")

    # 1) Escolher arquivo EstagioClusterizacao.* (upload OU pasta padrão)
    colL, colR = st.columns([2,1])
    with colL:
        up = st.file_uploader("Upload (opcional) EstagioClusterizacao.csv/parquet", type=["csv", "parquet"], key="t2_up")
    df_est = None
    source = ""

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
        candidates = [f for f in files if re.search(r"(?i)est[aá]gio.*cluster", f["name"]) or f["name"].lower() == "estagioclusterizacao.csv" or f["name"].lower() == "estagioclusterizacao.parquet"]
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
    # coluna do cluster (tenta vários nomes comuns)
    cand_cols = [c for c in df_est.columns if re.search(r"(?i)(cluster|est[aá]gio|label)", c)]
    if not cand_cols:
        st.error("Não encontrei coluna de cluster (ex.: EstagioClusterizacao, Cluster, Label).")
        return
    cl_col = next((c for c in cand_cols if c.lower() == "estagioclusterizacao"), cand_cols[0])

    # ano (opcional)
    ano_col = next((c for c in df_est.columns if str(c).lower() in ("ano","year")), None)
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
    # 4) Cores por categoria
    cats = sorted(g[cl_col].dropna().astype(str).unique().tolist())
    pal = pick_categorical(len(cats))
    cmap = {cats[i]: pal[i] for i in range(len(cats))}

    gj = make_geojson(g[[cl_col, "geometry"]])
    for feat in gj.get("features", []):
        v = feat.get("properties", {}).get(cl_col, None)
        hexc = cmap.get(str(v), "#999999")
        feat.setdefault("properties", {})["fill_color"] = hex_to_rgba(hexc)

    st.caption(f"Fonte clusters: {source}")
    lyr = layer_geojson(gj, name="clusters")
    deck_osm([lyr])

    # 5) Resumo simples
    st.markdown("### Resumo")
    freq = g[cl_col].astype("string").value_counts(dropna=False).rename_axis("cluster").reset_index(name="n")
    st.dataframe(freq, use_container_width=True)
    fig = px.bar(freq[freq["cluster"].notna()], x="cluster", y="n", title="Contagem por cluster")
    st.plotly_chart(fig, use_container_width=True)

def _try_load_clusters(repo: str, branch: str) -> pd.DataFrame | None:
    """Tenta carregar EstagioClusterizacao.* de Data/dados/Originais"""
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


def render_tab3(repo: str, branch: str):
    st.subheader("📊 Univariadas — distribuição e estatísticas")

    # 1) Escolher dataset
    colA, colB = st.columns([2, 1])
    with colA:
        src = st.radio("Origem dos dados", ["originais", "winsorize"], horizontal=True, index=0, key="t3_src")
        base_dir = f"Data/dados/{src}"
        files = list_files(repo, base_dir, branch, (".parquet", ".csv"))
        if not files:
            st.warning(f"Nenhum .parquet/.csv em `{base_dir}` (local ou GitHub).")
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

    # 2) Variável
    id_like = {c for c in df.columns if str(c).lower() in {"sq", "id", "codigo", "code"}}
    time_like = {c for c in df.columns if str(c).lower() in {"ano", "year"}}
    candidates = [c for c in df.columns if c not in (id_like | time_like)]
    if not candidates:
        st.warning("Não encontrei variáveis para analisar.")
        return
    var = st.selectbox("Variável", candidates, index=0, key="t3_var")

    # 3) (Opcional) juntar clusters
    join_clusters = st.checkbox("Juntar EstagioClusterizacao (opcional)", value=False, key="t3_joincl")
    df_cl = None
    cl_col = None
    if join_clusters:
        df_cl = _try_load_clusters(repo, branch)
        if df_cl is None:
            st.info("Arquivo EstagioClusterizacao.* não encontrado.")
        else:
            sq_a = next((c for c in df.columns if str(c).upper() == "SQ"), None)
            sq_b = next((c for c in df_cl.columns if str(c).upper() == "SQ"), None)
            if sq_a and sq_b:
                df["_SQ_norm"] = norm_sq_series(df[sq_a])
                df_cl["_SQ_norm"] = norm_sq_series(df_cl[sq_b])
                cl_cands = [c for c in df_cl.columns if "estagio" in c.lower() or "cluster" in c.lower()]
                cl_col = cl_cands[0] if cl_cands else None
                if cl_col:
                    df = df.merge(df_cl[["_SQ_norm", cl_col]], on="_SQ_norm", how="left")

    # 4) Visualizações
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

    # 5) Download
    csv = df[[c for c in [var, cl_col] if c in df.columns]].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Baixar CSV (variável + cluster se houver)", csv, file_name=f"univariada_{var}.csv", mime="text/csv")

def _classify_pca_file(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    if "explained_variance_ratio" in cols or any("variancia" in c and "explic" in c for c in cols):
        return "evr"
    if ("component" in cols and any(x in cols for x in ["loading","valor","carga"])) or any(str(c).lower().startswith("pc") for c in df.columns):
        # pode ser loadings wide/long
        if "component" in cols and any(x in cols for x in ["loading","valor","carga"]):
            return "loadings_long"
        pc_like = [c for c in df.columns if str(c).lower().startswith("pc")]
        return "loadings_wide" if len(pc_like) >= 2 else "unknown"
    pc_cols = [c for c in df.columns if str(c).lower().startswith("pc")]
    if pc_cols:
        return "scores"
    return "unknown"


def _tidy_loadings(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    # long
    if "component" in cols and any(k in cols for k in ["loading","valor","carga"]):
        comp = cols.get("component")
        load = cols.get("loading") or cols.get("valor") or cols.get("carga")
        # variável: primeira não numérica diferente de comp
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
        return long[["variable","component","loading"]]
    return pd.DataFrame(columns=["variable","component","loading"])


def _safe_literal_list(x):
    if isinstance(x, (list,tuple)):
        return list(x)
    s = str(x).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list,tuple)):
            return list(v)
    except Exception:
        pass
    s2 = s.strip("[]()")
    parts = [p.strip() for p in s2.split(",")]
    out = []
    for p in parts:
        if p:
            try: out.append(float(p))
            except Exception: pass
    return out


def render_tab4(repo: str, branch: str):
    st.subheader("🧠 PCA — variância, loadings e scores (sem recálculo)")

    base = "Data/analises/PCA"
    files = list_files(repo, base, branch, (".csv", ".parquet"))
    if not files:
        st.info("Nenhum arquivo encontrado em `Data/analises/PCA`.")
        return

    names = [f["name"] for f in files]
    st.markdown("### 1) Variância explicada (scree)")
    evr_name = st.selectbox("Arquivo de variância explicada", names, key="t4_evr")
    evr_obj = next((x for x in files if x["name"] == evr_name), None)
    df_evr = load_parquet(repo, evr_obj["path"], branch) if evr_name.lower().endswith(".parquet") else load_csv(repo, evr_obj["path"], branch)
    kind = _classify_pca_file(df_evr)

    if kind == "evr":
        cols = {c.lower(): c for c in df_evr.columns}
        if "explained_variance_ratio" in cols:
            evr_col = cols["explained_variance_ratio"]
            dfp = df_evr[[evr_col]].copy()
            dfp["component"] = [f"PC{i+1}" for i in range(len(dfp))]
            dfp["explained_variance_ratio"] = pd.to_numeric(dfp[evr_col], errors="coerce")
        else:
            # pode vir como um array em uma célula
            row = df_evr.iloc[0]
            arr = None
            for c in df_evr.columns:
                if "variancia" in c.lower() or "explained" in c.lower():
                    arr = _safe_literal_list(row[c])
                    break
            if not arr:
                st.dataframe(df_evr.head(), use_container_width=True)
                st.warning("Não identifiquei a coluna de variância.")
                return
            dfp = pd.DataFrame({"component":[f"PC{i+1}" for i in range(len(arr))],
                                "explained_variance_ratio": arr})
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
    st.markdown("### 2) Loadings")
    load_name = st.selectbox("Arquivo de loadings", names, index=min(1, len(names)-1), key="t4_load")
    load_obj = next((x for x in files if x["name"] == load_name), None)
    df_load = load_parquet(repo, load_obj["path"], branch) if load_name.lower().endswith(".parquet") else load_csv(repo, load_obj["path"], branch)

    long = _tidy_loadings(df_load)
    if long.empty:
        st.info("Não foi possível identificar estrutura de loadings.")
        st.dataframe(df_load.head(), use_container_width=True)
    else:
        comps = sorted(long["component"].astype(str).unique(), key=lambda x: (len(x), x))
        c1, c2 = st.columns([2,1])
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
    st.markdown("### 3) Scores (dispersão)")
    score_name = st.selectbox("Arquivo de scores", names, index=min(2, len(names)-1), key="t4_scores")
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
    st.dataframe(df_sc[[pcx, pcy] + ([] if color_col=="(nenhum)" else [color_col])].head(200), use_container_width=True)

def _list_subdirs(repo: str, base: str, branch: str):
    """Lista subpastas dentro de base (local → preferido; fallback GitHub contents)."""
    out = []
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

    # GitHub
    try:
        url = f"https://api.github.com/repos/{normalize_repo(repo)}/contents/{base}?ref={branch}"
        r = requests.get(url, headers=github_headers(), timeout=15)
        if r.status_code == 200 and isinstance(r.json(), list):
            for it in r.json():
                if it.get("type") == "dir":
                    out.append(it.get("name"))
    except Exception:
        pass
    return out


def _load_first_existing(repo: str, base: str, branch: str, names: list[str]) -> tuple[pd.DataFrame | None, str | None]:
    files = list_files(repo, base, branch, (".csv", ".parquet", ".json", ".txt"))
    low_map = {f["name"].lower(): f for f in files}
    for nm in names:
        f = low_map.get(nm.lower())
        if f:
            if f["name"].lower().endswith(".parquet"):
                try: return load_parquet(repo, f["path"], branch), f["name"]
                except Exception: continue
            if f["name"].lower().endswith(".csv"):
                try: return load_csv(repo, f["path"], branch), f["name"]
                except Exception: continue
            if f["name"].lower().endswith(".json"):
                try:
                    # retorna DataFrame com json embutido para exibir
                    from simple_io import fetch_bytes  # lazy import
                    b = fetch_bytes(repo, f["path"], branch)
                    return pd.DataFrame({"__raw_json__":[json.loads(b.decode("utf-8", errors="replace"))]}), f["name"]
                except Exception: continue
            if f["name"].lower().endswith(".txt"):
                try:
                    from simple_io import fetch_bytes
                    b = fetch_bytes(repo, f["path"], branch)
                    return pd.DataFrame({"__raw__":[b.decode("utf-8", errors="replace")]}), f["name"]
                except Exception: continue
    return None, None


def render_tab5(repo: str, branch: str):
    st.subheader("🤖 Clusterizador (ANN) — relatórios e mapas")

    ann_root = "Data/ANN"
    subdirs = _list_subdirs(repo, ann_root, branch)
    run_sel = st.selectbox("Execução (subpasta em Data/ANN)", options=(["(raiz)"] + subdirs) if subdirs else ["(raiz)"], key="t5_run")
    base = ann_root if run_sel == "(raiz)" else f"{ann_root}/{run_sel}"
    st.caption(f"Lendo arquivos de: `{base}`")

    # 1) Histórico por época
    st.markdown("### 📈 Histórico por época")
    df_hist, hist_name = _load_first_existing(repo, base, branch, ["metrics_over_epochs.csv","keras_history.csv","val_metrics_per_epoch.csv"])
    if isinstance(df_hist, pd.DataFrame) and not df_hist.empty and "__raw__" not in df_hist.columns:
        st.caption(f"Arquivo: {hist_name}")
        # detect epoch col
        epoch_col = next((c for c in df_hist.columns if c.lower() == "epoch"), None)
        # métricas comuns
        metric_cols = [c for c in df_hist.columns if any(k in c.lower() for k in ["loss","acc","auc","precision","recall","f1","mae","mse","rmse"])]

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
    df_auc, auc_name = _load_first_existing(repo, base, branch, ["auc_summary.csv","roc_auc.csv"])
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
    df_cr, cr_name = _load_first_existing(repo, base, branch, ["classificationreport.json","classificationreport.txt","classificationreport.csv"])
    if isinstance(df_cr, pd.DataFrame) and not df_cr.empty:
        if "__raw_json__" in df_cr.columns:
            data = df_cr["__raw_json__"].iloc[0]
            try:
                df_show = pd.DataFrame(data).T.reset_index().rename(columns={"index":"label"})
            except Exception:
                df_show = pd.json_normalize(data)
        elif "__raw__" in df_cr.columns:
            # tenta parse simples: label prec rec f1 support
            lines = [l for l in df_cr["__raw__"].iloc[0].splitlines() if l.strip()]
            rows = []
            for ln in lines:
                parts = [p for p in ln.strip().split() if p]
                if len(parts) >= 5 and parts[0].lower() not in {"accuracy"}:
                    label = " ".join(parts[:-4]); prec, rec, f1, sup = parts[-4:]
                    def _float(x): 
                        try: return float(x)
                        except: return np.nan
                    rows.append({"label":label,"precision":_float(prec),"recall":_float(rec),"f1":_float(f1),"support":_float(sup)})
                elif ln.lower().startswith("accuracy"):
                    toks = ln.split()
                    try:
                        acc = float(toks[1]); rows.append({"label":"accuracy","f1":acc})
                    except Exception:
                        pass
            df_show = pd.DataFrame(rows) if rows else pd.DataFrame({"raw":[df_cr["__raw__"].iloc[0]]})
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
    pred_df, pred_name = _load_first_existing(repo, base, branch, ["df25_com_previsoes.csv","predictions_df25_with_meta.csv","test_predictions_with_meta.csv"])
    if isinstance(pred_df, pd.DataFrame) and not pred_df.empty and "__raw__" not in pred_df.columns:
        # tenta detectar colunas
        sq_col = next((c for c in pred_df.columns if str(c).upper() == "SQ"), None)
        if not sq_col:
            sq_col = next((c for c in pred_df.columns if str(c).lower() in ("id","codigo","code")), None)

        cat_candidates = [c for c in pred_df.columns 
                          if pred_df[c].dtype.kind in ("O","U","S") and c != sq_col and pred_df[c].nunique() <= 30]
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
            g = gdf.copy()
            g[map_col] = pd.NA

        cats = sorted(g[map_col].dropna().astype(str).unique().tolist())
        pal = pick_categorical(len(cats))
        cmap = {cats[i]: pal[i] for i in range(len(cats))}

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


st.set_page_config(page_title="UrbanTechCluster — simples", layout="wide")

with st.sidebar:
    st.header("🔗 Dados")
    repo_in = st.text_input("owner/repo (opcional se arquivos estiverem locais)", value="emiliobneto/UrbanTechCluster")
    branch_in = st.text_input("branch (vazio = default do repo)", value="")
    if st.button("🧹 Limpar cache"):
        st.cache_data.clear(); st.cache_resource.clear()
        st.success("Caches limpos — recarregue a página.")

repo = repo_in.strip()
branch = resolve_branch(repo, branch_in.strip()) if repo else "main"
if repo:
    try:
        repo = normalize_repo(repo)
        st.caption(f"Usando: **{repo}@{branch}** (prioriza arquivos locais; GitHub é fallback).")
    except Exception as e:
        st.error(f"Repo inválido: {e}")
        repo = ""

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Principal", "🧬 Clusterização", "📊 Univariadas", "🧠 ML → PCA", "🤖 Clusterizador"])

with tab1:
    render_tab1(repo or "", branch)

with tab2:
    render_tab2(repo or "", branch)

with tab3:
    render_tab3(repo or "", branch)

with tab4:
    render_tab4(repo or "", branch)

with tab5:
    render_tab5(repo or "", branch)


