# -*- coding: utf-8 -*-
"""
app_main.py — Aplicação Streamlit principal (4 abas)
Dependências locais: app_core.py (no mesmo diretório).
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Importa helpers do módulo core
from app_core import (
    normalize_repo,
    resolve_branch,
    pick_existing_dir,
    github_tree_paths,
    list_files,
    load_gpkg,
    load_parquet,
    load_csv,
    load_tabular,
    find_files_by_patterns,
    pairs_to_matrix,
    hex_to_rgba,
    pick_categorical,
    pick_sequential,
    is_categorical,
    make_geojson,
    render_geojson_layer,
    render_line_layer,
    render_point_layer,
    deck,
    osm_basemap_deck,
    render_legend_categorical,
    render_legend_numeric,
    render_pca_tab_inline,
)

# ==========================
# CONFIG GERAL
# ==========================
st.set_page_config(
    page_title="MODELO DE REDE NEURAL ARTIFICIAL — Clusters SP",
    page_icon="🧠",
    layout="wide",
)
TITLE = (
    "MODELO DE REDE NEURAL ARTIFICIAL PARA MAPEAMENTO DE CLUSTERS DE INTELIGÊNCIA "
    "E SUA APLICAÇÃO NO MUNICÍPIO DE SÃO PAULO"
)
st.title(TITLE)

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
with tab1:
    st.subheader("Quadras e camadas adicionais (GPKG)")
    colA, colB = st.columns([2, 1], gap="large")
    with colA:
        st.caption("Carrega `Data/mapa/quadras.gpkg` e camadas auxiliares.")
    with colB:
        basemap = st.radio(
            "Plano de fundo", ["OpenStreetMap", "Satélite (Mapbox)"], index=0, key="main_base"
        )

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
            candidates = sorted(
                candidates,
                key=lambda p: ("/data/" not in p.lower(), "/mapa/" not in p.lower(), len(p)),
            )
            if not candidates:
                st.error(
                    f"Não encontrei 'quadras.gpkg'. Erro ao tentar '{quadras_path_default}': {first_err}"
                )
                st.stop()
            sel_quadras = st.selectbox(
                "Selecione o arquivo de quadras:", candidates, index=0, key="quadras_tab1"
            )
            gdf_quadras = load_gpkg(repo, sel_quadras, branch)
            st.success(f"Carregado: {sel_quadras}")
        st.session_state["gdf_quadras_cached"] = gdf_quadras

    # camadas auxiliares
    try:
        mapa_dir = pick_existing_dir(repo, branch, ["Data/mapa", "data/mapa", "Data/Mapa"])
        mapa_files = list_files(repo, mapa_dir, branch, (".gpkg",))
        other_layers = [f for f in mapa_files if f["name"].lower() != "quadras.gpkg"]
        layer_names = [f["name"] for f in other_layers]
        sel_layers = st.multiselect(
            "Camadas auxiliares (opcional)", layer_names, default=[], key="main_layers"
        )
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
    col1, col2, col3 = st.columns([1.6, 1, 1.2], gap="large")
    with col1:
        src_label = st.radio(
            "Origem dos dados", ["originais", "winsorize"], index=0, horizontal=True, key="main_src"
        )
        base_dir = pick_existing_dir(
            repo,
            branch,
            [
                f"Data/dados/{src_label}",
                f"Data/dados/{'Originais' if src_label=='originais' else 'winsorizados'}",
                f"Data/dados/{'originais' if src_label=='originais' else 'winsorize'}",
            ],
        )
        # Listagem de arquivos .parquet (com opção de incluir/excluir predições)
        parquets_all = list_files(repo, base_dir, branch, (".parquet",))
        incl_pred = st.checkbox("Incluir arquivos de predição (pred_*)", value=True, key="main_incl_pred")
        parquet_files = [
            f for f in parquets_all
            if incl_pred or not f["name"].lower().startswith("pred_")
        ]
        if not parquet_files:
            st.warning(f"Nenhum .parquet encontrado em {base_dir}.")
            st.stop()
        
        sel_file = st.selectbox(
            "Arquivo .parquet com variáveis",
            [f["name"] for f in parquet_files],
            key="main_varfile",
        )
        fobj = next(x for x in parquet_files if x["name"] == sel_file)
        df_vars = load_parquet(repo, fobj["path"], branch)

    with col2:
        # Detecta coluna SQ (case-insensitive)
        join_col = next((c for c in df_vars.columns if str(c).upper() == "SQ"), None)
        if join_col is None:
            st.error("Dataset selecionado não possui coluna 'SQ'.")
            st.stop()
        # Detecta coluna de ano (case-insensitive)
        years_col = next((c for c in df_vars.columns if str(c).lower() in ("ano", "year")), None)
        years = (
            sorted([int(y) for y in df_vars[years_col].dropna().unique()]) if years_col else []
        )
        year = (
            st.select_slider("Ano", options=years, value=years[-1], key="main_ano") if years else None
        )
        if year is not None and years_col:
            df_vars = df_vars[df_vars[years_col] == year]

    with col3:
        # Variáveis candidatas: numéricas e que NÃO são id/tempo
        id_like = {c for c in df_vars.columns if str(c).lower() in {"sq", "id", "codigo", "code"}}
        time_like = {c for c in df_vars.columns if str(c).lower() in {"ano", "year"}}
        ignore_cols = id_like | time_like
        num_cols = [c for c in df_vars.columns if pd.api.types.is_numeric_dtype(df_vars[c])]
        var_options = [c for c in num_cols if c not in ignore_cols]
        if not var_options:
            var_options = [c for c in df_vars.columns if c not in ignore_cols]
        var_sel = st.selectbox("Variável a mapear", var_options, key="main_varname")
        n_classes = st.slider("Quebras (Jenks)", min_value=4, max_value=8, value=6, key="main_jenks")

    # merge com quadras
    sq_col_quadras = (
        "SQ" if "SQ" in gdf_quadras.columns else next((c for c in gdf_quadras.columns if str(c).upper() == "SQ"), None)
    )
    if not sq_col_quadras:
        st.error("Camada de quadras não possui coluna 'SQ'.")
        st.stop()
    gdf = gdf_quadras.merge(
        df_vars[[join_col, var_sel]], left_on=sq_col_quadras, right_on=join_col, how="left"
    )

    # classificação + legenda
    series = gdf[var_sel]
    if is_categorical(series):
        cats = [c for c in series.dropna().unique()]
        palette = pick_categorical(len(cats))
        try:
            cats_sorted = sorted(cats, key=lambda x: float(x))
        except Exception:
            cats_sorted = sorted(cats, key=lambda x: str(x))
        cmap = {cat: palette[i] for i, cat in enumerate(cats_sorted)}
        gdf["value"] = series
        legend_kind = "categorical"
        legend_info = cmap
    else:
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
        legend_kind = "numeric"
        legend_info = (bins, palette)

    geojson = make_geojson(gdf)
    for feat in geojson.get("features", []):
        val = feat.get("properties", {}).get("value", None)
        hexc = cmap.get(val, "#999999") if legend_kind == "numeric" else legend_info.get(val, "#999999")
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

    st.markdown("#### Mapa — Quadras + Camadas auxiliares")
    map_col, legend_col = st.columns([4, 1], gap="large")
    with map_col:
        if basemap.startswith("Satélite"):
            deck(layers, satellite=True)
        else:
            osm_basemap_deck(layers)
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
        rec_dir = pick_existing_dir(
            repo, branch, ["Data/mapa/recortes", "Data/Mapa/recortes", "data/mapa/recortes"]
        )
        recorte_files = list_files(repo, rec_dir, branch, (".gpkg",))
        if not recorte_files:
            st.info("Nenhum GPKG de recorte encontrado.")
        else:
            rec_sel = st.selectbox(
                "Arquivo de recorte", [f["path"] for f in recorte_files], index=0, key="main_rec_file"
            )
            gdf_rec = load_gpkg(repo, rec_sel, branch)
            layers_rec = [render_line_layer(make_geojson(gdf_rec), name="recorte")]
            st.markdown("#### Mapa — Recorte selecionado")
            col_m, col_l = st.columns([4, 1], gap="large")
            with col_m:
                deck(layers_rec, satellite=basemap.startswith("Satélite"))
            with col_l:
                st.markdown("**Legenda — Recorte**")
                from app_core import _legend_row

                _legend_row("#444444", "Contorno do recorte")
    except Exception as e:
        st.warning(f"Não foi possível listar/ler recortes: {e}")

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
        from app_core import _legend_row

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
                from app_core import _legend_row

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
