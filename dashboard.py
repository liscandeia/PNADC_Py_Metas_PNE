import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc  # Importa a biblioteca Bootstrap
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# =====================================================================
# Carrega o banco de dados único
df = pd.read_parquet('banco_metas.parquet')
# =====================================================================

# ------------------------- FUNÇÕES AUXILIARES -------------------------
def weighted_mean_std(g, value_col, weight_col):
    """
    Calcula média e desvio padrão ponderados para um grupo g.
    """
    w = g[weight_col]
    x = g[value_col]
    mean = (x * w).sum() / w.sum()
    variance = (w * (x - mean)**2).sum() / w.sum()
    std = np.sqrt(variance)
    return mean, std

def calc_ratio_stats(g):
    """
    Para a Meta 8D: calcula a razão (em %) entre os grupos 'negro' e 'branco'
    e propaga o erro (desvio padrão).
    """
    negro = g[g['negro'] == 1]
    branco = g[g['branco'] == 1]
    if len(negro)==0 or len(branco)==0:
        return (np.nan, np.nan)
    mean_negro, std_negro = weighted_mean_std(negro, 'indicador_8', 'peso')
    mean_branco, std_branco = weighted_mean_std(branco, 'indicador_8', 'peso')
    ratio = (mean_negro / mean_branco) * 100
    error = ratio * np.sqrt((std_negro/mean_negro)**2 + (std_branco/mean_branco)**2)
    return ratio, error

# ------------------------- CÁLCULOS AGREGADOS -------------------------
mask_2a = df[df['idade_cne'].between(6, 14)]
meta2a_mean = mask_2a.groupby('Ano').apply(lambda g: (g['indicador_2a'] * g['peso']).sum() / g['peso'].sum()) * 100
meta2a_std = mask_2a.groupby('Ano').apply(lambda g: weighted_mean_std(g, 'indicador_2a', 'peso')[1]) * 100

mask_2b = df[df['idade_cne'] == 16]
meta2b_mean = mask_2b.groupby('Ano').apply(lambda g: (g['indicador_2b'] * g['peso']).sum() / g['peso'].sum()) * 100
meta2b_std = mask_2b.groupby('Ano').apply(lambda g: weighted_mean_std(g, 'indicador_2b', 'peso')[1]) * 100

mask_3 = df[df['idade_cne'].between(15, 17)]
meta3a_mean = mask_3.groupby('Ano').apply(lambda g: (g['indicador_3a'] * g['peso']).sum() / g['peso'].sum()) * 100
meta3a_std = mask_3.groupby('Ano').apply(lambda g: weighted_mean_std(g, 'indicador_3a', 'peso')[1]) * 100
meta3b_mean = mask_3.groupby('Ano').apply(lambda g: (g['indicador_3b'] * g['peso']).sum() / g['peso'].sum()) * 100
meta3b_std = mask_3.groupby('Ano').apply(lambda g: weighted_mean_std(g, 'indicador_3b', 'peso')[1]) * 100

mask_8a = df[df['idade_cne'].between(18, 29)]
meta8a_mean = mask_8a.groupby('Ano').apply(lambda g: (g['indicador_8'] * g['peso']).sum() / g['peso'].sum()) * 100
meta8a_std = mask_8a.groupby('Ano').apply(lambda g: weighted_mean_std(g, 'indicador_8', 'peso')[1]) * 100

mask_8b = df[(df['idade_cne'].between(18, 29)) & (df['urbana_rural'] == 2)]
meta8b_mean = mask_8b.groupby('Ano').apply(lambda g: (g['indicador_8'] * g['peso']).sum() / g['peso'].sum()) * 100
meta8b_std = mask_8b.groupby('Ano').apply(lambda g: weighted_mean_std(g, 'indicador_8', 'peso')[1]) * 100

mask_8 = df[df['idade'].between(18, 29)]
media_negro = mask_8.groupby('Ano').apply(lambda g: (g['negro'] * g['indicador_8'] * g['peso']).sum() / (g['peso'] * g['negro']).sum()) * 100
media_branco = mask_8.groupby('Ano').apply(lambda g: (g['branco'] * g['indicador_8'] * g['peso']).sum() / (g['peso'] * g['branco']).sum()) * 100
meta8d_mean = (media_negro / media_branco) * 100
stats_negro = mask_8.groupby('Ano').apply(lambda g: weighted_mean_std(g[g['negro']==1], 'indicador_8', 'peso') if g['negro'].sum()>0 else (np.nan, np.nan))
stats_branco = mask_8.groupby('Ano').apply(lambda g: weighted_mean_std(g[g['branco']==1], 'indicador_8', 'peso') if g['branco'].sum()>0 else (np.nan, np.nan))
mean_negro = stats_negro.apply(lambda x: x[0])
std_negro = stats_negro.apply(lambda x: x[1])
mean_branco = stats_branco.apply(lambda x: x[0])
std_branco = stats_branco.apply(lambda x: x[1])
meta8d_std = meta8d_mean * np.sqrt((std_negro/mean_negro)**2 + (std_branco/mean_branco)**2)

mask_9 = df[df['idade'] >= 15]
meta9a_mean = mask_9.groupby('Ano').apply(lambda g: (g['indicador_9a'] * g['peso']).sum() / g['peso'].sum()) * 100
meta9a_std = mask_9.groupby('Ano').apply(lambda g: weighted_mean_std(g, 'indicador_9a', 'peso')[1]) * 100
meta9b_mean = mask_9.groupby('Ano').apply(lambda g: (g['indicador_9b'] * g['peso']).sum() / g['peso'].sum()) * 100
meta9b_std = mask_9.groupby('Ano').apply(lambda g: weighted_mean_std(g, 'indicador_9b', 'peso')[1]) * 100

mask_12 = df['idade'].between(18, 24)
meta12a_mean = df.groupby('Ano').apply(lambda g: (g['indicador_12a'] * g['peso']).sum() / g.loc[mask_12, 'peso'].sum()) * 100
meta12a_std = df.groupby('Ano').apply(lambda g: weighted_mean_std(g.loc[mask_12], 'indicador_12a', 'peso')[1]) * 100
meta12b_mean = df[mask_12].groupby('Ano').apply(lambda g: (g['indicador_12b'] * g['peso']).sum() / g['peso'].sum()) * 100
meta12b_std = df[mask_12].groupby('Ano').apply(lambda g: weighted_mean_std(g, 'indicador_12b', 'peso')[1]) * 100

# ------------------------- PARÂMETROS DE CADA META -------------------------
meta_options = {
    "Meta 2A": {"filter": lambda df: df['idade_cne'].between(6, 14), "indicator": "indicador_2a"},
    "Meta 2B": {"filter": lambda df: df['idade_cne'] == 16, "indicator": "indicador_2b"},
    "Meta 3A": {"filter": lambda df: df['idade_cne'].between(15, 17), "indicator": "indicador_3a"},
    "Meta 3B": {"filter": lambda df: df['idade_cne'].between(15, 17), "indicator": "indicador_3b"},
    "Meta 8A": {"filter": lambda df: df['idade_cne'].between(18, 29), "indicator": "indicador_8"},
    "Meta 8B": {"filter": lambda df: (df['idade_cne'].between(18, 29)) & (df['urbana_rural'] == 2), "indicator": "indicador_8"},
    "Meta 8D": {"filter": lambda df: df['idade'].between(18, 29), "indicator": "indicador_8", "special": True},
    "Meta 9A": {"filter": lambda df: df['idade'] >= 15, "indicator": "indicador_9a", "multiply": 100},
    "Meta 9B": {"filter": lambda df: df['idade'] >= 15, "indicator": "indicador_9b", "multiply": 100},
    "Meta 12A": {"filter": lambda df: df['idade'].between(18, 24), "indicator": "indicador_12a"},
    "Meta 12B": {"filter": lambda df: df['idade'].between(18, 24), "indicator": "indicador_12b"}
}

segmentation_vars = ['urbana_rural', 'cor', 'sexo', 'UF', 'REGIAO']

# ------------------------- FUNÇÕES DE CRIAÇÃO DOS GRÁFICOS -------------------------
def create_line_chart(series, title):
    trace = go.Scatter(
        x=series.index,
        y=series.values,
        mode='lines+markers',
        name=title
    )
    fig = go.Figure(data=[trace])
    fig.update_layout(
        title=title,
        xaxis_title='Ano',
        yaxis_title=title,
        template='plotly_white'
    )
    return dcc.Graph(figure=fig)

def create_segmented_chart(meta_key, segmentation):
    meta_def = meta_options[meta_key]
    df_filtered = df[meta_def['filter'](df)]
    
    if meta_def.get("special", False):
        if segmentation == "cor":
            df_seg = df_filtered.copy()
            df_seg['grupo_racial'] = df_seg['negro'].apply(lambda x: 'Negro' if x==1 else 'Não Negro')
            grouped = df_seg.groupby(['Ano', 'grupo_racial']).apply(
                lambda g: (g['indicador_8'] * g['peso']).sum() / g['peso'].sum()
            ).reset_index(name='weighted_avg')
            err = df_seg.groupby(['Ano', 'grupo_racial']).apply(
                lambda g: weighted_mean_std(g, 'indicador_8', 'peso')[1]
            ).reset_index(name='std')
            merged = pd.merge(grouped, err, on=['Ano','grupo_racial'])
            fig = go.Figure()
            for grp in sorted(merged['grupo_racial'].unique()):
                sub = merged[merged['grupo_racial'] == grp]
                fig.add_trace(go.Scatter(
                    x=sub['Ano'],
                    y=sub['weighted_avg'],
                    mode='lines+markers',
                    name=grp
                ))
            fig.update_layout(
                title=f"{meta_key} segmentado por Negro vs Não Negro",
                xaxis_title='Ano',
                yaxis_title='Média Ponderada',
                template='plotly_white'
            )
            return dcc.Graph(figure=fig)
        else:
            grouped = df_filtered.groupby(['Ano', segmentation]).apply(
                lambda g: calc_ratio_stats(g)
            ).reset_index(name='stats')
            grouped[['ratio', 'std']] = pd.DataFrame(grouped['stats'].tolist(), index=grouped.index)
            fig = go.Figure()
            for val in sorted(grouped[segmentation].unique()):
                sub = grouped[grouped[segmentation] == val]
                fig.add_trace(go.Scatter(
                    x=sub['Ano'],
                    y=sub['ratio'],
                    mode='lines+markers',
                    name=str(val)
                ))
            fig.update_layout(
                title=f"{meta_key} segmentado por {segmentation}",
                xaxis_title='Ano',
                yaxis_title='Razão (%)',
                template='plotly_white'
            )
            return dcc.Graph(figure=fig)
    else:
        grouped = df_filtered.groupby(['Ano', segmentation]).apply(
            lambda g: weighted_mean_std(g, meta_def['indicator'], 'peso')
        ).apply(pd.Series)
        grouped.columns = ['weighted_avg', 'std']
        if 'multiply' in meta_def:
            grouped['weighted_avg'] *= meta_def['multiply']
            grouped['std'] *= meta_def['multiply']
        grouped = grouped.reset_index()
        fig = go.Figure()
        for val in sorted(grouped[segmentation].unique()):
            sub = grouped[grouped[segmentation] == val]
            fig.add_trace(go.Scatter(
                x=sub['Ano'],
                y=sub['weighted_avg'],
                mode='lines+markers',
                name=str(val)
            ))
        fig.update_layout(
            title=f"{meta_key} segmentado por {segmentation}",
            xaxis_title='Ano',
            yaxis_title='Média Ponderada',
            template='plotly_white'
        )
        return dcc.Graph(figure=fig)

def create_meta_layout(meta_key, aggregated_mean, aggregated_std):
    segmentation_graphs = []
    for seg in segmentation_vars:
        segmentation_graphs.append(
            dbc.Card(
                dbc.CardBody([
                    html.H4(f"Segmentação por: {seg}"),
                    create_segmented_chart(meta_key, seg)
                ]),
                className="mb-4"
            )
        )
    
    if meta_key == "Meta 8D":
        agg_mean = meta8d_mean
        agg_std = meta8d_std
        title_text = "Razão (%)"
    else:
        agg_mean = aggregated_mean
        agg_std = aggregated_std
        title_text = meta_options[meta_key]['indicator']
        
    return dbc.Container([
        dbc.Row([
            dbc.Col(
                dcc.Link(
                    "←",  # Ícone da seta (Font Awesome)
                    href="/"
                ),
                width="auto",
                className="d-flex align-items-center"
            ),
            dbc.Col(
                html.H2(f"{meta_key}", className="text-center"),
                width=True,
                className="d-flex align-items-center justify-content-center"
            ),
            dbc.Col(
                html.Img(src="/assets/Design_sem_nome-removebg-preview.png", className="img-fluid", style={"height": "50px"}),
                width="auto",
                className="d-flex align-items-center justify-content-end"
            )
        ], className="mb-4"),
        dbc.Row(
            dbc.Col(create_line_chart(agg_mean, title_text), width=12)
        ),
        html.H3("Gráficos Segmentados", className="mt-4"),
        dbc.Row([dbc.Col(graph, md=6) for graph in segmentation_graphs]),
        html.Br(),
        dbc.Row(
            dbc.Col(
                dcc.Link("Voltar para Home", href="/", className="btn btn-warning"),
                width={"size": 15}
            )
        )
    ], fluid=True)

# ------------------------- LAYOUTS DAS PÁGINAS -------------------------
home_layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            html.Img(src="/assets/Design_sem_nome-removebg-preview.png", className="img-fluid mx-auto d-block"),
            width=12,
            className="text-center"
        )
    ),
    dbc.Row(
        dbc.Col(html.H3("Novo Painel de Monitoramento do Plano Nacional de Educação - PNE", className="text-center mb-4 bg-primary text-white p-3 w-100"))
    ), 
    dbc.Row([
        dbc.Col(dcc.Link(html.Button("Meta 2 - A", className="btn btn-outline-primary w-100"), href="/meta2a"), width=3),
        dbc.Col(dcc.Link(html.Button("Meta 2 - B", className="btn btn-outline-primary w-100"), href="/meta2b"), width=3),
        dbc.Col(dcc.Link(html.Button("Meta 3 - A", className="btn btn-outline-primary w-100"), href="/meta3a"), width=3),
        dbc.Col(dcc.Link(html.Button("Meta 3 - B", className="btn btn-outline-primary w-100"), href="/meta3b"), width=3)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Link(html.Button("Meta 8 - A", className="btn btn-outline-primary w-100"), href="/meta8a"), width=3),
        dbc.Col(dcc.Link(html.Button("Meta 8 - B", className="btn btn-outline-primary w-100"), href="/meta8b"), width=3),
        dbc.Col(dcc.Link(html.Button("Meta 8 - D", className="btn btn-outline-primary w-100"), href="/meta8d"), width=3),
        dbc.Col(dcc.Link(html.Button("Meta 9 - A", className="btn btn-outline-primary w-100"), href="/meta9a"), width=3)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Link(html.Button("Meta 9 - B", className="btn btn-outline-primary w-100"), href="/meta9b"), width=3),
        dbc.Col(dcc.Link(html.Button("Meta 12 - A", className="btn btn-outline-primary w-100"), href="/meta12a"), width=3),
        dbc.Col(dcc.Link(html.Button("Meta 12 - B", className="btn btn-outline-primary w-100"), href="/meta12b"), width=3)
    ])
], fluid=True)

meta2a_layout = create_meta_layout("Meta 2A", meta2a_mean, meta2a_std)
meta2b_layout = create_meta_layout("Meta 2B", meta2b_mean, meta2b_std)
meta3a_layout = create_meta_layout("Meta 3A", meta3a_mean, meta3a_std)
meta3b_layout = create_meta_layout("Meta 3B", meta3b_mean, meta3b_std)
meta8a_layout = create_meta_layout("Meta 8A", meta8a_mean, meta8a_std)
meta8b_layout = create_meta_layout("Meta 8B", meta8b_mean, meta8b_std)
meta8d_layout = create_meta_layout("Meta 8D", meta8d_mean, meta8d_std)
meta9a_layout = create_meta_layout("Meta 9A", meta9a_mean, meta9a_std)
meta9b_layout = create_meta_layout("Meta 9B", meta9b_mean, meta9b_std)
meta12a_layout = create_meta_layout("Meta 12A", meta12a_mean, meta12a_std)
meta12b_layout = create_meta_layout("Meta 12B", meta12b_mean, meta12b_std)

# ------------------------- ROTEAMENTO E EXECUÇÃO -------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/meta2a':
        return meta2a_layout
    elif pathname == '/meta2b':
        return meta2b_layout
    elif pathname == '/meta3a':
        return meta3a_layout
    elif pathname == '/meta3b':
        return meta3b_layout
    elif pathname == '/meta8a':
        return meta8a_layout
    elif pathname == '/meta8b':
        return meta8b_layout
    elif pathname == '/meta8d':
        return meta8d_layout
    elif pathname == '/meta9a':
        return meta9a_layout
    elif pathname == '/meta9b':
        return meta9b_layout
    elif pathname == '/meta12a':
        return meta12a_layout
    elif pathname == '/meta12b':
        return meta12b_layout
    else:
        return home_layout

if __name__ == '__main__':
    app.run_server(debug=True)
