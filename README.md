# Dashboard de Monitoramento do PNE

Este repositório contém um dashboard replicando as principais metas do Plano Nacional de Educação (PNE) que utilizam os dados da PNAD Contínua. O aplicativo foi desenvolvido em Python, utilizando as bibliotecas [Dash](https://dash.plotly.com/), [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/), [Plotly](https://plotly.com/python/), [Pandas](https://pandas.pydata.org/) e [Numpy](https://numpy.org/).


## Visão Geral

O dashboard exibe diversas métricas (Metas) relacionadas ao PNE, que incluem:

- **Meta 2A**: Indicador para a faixa etária de 6 a 14 anos.
- **Meta 2B**: Indicador para 16 anos.
- **Meta 3A e Meta 3B**: Indicadores para a faixa etária de 15 a 17 anos.
- **Meta 8A, 8B e Meta 8D**: Indicadores para a faixa etária de 18 a 29 anos, com a Meta 8D apresentando análises segmentadas por raça.
- **Meta 9A e Meta 9B**: Indicadores para indivíduos a partir de 15 anos.
- **Meta 12A e Meta 12B**: Indicadores para a faixa etária de 18 a 24 anos.

Os gráficos são gerados de forma interativa e podem ser segmentados por variáveis como:
- Urbana/Rural
- Cor
- Sexo
- UF
- Região

## Estrutura do Projeto

- **`dashboard.py`**: Script principal do aplicativo Dash. Contém:
  - Leitura dos dados a partir do arquivo `banco_metas.parquet`;
  - Funções auxiliares para cálculos de média ponderada e desvio padrão;
  - Cálculos agregados e definição de indicadores;
  - Funções para criação dos gráficos (linhas e segmentados);
  - Layouts para as diferentes metas e roteamento das páginas.
  
- **`etl.ipynb`**: Notebook contendo o processo de ETL (Extração, Transformação e Carga) dos dados utilizados no dashboard.

- **`banco_metas.parquet`**: Arquivo de dados em formato Parquet que deve estar presente na mesma pasta do `dashboard.py`.

## Requisitos

- Python 3.7 ou superior
- [Dash](https://dash.plotly.com/)
- [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/)
- [Plotly](https://plotly.com/python/)
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)

## Instalação

1. **Clone o repositório:**

   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DO_DIRETORIO>
2. **Instale as dependencias**
   ```bash
    pip install dash dash-bootstrap-components plotly pandas numpy
3. **Execute o Dashboard**
   ```bash
    python dashboard.py
4. **Abra o navegador e acesse http://127.0.0.1:8050/ para visualizar o dashboard.**
