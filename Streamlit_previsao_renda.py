import streamlit as st
import io
import os

import pandas_profiling

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.tree import export_text

sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Projeto #02 | Previsão de renda",
     page_icon="https://e3ba6e8732e83984.cdn.gocache.net/uploads/image/file/530714/regular_ebac-logo.png",
    layout="wide",
    initial_sidebar_state="auto",
    )

st.write('# Análise exploratória da previsão de renda')

st.sidebar.markdown('''
<div style="text-align:center">
<img src="https://e3ba6e8732e83984.cdn.gocache.net/uploads/image/file/530714/regular_ebac-logo.png" alt="ebac-logo" width=50%>
</div>

# **Profissão: Cientista de Dados**
### [**Projeto #02** | Previsão de renda](https://github.com/ohallao/Projects/tree/main/Income_prediction)

**Por:** [Allan Figueiredo Santiago](https://www.linkedin.com/in/allan-santiago/)<br>
**Data:** 31 de outubro de 2023.<br>

---
''', unsafe_allow_html=True)

with st.sidebar.expander("Índice", expanded=False):
    st.markdown('''
    - [Etapa 1 CRISP - DM: Entendimento do negócio](#1)
    - [Etapa 2 Crisp-DM: Entendimento dos dados](#2)
        > - [Dicionário de dados](#dicionario)
        > - [Carregando os dados](#dados)
        > - [Entendimento dos dados - Univariada](#univariada)
        >> - [Estatísticas descritivas das variáveis quantitativas](#describe)
        > - [Entendimento dos dados - Bivariadas](#bivariada)
        >> - [Matriz de correlação](#correlacao)
        >> - [Matriz de dispersão](#dispersao)
        >>> - [Clustermap](#clustermap)
        >>> - [Linha de tendência](#tendencia)
        >> - [Análise das variáveis qualitativas](#qualitativas)             
    - [Etapa 3 Crisp-DM: Preparação dos dados](#3)
    - [Etapa 4 Crisp-DM: Modelagem](#4)
        > - [Divisão da base em treino e teste](#train_test)
        > - [Seleção do modelo com for loop](#for_loop)
        > - [Escolhendo o Modelo](#escolha)
    - [Etapa 5 Crisp-DM: Avaliação dos resultados](#5)
    - [Etapa 6 Crisp-DM: Implantação/Simulação](#6)
    ''', unsafe_allow_html=True)


with st.sidebar.expander("Bibliotecas/Pacotes", expanded=False):
    st.code('''
    import streamlit as st
    import io
    import os

    import pandas_profiling

    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt
    import seaborn as sns

    from streamlit_pandas_profiling import st_profile_report
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn import tree
    from sklearn.tree import plot_tree
    from sklearn.tree import export_text
    ''', language='python')

st.markdown('''
### Carregando os dados <a name="dados"></a>
''', unsafe_allow_html=True)


# Um botao import para qualquer csv file
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df)


st.markdown('''
## Etapa 1 CRISP - DM: Entendimento do negócio <a name="1"></a>
''', unsafe_allow_html=True)


st.markdown('''
Uma instituição financeira deseja entender melhor o perfil de renda de seus novos clientes para diferentes objetivos, como, por exemplo, dimensionar adequadamente o limite dos cartões de crédito sem necessariamente solicitar holerites ou outros documentos que possam comprometer a experiência do cliente.

Para isso, realizou um estudo com alguns clientes, comprovando suas rendas por meio de holerites e outros documentos. Agora, pretende construir um modelo preditivo dessa renda com base em algumas variáveis que já tem em seu banco de dados.
''')

st.markdown('''
## Etapa 2 Crisp-DM: Entendimento dos dados<a name="2"></a>
''', unsafe_allow_html=True)


st.markdown('''
### Dicionário de dados <a name="dicionario"></a>

| Variável              | Descrição                                                                                                  | Tipo             |
| --------------------- |:----------------------------------------------------------------------------------------------------------:| ----------------:|
| data_ref              | Data de referência de coleta das variáveis                                                                 | object           |
| id_cliente            | Código identificador exclusivo do cliente                                                                  | int              |
| sexo                  | Sexo do cliente                                                                                            | object (binária) |
| posse_de_veiculo      | Indica se o cliente possui veículo                                                                         | bool (binária)   |
| posse_de_imovel       | Indica se o cliente possui imóvel                                                                          | bool (binária)   |
| qtd_filhos            | Quantidade de filhos do cliente                                                                            | int              |
| tipo_renda            | Tipo de renda do cliente (Empresário, Assalariado, Servidor público, Pensionista, Bolsista)                | object           |
| educacao              | Grau de instrução do cliente (Primário, Secundário, Superior incompleto, Superior completo, Pós graduação) | object           |
| estado_civil          | Estado civil do cliente                                                                                    | object           |
| tipo_residencia       | Tipo de residência do cliente                                                                              | object           |
| idade                 | Idade do cliente em anos                                                                                   | int              |
| tempo_emprego         | Tempo no emprego atual                                                                                     | float            |
| qt_pessoas_residencia | Quantidade de pessoas que moram na residência                                                              | float            |
| **renda**             | Valor numérico decimal representando a renda do cliente em reais                                           | float            |
''', unsafe_allow_html=True)


# VERIFICAR ARQUIVOS LOCAIS:
# path_to_find = os.listdir()
# st.title(path_to_find)
renda = df

buffer = io.StringIO()
renda.info(buf=buffer)
st.text(buffer.getvalue())
st.dataframe(renda)


st.table(renda.nunique()
              .to_frame()
              .reset_index()
              .rename(columns={'index': 'Variável',
                               0: 'Valores únicos'}))

renda.drop(columns=['Unnamed: 0', 'id_cliente'], inplace=True)
st.write('Quantidade total de linhas:',
         len(renda))
st.write('Quantidade de linhas duplicadas:',
         renda.duplicated().sum())
st.write('Quantidade após remoção das linhas duplicadas:',
         len(renda.drop_duplicates()))
renda.drop_duplicates(inplace=True, ignore_index=True)
buffer = io.StringIO()
renda.info(buf=buffer)
st.text(buffer.getvalue())

st.markdown('''
### Entendimento dos dados - Univariada <a name="univariada"></a>
''', unsafe_allow_html=True)

with st.expander("Pandas Profiling – Relatório interativo para análise exploratória de dados", expanded=True):
    pr = renda.profile_report()                   
    # st.components.v1.html(prof.to_html(), height=600, scrolling=True)
    st_profile_report(pr)
st.markdown('''
####  Estatísticas descritivas das variáveis quantitativas <a name="describe"></a>
''', unsafe_allow_html=True)


st.write(renda.describe().transpose())


st.markdown('''
### Entendimento dos dados - Bivariadas <a name="bivariada"></a>
''', unsafe_allow_html=True)


st.markdown('''
#### Matriz de correlação <a name="correlacao"></a>
''', unsafe_allow_html=True)

st.write((renda
          .iloc[:, 3:]
          .corr()
          .tail(n=1)
          ))


st.markdown('A partir da matriz de correlação, é possível observar que a variável que apresenta maior relação com a varíavel `renda` é `tempo_emprego`, com um índice de correlação de 38,5%.')


st.markdown('''
#### Matriz de dispersão <a name="dispersao"></a>
''', unsafe_allow_html=True)


sns.pairplot(data=renda,
             hue='tipo_renda',
             vars=['qtd_filhos',
                   'idade',
                   'tempo_emprego',
                   'qt_pessoas_residencia',
                   'renda'],
             diag_kind='hist')
st.pyplot(plt)


st.markdown('Ao analisar o *pairplot*, que consiste na matriz de dispersão, é possível identificar alguns *outliers* na variável `renda`, os quais podem afetar o resultado da análise de tendência, apesar de ocorrerem com baixa frequência. Além disso, é observada uma baixa correlação entre praticamente todas as variáveis quantitativas, reforçando os resultados obtidos na matriz de correlação.')


st.markdown('''
##### Clustermap <a name="clustermap"></a>
''', unsafe_allow_html=True)

fig, ax = plt.subplots()
sns.heatmap(renda.corr(), annot=True)
st.write(fig)

st.markdown('Utilizando o clustermap, observamos novamente uma correlação baixa com a variável renda. A variável tempo_emprego é a única que exibe um índice digno de análise. Adicionalmente, introduzimos duas variáveis booleanas: posse_de_imovel e posse_de_veiculo, que, no entanto, também demonstram uma correlação limitada com renda..')


st.markdown('''
#####  Linha de tendência <a name="tendencia"></a>
''', unsafe_allow_html=True)


plt.figure(figsize=(16, 9))
sns.scatterplot(x='tempo_emprego',
                y='renda',
                hue='tipo_renda',
                size='idade',
                data=renda,
                alpha=0.8)
sns.regplot(x='tempo_emprego',
            y='renda',
            data=renda,
            scatter=False,
            color='.2')
st.pyplot(plt)

st.markdown('A inclinação da linha de tendência revela uma covariância positiva, mesmo que a correlação entre as variáveis tempo_emprego e renda não seja tão alta.')


st.markdown('''
#### Análise das variáveis qualitativas <a name="qualitativas"></a>
''', unsafe_allow_html=True)


with st.expander("Análise de variáveis booleanas", expanded=True):
    plt.rc('figure', figsize=(12, 4))
    fig, axes = plt.subplots(nrows=1, ncols=1)
    sns.pointplot(x='posse_de_imovel',
                  y='renda',
                  data=renda,
                  dodge=True,
                  )
    line1 = plt.gca().get_lines()[0]
    y1 = line1.get_ydata().mean()
    plt.text(1.5, y1, 'posse_de_imovel', va="center", ha="right", color=line1.get_color())
    sns.pointplot(x='posse_de_veiculo',
                  y='renda',
                  data=renda,
                  dodge=True,
                  )
    line2 = plt.gca().get_lines()[0]
    y2 = line2.get_ydata().mean()+1000
    plt.text(1.5, y2, 'posse_de_veiculo', va="top", ha="right", color=line2.get_color())
    st.pyplot(plt)
st.markdown('Ao comparar os gráficos acima, nota-se que a variável `posse_de_veículo` apresenta maior relevância na predição de renda, evidenciada pela maior distância entre os intervalos de confiança para aqueles que possuem e não possuem veículo, ao contrário da variável `posse_de_imóvel` que não apresenta diferença significativa entre as possíveis condições de posse imobiliária.')

with st.expander("Gráfico de barras para avaliar a distribuição das variáveis qualitativas no tempo", expanded=True):
    renda['data_ref'] = pd.to_datetime(arg=renda['data_ref'])
    qualitativas = renda.select_dtypes(include=['object', 'boolean']).columns
    plt.rc('figure', figsize=(16, 4))
    for col in qualitativas:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.subplots_adjust(wspace=1.)
        tick_labels = renda['data_ref'].map(
            lambda x: x.strftime('%b/%Y')).unique()
        # barras empilhadas:
        renda_crosstab = pd.crosstab(index=renda['data_ref'],
                                     columns=renda[col],
                                     normalize='index')
        ax0 = renda_crosstab.plot.bar(stacked=True,
                                      ax=axes[0])
        ax0.set_xticklabels(labels=tick_labels, rotation=45)
        axes[0].legend(bbox_to_anchor=(1, .5), loc=6, title=f"'{col}'")
        # perfis médios no tempo:
        ax1 = sns.pointplot(x='data_ref', y='renda', hue=col,
                            data=renda, dodge=True, ci=95, ax=axes[1])
        ax1.set_xticklabels(labels=tick_labels, rotation=45)
        axes[1].legend(bbox_to_anchor=(1, .5), loc=6, title=f"'{col}'")
        st.pyplot(plt)
st.markdown('''
## Etapa 3 Crisp-DM: Preparação dos dados<a name="3"></a>
''', unsafe_allow_html=True)


renda.drop(columns='data_ref', inplace=True)
renda.dropna(inplace=True)
st.table(pd.DataFrame(index=renda.nunique().index,
                      data={'tipos_dados': renda.dtypes,
                            'qtd_valores': renda.notna().sum(),
                            'qtd_categorias': renda.nunique().values}))


with st.expander("Conversão das variáveis categóricas em variáveis numéricas (dummies)", expanded=True):
    renda_dummies = pd.get_dummies(data=renda)
    buffer = io.StringIO()
    renda_dummies.info(buf=buffer)
    st.text(buffer.getvalue())

    st.table((renda_dummies.corr()['renda']
              .sort_values(ascending=False)
              .to_frame()
              .reset_index()
              .rename(columns={'index': 'var',
                               'renda': 'corr'})
              .style.bar(color=['darkred', 'darkgreen'], align=0)
              ))

st.markdown('''
## Etapa 4 Crisp-DM: Modelagem <a name="4"></a>
''', unsafe_allow_html=True)


st.markdown('A técnica escolhida foi o DecisionTreeRegressor, devido à sua capacidade de lidar com problemas de regressão, como a previsão de renda dos clientes. Além disso, árvores de decisão são fáceis de interpretar e permitem a identificação dos atributos mais relevantes para a previsão da variável-alvo, tornando-a uma boa escolha para o projeto.')


st.markdown('''
### Divisão da base em treino e teste <a name="train_test"></a>
''', unsafe_allow_html=True)


X = renda_dummies.drop(columns='renda')
y = renda_dummies['renda']
st.write('Quantidade de linhas e colunas de X:', X.shape)
st.write('Quantidade de linhas de y:', len(y))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
st.write('X_train:', X_train.shape)
st.write('X_test:', X_test.shape)
st.write('y_train:', y_train.shape)
st.write('y_test:', y_test.shape)


st.markdown('''
### Seleção do modelo com for loop <a name="for_loop"></a>
''', unsafe_allow_html=True)

score_df = pd.DataFrame(columns=['max_depth', 'score'])
max_depth_range = range(1, 11)  # Considering trees of depth 1 to 10

for depth in max_depth_range:
    tree = DecisionTreeRegressor(max_depth=depth)
    tree.fit(X_train, y_train)

    score_value = tree.score(X_train, y_train)

    score_df = score_df.append({'max_depth': depth, 'score': score_value}, ignore_index=True)

st.dataframe(score_df)

st.markdown('''
### Escolhendo o modelo <a name="rodando"></a>
''', unsafe_allow_html=True)

tree1 = DecisionTreeRegressor(max_depth = 4)

st.write(tree1.fit(X_train, y_train))
st.text('''
        Realizou-se um teste com o DecisionTreeRegressor definindo max_depth=10. No entanto,
        não se observou uma variação significativa nos valores previstos de renda ao simular
        diferentes valores para as variáveis. Portanto, para otimizar o desempenho do script,
        optou-se por manter max_depth=4 mesmo o max_depth=10 tendo um R^2 superior.
        ''')

st.markdown('''
### Rodando o modelo <a name="rodando"></a>
''', unsafe_allow_html=True)

with st.expander("Visualização gráfica da árvore com plot_tree", expanded=True):
    plt.figure(figsize=(20, 10))
    plot_tree(decision_tree=tree1, feature_names=list(X.columns), filled=True)
    st.pyplot(plt)

st.markdown('''
## Etapa 5 Crisp-DM: Avaliação dos resultados <a name="5"></a>
''', unsafe_allow_html=True)

r2_train = round(tree1.score(X=X_train, y=y_train), 2)
r2_test = round(tree1.score(X=X_test, y=y_test), 2)

st.write(f"R² score para a base de treino: {r2_train:.2f}")
st.write(f"R² score para a base de teste: {r2_test:.2f}")

renda['renda_predict'] = np.round(tree1.predict(X), 2)
st.dataframe(renda[['renda', 'renda_predict']])

st.markdown('''
## Etapa 6 Crisp-DM: Implantação <a name="6"></a>
''', unsafe_allow_html=True)

st.write('''Nessa etapa colocamos em uso o modelo desenvolvido, 
         normalmente implementando o modelo desenvolvido em um motor 
         que toma as decisões com algum nível de automação.''')

# Criando input para as variaveis com widgets do Streamlit
sexo = st.radio("Informe o sexo:", ('M', 'F'))
posse_de_veiculo = st.checkbox("Possui veículo?")
posse_de_imovel = st.checkbox("Possui imóvel?")
qtd_filhos = st.number_input("Quantidade de filhos:", min_value=0, step=1, format='%d')
tipo_renda = st.selectbox("Tipo de renda:", ['Assalariado', 'Bolsista', 'Empresário', 'Pensionista', 'Servidor público'])
educacao = st.selectbox("Educação:", ['Primário', 'Pós graduação', 'Secundário', 'Superior completo', 'Superior incompleto'])
estado_civil = st.selectbox("Estado civil:", ['Casado', 'Separado', 'Solteiro', 'União', 'Viúvo'])
tipo_residencia = st.selectbox("Tipo de residência:", ['Aluguel', 'Casa', 'Com os pais', 'Comunitário', 'Estúdio', 'Governamental'])
idade = st.number_input("Idade:", min_value=0, step=1, format='%d')
tempo_emprego = st.text_input("Tempo de emprego (anos ou deixe vazio se não aplicável):")
tempo_emprego = None if tempo_emprego == '' else int(tempo_emprego)
qt_pessoas_residencia = st.number_input("Quantidade de pessoas na residência:", min_value=1, step=1, format='%d')

# Botão de previsão
if st.button('Prever Renda'):
    # Criando um Dataframe com input
    entrada = pd.DataFrame([{'sexo': sexo,
                             'posse_de_veiculo': posse_de_veiculo,
                             'posse_de_imovel': posse_de_imovel,
                             'qtd_filhos': qtd_filhos,
                             'tipo_renda': tipo_renda,
                             'educacao': educacao,
                             'estado_civil': estado_civil,
                             'tipo_residencia': tipo_residencia,
                             'idade': idade,
                             'tempo_emprego': tempo_emprego,
                             'qt_pessoas_residencia': qt_pessoas_residencia}])
    
    # Processando e prevendo a renda
    entrada = pd.concat([X, pd.get_dummies(entrada)]).fillna(value=0).tail(1)
    renda_estimada = tree1.predict(entrada).item()
    st.write(f"Renda estimada: R${str(np.round(renda_estimada, 2)).replace('.', ',')}")

