# 🧠 Roteiro de treinamento para o Exercício

## 1. Análise de Dados

### 1.1 Valores Vazios
- Verifique se existem dados faltantes na base.
- Possíveis soluções:
  - **Remover registros** → se for uma parcela pequena (~até 10%).
  - **Preencher valores numéricos** → média ou mediana (mais robusta a outliers).
  - **Preencher valores categóricos** → moda (valor mais frequente).

### 1.2 Valores Inconsistentes
- Exemplo:
  - Correto: `"Alto"`
  - Incorreto: `"Auto"`
- Corrija via scripts de padronização em Python, sem precisar editar manualmente cada caso.

### 1.3 Outliers
- Como identificar:
  - Gráficos: scatter, boxplot ou histograma.
  - Estatística: **Intervalo Interquartil (IQR)**.
- Observação: nem sempre outliers são erros; às vezes representam casos especiais (ex.: clientes VIP).

---

## 2. Pré-Processamento de Dados

### 2.1 Codificação de Variáveis Categóricas
- **LabelEncoder** → para variáveis categóricas simples.
- **OneHotEncoder** → variáveis **nominais** (sem ordem).
- **OrdinalEncoder** → variáveis **ordinais** (com ordem, ex.: Baixo < Médio < Alto).

O script abaixo, mostra como você deve utilizar cada codificador. Escolha um e aplique na sua modelagem: 

```Py
# ...
x_data = df.iloc[:, 0:23].values
y_data = df.iloc[:, 23].values

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer # Caso for utilizar o OneHotEncoder ou OrdinalEncoder

# ---------------------------------------------------------------
# 1. Label Encoder:
labelEncoder = LabelEncoder()

for i in range(x_data.shape[1]):
    x_data[:, i] = labelEncoder.fit_transform(x_data[:, i])

# ---------------------------------------------------------------
# 2. One Hot Encoder:
oneEncoder = OneHotEncoder()

oneTransform = ColumnTransformer(transformers=[('OneHot', oneEncoder, objects)], remainder='passthrough')

x_data = oneTransform.fit_transform(x_data)
# Observação: Dependendo do tamanho da sua base, o script pode retornar um erro. Isso ocorre, porque esse codificador criar várias colunas para suportar digitações binárias. Para prevenir isso, tente realizar uma redução da base, aplicando conceitos de amostragem da estatística-

# ---------------------------------------------------------------
# 3. Ordinal Encoder:
ordinalEncoder = OrdinalEncoder()

ordinalTransform = ColumnTransformer(transformers=[('Ordinal', ordinalEncoder, objects)], remainder='passthrough')

# ---------------------------------------------------------------
```

### 2.2 Escalonamento de Variáveis Numéricas
- **StandardScaler** → distribuições próximas da normal (média 0, desvio padrão 1).
- **MinMaxScaler** → distribuições não normais (normaliza para [0,1]).
- Observação: nem todos os modelos precisam de escalonamento (árvores de decisão e random forest geralmente não).

O script abaixo, mostra como você deve utilizar cada escalonador. Escolha um e aplique na sua modelagem: 

```Py
# ...
x_data = df.iloc[:, 0:23].values
y_data = df.iloc[:, 23].values

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ---------------------------------------------------------------
# 1. Standard Scaler:
scalerStandard = StandardScaler()

x_data = scalerStandard.fit_transform(x_data)

# ---------------------------------------------------------------
# 2. Min Max Scaler:
scalerMinMax = MinMaxScaler()

x_data = scalerMinMax.fit_transform(x_data)


```


### 2.3 Dados Desbalanceados
- **Oversampling** → aumenta a classe minoritária (ex.: SMOTE).
- **Undersampling** → reduz a classe majoritária.

O script abaixo, mostra como você deve utilizar cada técnica de tratar dados desbalanceados. Escolha um e aplique na sua modelagem: 

```Py
# ...
x_data = df.iloc[:, 0:23].values
y_data = df.iloc[:, 23].values

from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------
# 1. Oversampling:
smt = SMOTE(sampling_strategy = 'minority')

x_data, y_data = smt.fit_resample(x_data, y_data)

# 2. Undersampling:
tomek = TomekLinks(sampling_strategy = 'majority')

x_data, y_data = tomek.fit_resample(x_data, y_data)

```
---

## 3. Modelos de Machine Learning

Escolha de acordo com **tamanho da base, complexidade do problema e interpretabilidade desejada**.

### 3.1 Modelos Simples e Interpretáveis
- Regressão Logística;
- Naive Bayes;
- Árvores de Decisão;

```Py
# ...
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
```

### 3.2 Modelos Robustos (base média)
- Random Forest;
- Gradient Boosting (XGBoost, LightGBM, CatBoost) 

```Py
# ...
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
```

### 3.3 Modelos Complexos (grandes bases / alta dimensionalidade)
- SVM → bom para dados de média a alta complexidade; pesado em grandes bases.
- Redes Neurais → recomendadas para problemas complexos com grande volume de dados (imagens, texto, séries temporais).

```Py
# ...
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
```

---

<div align='center'>
    <a href="Exercicio.md">Exercício</a>
</div>
