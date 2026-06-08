# Regressão Logística - CLassificador para Estrelas de Nêutrons Pulsares

![Python](https://img.shields.io/badge/Python-3.11.0-orange?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-orange?style=for-the-badge&logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-orange?style=for-the-badge)
![Pydantic](https://img.shields.io/badge/Pydantic-orange?style=for-the-badge&logo=pydantic&logoColor=white)
![TVM](https://img.shields.io/badge/TVM-orange?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-orange?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-orange?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-orange?style=for-the-badge&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-orange?style=for-the-badge)
![Scikit--Learn](https://img.shields.io/badge/scikit--learn-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-orange?style=for-the-badge)
![Joblib](https://img.shields.io/badge/Joblib-orange?style=for-the-badge)
![Pingouin](https://img.shields.io/badge/Pingouin-orange?style=for-the-badge)

## Resumo
Um classificador binário de estrelas de nêutrons pulsares reais a partir de observações astronômicas captadas por radiotelescópios. O contexto envolve a aplicação de técnicas de classificação para diferenciar pulsares de outras fontes astrofísicas ou ruído, utilizando dados estatísticos extraídos de sinais de rádio.

## Bibliotecas utilizadas
1. Python
2. FastAPI
3. Uvicorn
4. Pydantic
5. TVM
6. NumPy
7. Pandas
8. Matplotlib
9. Seaborn
10. Scikit-learn
11. Optuna
12. Joblib
13. Pingouin

## Como rodar
### Construção do modelo
```bash
pipenv sync
pipenv shell
python modelo.py
```
### API
Testar na porta 8080 do postman na rota /classify
```bash
pipenv sync
pipenv shell
uvicorn api:app --host 0.0.0.0 --port 8080
```
### API com Docker
Testar na porta 3333 do postman na rota /classify
```bash
docker build -t modeloqualidadefrutas .
docker run -d --name mymodelapp -p 3333:8080 modeloqualidadefrutas
```
### Exemplo de body para a requisição
```bash
{
    "A_id": 4,
    "Size": 1.36421682,
    "Weight": -1.296611877,
    "Sweetness": -0.384658206,
    "Crunchiness": -0.55300577,
    "Juiciness": 3.030874354,
    "Ripeness": -1.303849429,
    "Acidity": 0.501984036
}
```

## Análise Exploratória dos Dados (EDA)
### Variável Target
![Qualidade](./dataviz/quality-barplot.png)
A distribuição da variável target é praticamente uniforme. Sabendo disso, classificar ambos valores binários é importante, logo a métrica escolhida será a ***'F1-Score'***.

### Análises Bivariadas
![Qualidade](./dataviz/pairplot.png)

O pairplot revela que, com exceção da target, os dados seguem uma distribuição normal. Além disso, as variáveis não mostram uma relação clara entre si.

### Nova métrica estudada de análise da variável para ser usada? T-Student Test
A métrica **T-Student** está sendo usada para dizer se uma variável tem uma diferença de média significativa entre sua classificação binária para alguma variável.
Assume a seguinte Hipótese nula(H0):
+ H0: Os 2 grupos não apresentam uma diferença média significativa.
+ H1: Os 2 grupos apresentam uma diferença média significativa.
> A hipótese nula não é rejeitada caso o p-value do teste assuma um valor maior que *0.05*
> p-value >= 0.05 -> Não rejeita H0

Ex: divide os dados de peso da fruta a partir da classificação de qualidade. Se a amostra *'good'* e a mostra *'bad'* tiverem uma diferença média não significativa, há uma evidência(não é uma prova) para não usar essa variável como uma variável significativa da amostra.

### Peso
![Peso](./dataviz/weight-by-quality-boxplot.png)

O peso não aparenta ser uma variável muito impactante para determinar a variável dependente. A média de ambas situações de qualidade é bem próxima, o que muda é a amplitude dos dados. As frutas nos extremos tendem a ser frutas boas nessa amostra.<br/>
O teste T-Student revelou um p-value de 0.92, logo a hipótese nula não é aceita. Isso mostra que o Peso realmente apresenta evidências de não ser bom por si só para o modelo de classificação.

### Doçura
![Doçura](./dataviz/sweetness-by-quality-boxplot.png)

A doçura aparenta ser uma variável impactante para determinar a variável dependente. A média de ambas situações de qualidade é diferente. Além disso, as frutas boas tendem a ser deslocadas pra cima, ou seja, são mais doces<br/>
O teste T-Student revelou um p-value de ≃ 0, logo a hipótese nula é rejeitada. Isso mostra que a doçura realmente apresenta evidências de ser boa para o modelo de classificação.

### Tamanho
![Tamanho](./dataviz/size-by-quality-boxplot.png)

O tamanho aparenta ser uma variável impactante para determinar a variável dependente. A média de ambas situações de qualidade é diferente. Além disso, as frutas boas tendem a ser deslocadas pra cima, ou seja, são maiores<br/>
O teste T-Student revelou um p-value de ≃ 0, logo a hipótese nula é rejeitada. Isso mostra que o tamanho realmente apresenta evidências de ser bom para o modelo de classificação.

### Matriz de Correlação de Pearson
![Matriz de Correlação](./dataviz/matriz-correlacao.png)

Pela matriz de correlação, vê-se que nenhuma variável tem uma correlação forte com a qualidade, o que mostra uma dificuldade dos dados em explicar a classificação.

## Treinamento do modelo
O modelo uso o algoritmo de Regressão Logística clássico usando todas as variáveis. O modelo de separação de teste foi a separação aleatória dos dados em conjunto de treino(70% da amostra) e de teste(30% da amostra). Em seguida, tentou-se fazer tuning de hiperparâmetros com ***Optuna***, que foram:

1. C-value
2. penalty

Esses parâmetros buscaram nessa ordem:

1. Maximizar a **área sob a curva ROC**
2. Maximizar o **F1-Score**
3. Minimizar o **Log Loss**

O modelo com tuning de hiperparâmetros não resultou em uma melhora do modelo, logo o modelo que permaneceu foi o caso base.

## Métricas do Modelo
### Curva ROC e Área sob a curva
![Curva ROC](./dataviz/curva-roc-baseline.png)

|Área sob curva ROC|
|:-:|
|≃ 0.841457|

A área sob a curva foi de ≃ 0.8415 unidades de área(u.a). Um classificador aleatório que chuta aleatoriamente a classificação tem 50% de acerto, ou seja, tem uma área de 0.5, cuja curva representa a reta vermelha no gráfico. **Por ter uma curva acima dessa linha/ter uma área maior que essa linha, o modelo é melhor que um classificador aletório.**

### F1-Score, Accuracy e Precision
|F1-Score|Acuracy|Precision|
|:-:|:-:|:-:|
|≃ 0.778790|0.7775|≃ 0.78|

O F1-Score revela que o modelo não é excelente, mas é um modelo razoável. A acurácia confirma que o modelo está acertando 77.75% dos dados, o que mostra sua razoabilidade.

### Matriz de confusão
![Matriz de Confusão](./dataviz/modelo/matriz-de-confusao-baseline.png)

A matriz de confusão mostra esses dados, mas de forma numérica. É possível ver que há muitos falso positivos e falso negativos. O ideal seria encontrar um modelo que melhora isso, apesar de não estar ruim.

## Um ensaio de aumentar o threshold
O threshold usado no modelo é o padrão de 0.5. Ou seja, se houver a probabilidade >= 0.5 de ser 1, o modelo classifica como 1. Contudo, houve um estudo para mostrar o que haveria se o threshold fosse maior para somente aceitar aqueles que tem uma certeza maior de ser classiicado como 1.

![Ensaio - SUbindo Threshold](./dataviz/modelo/aumento-do-threshold-x-f1-score.png)

Como é possível ver, o F1-Score cai com o aumento do *'Threshold'*. Possivelmente o modelo toma decisões muitas vezes sem conseguir afirmar com muita certeza se a fruta é de qualidade *good'* ou *'bad '*.

O *Threshold* alto seria muito bom caso o modelo tivesse mais certeza das suas decisões. Exemplo: classificar uma transferência como fraude não é interessante se não estivermos muito certos de ser uma fraude. Com o threshold baixo, mesmo aceitando muito, talvez muitas frutas sejam enviadas sem estarem boas de fato, sobretudo quando olhamos para uma escala de milhões de frutas, o que pode gerar prejuízo.

## Melhorias para o modelo
1. Certamente seriam necessários mais registros.
2. Certamente é possível encontrar mais variáveis para essas frutas.
3. Especificidade: é melhor olhar para um tipo de fruta, que olhar de forma generalizada. Ex: É melhor fazer uma amostra de morango, outra de abacaxis, outra de bananas, etc, que fazerr uma análise de todas juntas.

## Créditos
Pedro Sodré, 8 de Junho de 2026