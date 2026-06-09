import joblib
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from pingouin import ttest
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, auc, roc_curve, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# 1: Ler dados
# Não há dados nulos
df_pulsar = pd.read_csv("./dataset/pulsar.csv")
print(df_pulsar.head(3))
print(df_pulsar.info())
print(df_pulsar.describe())

# 2: Análise Exploratória dos Dados (EDA)
df_pulsar_eda = df_pulsar.copy()
## Análise da target
# Estrelas de nêutron ≃ 9.15% da amostra
# |-- Por ser pouco, o excesso de falso positivos é ruim.
# |-- É interessante ter uma precisão bem alta.
porcentagem = df_pulsar_eda['target_class'].value_counts(normalize=True)
print(f"Porcentagem de estrelas de nêutron:\n{porcentagem}")
sns.barplot(
  x=porcentagem.index,
  y=porcentagem.values,
  color='purple',
  width=0.2,
)
plt.title("Porcentagem de Estrelas de Nêutrons")
plt.xlabel("0 = Estrela de Nêutron | 1 = Não é Estrela de Nêutron")
plt.ylabel("Porcentagem")
plt.grid(visible=True, alpha=.6)
plt.savefig("./dataviz/porcentagem-estrelas-barplot.png")
plt.close()

## Mean of the integrated profile
# |-- Shift à esquerda para variável target.
# |-- Estrelas de Nêutron tendem a ter média menor.
# |-- Problema: distribuição uniforme para dados de estrelas de nêutron.
# |-- Caso não seja essa estrela, a distribuição é normal.
sns.histplot(
  data=df_pulsar_eda,
  x='Mean of the integrated profile',
  hue='target_class',
  palette='coolwarm',
)
plt.ylabel('Frequência')
plt.grid(visible=True, alpha=.6)
plt.xlabel('Mean of the integrated profile')
plt.title("Mean of the integrated profile X Target Class - Histogram")
plt.savefig("./dataviz/mean-of-integrated-profile-x-target-histplot.png")
plt.close()

sns.boxplot(
  data=df_pulsar_eda,
  y='Mean of the integrated profile',
  hue='target_class',
  palette='coolwarm'
)
plt.ylabel('Média')
plt.grid(visible=True, alpha=.6)
plt.xlabel('0 - Não é estrela de nêutron | 1 - estrela de nêutron')
plt.title("Mean of the integrated profile X Target Class - Boxplot")
plt.savefig("./dataviz/mean-of-integrated-profile-x-target-boxplot.png")
plt.close()

# H0: os 2 grupos não apresentam uma diferença média significativa
# H1: os 2 grupos apresentam uma diferença média significativa
# p_value >= 0.05 --> Não rejeita H0
# |-- p_val ≃ 0, logo rejeita H0 e aponta evidência de significatividade da variável
grupo_positivo = df_pulsar_eda[(df_pulsar_eda['target_class'] == 1)]['Mean of the integrated profile']
grupo_negativo = df_pulsar_eda[(df_pulsar_eda['target_class'] == 0)]['Mean of the integrated profile']
result = ttest(x=grupo_positivo, y=grupo_negativo)
p_value = result['p_val'].iloc[0]
print(f'Student Test Mean of the integrated profile\np-value: {p_value:.6f}')

## Standard deviation of the integrated profile
# |-- Desvio padrão tende a ser menor em estrelas
# |-- Contudo essa métrica não é tão forte quanto a média.
sns.histplot(
  data=df_pulsar_eda,
  x='Standard deviation of the integrated profile',
  hue='target_class',
  palette='coolwarm',
)
plt.ylabel('Frequência')
plt.grid(visible=True, alpha=.6)
plt.xlabel('Standard deviation of the integrated profile')
plt.title("Standard deviation of the integrated profile X Target Class - Histogram")
plt.savefig("./dataviz/std-of-integrated-profile-x-target-histplot.png")
plt.close()

sns.boxplot(
  data=df_pulsar_eda,
  y='Standard deviation of the integrated profile',
  hue='target_class',
  palette='coolwarm'
)
plt.ylabel('Desvio Padrão')
plt.grid(visible=True, alpha=.6)
plt.xlabel('0 - Não é estrela de nêutron | 1 - estrela de nêutron')
plt.title("Standard deviation of the integrated profile X Target Class - Boxplot")
plt.savefig("./dataviz/std-of-integrated-profile-x-target-boxplot.png")
plt.close()

# H0: os 2 grupos não apresentam uma diferença média significativa
# H1: os 2 grupos apresentam uma diferença média significativa
# p_value >= 0.05 --> Não rejeita H0
# |-- p_val ≃ 0, logo rejeita H0 e aponta evidência de significatividade da variável.
# |-- Contudo, sabe-se pela análise gráfica que a variável não é tão expressiva assim
# |-- para o modelo como a média.
grupo_positivo = df_pulsar_eda[(df_pulsar_eda['target_class'] == 1)]['Standard deviation of the integrated profile']
grupo_negativo = df_pulsar_eda[(df_pulsar_eda['target_class'] == 0)]['Standard deviation of the integrated profile']
result = ttest(x=grupo_positivo, y=grupo_negativo)
p_value = result['p_val'].iloc[0]
print(f'Student Test Standard deviation of the integrated profile\np-value: {p_value:.6f}')

## Excess kurtosis of the integrated profile
# |-- Variável impacta fortemente
# |-- 0: Histograma com distribuição normal com foco extremo em um ponto
# |-- 1: Histograma com distribuição uniforme com mediana muito maior que do cenário acima
sns.histplot(
  data=df_pulsar_eda,
  x='Excess kurtosis of the integrated profile',
  hue='target_class',
  palette='coolwarm',
)
plt.ylabel('Frequência')
plt.grid(visible=True, alpha=.6)
plt.xlabel('Excess kurtosis of the integrated profile')
plt.title("Excess kurtosis of the integrated profile X Target Class - Histogram")
plt.savefig("./dataviz/kurtosis-of-integrated-profile-x-target-histplot.png")
plt.close()

sns.boxplot(
  data=df_pulsar_eda,
  y='Excess kurtosis of the integrated profile',
  hue='target_class',
  palette='coolwarm'
)
plt.ylabel('Kurtosis')
plt.grid(visible=True, alpha=.6)
plt.xlabel('0 - Não é estrela de nêutron | 1 - estrela de nêutron')
plt.title("Excess kurtosis of the integrated profile X Target Class - Boxplot")
plt.savefig("./dataviz/kurtosis-of-integrated-profile-x-target-boxplot.png")
plt.close()

# H0: os 2 grupos não apresentam uma diferença média significativa
# H1: os 2 grupos apresentam uma diferença média significativa
# p_value >= 0.05 --> Não rejeita H0
# |-- p_val ≃ 0, logo rejeita H0 e aponta evidência de significatividade da variável.
grupo_positivo = df_pulsar_eda[(df_pulsar_eda['target_class'] == 1)]['Excess kurtosis of the integrated profile']
grupo_negativo = df_pulsar_eda[(df_pulsar_eda['target_class'] == 0)]['Excess kurtosis of the integrated profile']
result = ttest(x=grupo_positivo, y=grupo_negativo)
p_value = result['p_val'].iloc[0]
print(f'Student Test Excess kurtosis of the integrated profile\np-value: {p_value:.6f}')

## Skewness of the integrated profile
# |-- Quase explica a classificação.
# |-- 0: Histograma com distribuição normal com foco extremo em um ponto de mediana baixa.
# |-- 1: Histograma com distribuição uniforme com mediana bem maior que do cenário acima.
sns.histplot(
  data=df_pulsar_eda,
  x='Skewness of the integrated profile',
  hue='target_class',
  palette='coolwarm',
)
plt.ylabel('Frequência')
plt.grid(visible=True, alpha=.6)
plt.xlabel('Skewness of the integrated profile')
plt.title("Skewness of the integrated profile X Target Class - Histogram")
plt.savefig("./dataviz/skewness-of-integrated-profile-x-target-histplot.png")
plt.close()

sns.boxplot(
  data=df_pulsar_eda,
  y='Skewness of the integrated profile',
  hue='target_class',
  palette='coolwarm'
)
plt.ylabel('Skewness')
plt.grid(visible=True, alpha=.6)
plt.xlabel('0 - Não é estrela de nêutron | 1 - estrela de nêutron')
plt.title("Skewness of the integrated profile X Target Class - Boxplot")
plt.savefig("./dataviz/skewness-of-integrated-profile-x-target-boxplot.png")
plt.close()

# H0: os 2 grupos não apresentam uma diferença média significativa
# H1: os 2 grupos apresentam uma diferença média significativa
# p_value >= 0.05 --> Não rejeita H0
# |-- p_val ≃ 0, logo rejeita H0 e aponta evidência de significatividade da variável.
grupo_positivo = df_pulsar_eda[(df_pulsar_eda['target_class'] == 1)]['Skewness of the integrated profile']
grupo_negativo = df_pulsar_eda[(df_pulsar_eda['target_class'] == 0)]['Skewness of the integrated profile']
result = ttest(x=grupo_positivo, y=grupo_negativo)
p_value = result['p_val'].iloc[0]
print(f'Student Test Skewness of the integrated profile\np-value: {p_value:.6f}')

## Mean of the DM-SNR curve
# |-- Não é uma métrica que pode explicar o modelo fortemente.
# | 0 e 1 possuem amplitudes parecidas, logo os dados podem se confunfir no classificador.
# |-- Apesar do p_valor do Student Test apontar 0, analiticamente sabe-se que esta
# |-- variável não é a mais relevante.
sns.histplot(
  data=df_pulsar_eda,
  x='Mean of the DM-SNR curve',
  hue='target_class',
  palette='coolwarm',
)
plt.ylabel('Frequência')
plt.grid(visible=True, alpha=.6)
plt.xlabel('Mean of the DM-SNR curve')
plt.title("Mean of the DM-SNR curve X Target Class - Histogram")
plt.savefig("./dataviz/dm-snr-curve-mean-x-target-histplot.png")
plt.close()

sns.boxplot(
  data=df_pulsar_eda,
  y='Mean of the DM-SNR curve',
  hue='target_class',
  palette='coolwarm'
)
plt.ylabel('DM-SNR mean')
plt.grid(visible=True, alpha=.6)
plt.xlabel('0 - Não é estrela de nêutron | 1 - estrela de nêutron')
plt.title("Mean of the DM-SNR curve X Target Class - Boxplot")
plt.savefig("./dataviz/dm-snr-curve-mean-x-target-boxplot.png")
plt.close()

# H0: os 2 grupos não apresentam uma diferença média significativa
# H1: os 2 grupos apresentam uma diferença média significativa
# p_value >= 0.05 --> Não rejeita H0
# |-- p_val ≃ 0, logo rejeita H0 e aponta evidência de significatividade da variável.
grupo_positivo = df_pulsar_eda[(df_pulsar_eda['target_class'] == 1)]['Mean of the DM-SNR curve']
grupo_negativo = df_pulsar_eda[(df_pulsar_eda['target_class'] == 0)]['Mean of the DM-SNR curve']
result = ttest(x=grupo_positivo, y=grupo_negativo)
p_value = result['p_val'].iloc[0]
print(f'Student Test Mean of the DM-SNR curve\np-value: {p_value:.6f}')

## Matriz de correlação
# A correlação de Pearson mostra uma forte correlação linear
# |-- entre as variáveis que não são de SNR e a variável target.
# Contudo as variáveis independente formam correlações fortes entre
# |-- em muitos casos.
pearson_corr = df_pulsar_eda.corr("pearson")
spearman_corr = df_pulsar_eda.corr("spearman")
plt.figure(figsize=(8,8))
sns.heatmap(
  data=pearson_corr,
  vmin=-1,
  vmax=1,
  cmap="coolwarm",
  annot=True
)
plt.title("Matriz de Correlação de Pearson")
plt.savefig("./dataviz/pearson-correlation.png")
plt.close()

## Treinamento do Modelo Baseline
X = df_pulsar.drop(columns=['target_class'])
y = df_pulsar['target_class']

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  shuffle=True,
  random_state=51,
  train_size=0.8,
  # stratify=y -> teoricamente melhora por balancear, mas aqui não melhorou.
)

colunas_numericas = [
  'Mean of the integrated profile',
  'Standard deviation of the integrated profile',
  'Excess kurtosis of the integrated profile',
  'Skewness of the integrated profile',
  'Mean of the DM-SNR curve',
  'Standard deviation of the DM-SNR curve',
  'Excess kurtosis of the DM-SNR curve',
  'Skewness of the DM-SNR curve',
]

transformer_numericas = Pipeline(steps=[
  ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
  ('num_transformer', transformer_numericas, colunas_numericas)
])

model = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('lr_model', LogisticRegression(solver='liblinear'))
])

model.fit(X=X_train, y=y_train)
y_pred = model.predict(X=X_test)
y_proba = model.predict_proba(X=X_test)
y_results = model.decision_function(X=X_test)

## Métricas do Modelo Baseline
# AUC ≃ 0.978571
# precisão ≃ 0.94
# acurácia ≃ 0.98
# recall ≃ 0.81
# f1-score: ≃ 0.930955
# Log-loss 0.795377
# O modelo é bom mas pode ser melhorado.
# |-- A precisão está alta, o que é ótimo, já que não classifica outros corpos
#     |-- como estrelas de nêutrons
# |-- Contudo, o recall é alto ainda, logo há estrelas de nêutrons sendo classificadas
#     |-- como outros corpos, o que é ruim, já que elas são poucas.
print("\nMétricas:")
fpr, tpr, thresholds= roc_curve(y_true=y_test, y_score=y_results)
area_under_curve = auc(x=fpr, y=tpr)
print(f'Area under curve (AUC): {area_under_curve:.6f}')
sns.lineplot(
  x=fpr,
  y=tpr,
  color='blue',
  label='Curva ROC'
)
sns.lineplot(
  x=fpr,
  y=fpr,
  linestyle='--',
  color='red',
  label='Classificador Aleatório'
)
plt.fill_between(
  x=fpr,
  y1=tpr, # type: ignore
  color='blue',
  alpha=0.2,
  label='Área sob a curva ROC'
)
plt.grid(visible=True)
plt.title("Curva ROC")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.savefig("./dataviz/modelo/modelo-baseline-curva-roc.png")
plt.close()

f1score = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
print(f"F1-Score: {f1score:.6f}")

classification_report_str = classification_report(
  y_pred=y_pred,
  y_true=y_test,
)
print(f"classification_report:\n{classification_report_str}")

log_loss_baseline = log_loss(y_pred=y_pred, y_true=y_test)
print(f"Log Loss - Baseline: {log_loss_baseline:.6f}")

conf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test)
disp_conf_matrix = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp_conf_matrix.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("./dataviz/modelo/modelo-baseline-matriz-de-confusao.png")
plt.close()

# Tuning de hiperparâmetros
# |--> tentar maximizar o recall ou o f1-score.
# |-- Mexendo em C-Value e Penalty
# Nenhuma otimizaçãoa aumenta a área mais que já está
# |-- quando selecionamos a área como prioridade.
# Quando escolhemos o f1-score como prioridade
# |-- atingimos f1-score e log loss piores, mas uma área maior: 0.98075
# Modelo baseline ainda é melhjor
def otimizacao_com_optuna(trial):
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2']) # sugere estes da lista
    c_value = trial.suggest_categorical('c', [100, 10, 1, 0.1, 0.01]) # sugere estes da lista
    
    model_optuna = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(
            solver='liblinear',
            C=c_value,
            penalty=penalty
            )
        )
    ])

    model_optuna.fit(X=X_train, y=y_train)

    y_decision_optuna = model_optuna.decision_function(X=X_test)
    y_pred_optuna = model_optuna.predict(X=X_test)

    fpr_optuna, tpr_optuna, _ = roc_curve(y_true=y_test, y_score=y_decision_optuna)
    area_opt = auc(x=fpr_optuna, y=tpr_optuna)
    f1_opt = f1_score(y_pred=y_pred_optuna, y_true=y_test, average='macro')
    log_loss_opt = log_loss(y_pred=y_pred_optuna, y_true=y_test)

    return area_opt, f1_opt, log_loss_opt

search_space = {
    'penalty': ['l1', 'l2'],
    'c': [100, 10, 1, 0.1, 0.01]
}
sampler = optuna.samplers.GridSampler(search_space=search_space)
estudo_lr = optuna.create_study(sampler=sampler, directions=[
    'maximize', 'maximize', 'minimize'
])
estudo_lr.optimize(
  otimizacao_com_optuna, # type: ignore
  n_trials=30
)

melhor_trial = max(
  estudo_lr.best_trials,
  key= lambda t: t.values[1]
)

print("Trial com melhor área sob curva e f1-score, e menor log_loss")
print(f"\tIteração melhor trial: {melhor_trial.number}")
print(f"\tMelhores params: {melhor_trial.params}")
print(f"\tMelhores valores: {melhor_trial.values}")

## Tuning de Hiperparâmetros - Controlando o número de variáveis.
# |-- Usar Select KBest com teste f_classifier pois há valores negativos contínuos
#     |-- teste qui-quadrado não aceita valores negativos.
# Resultado:
# |-- Quando há somente 5 variáveis o modelo é melhor.
# |-- A área sob a curva cresce: 9827238031972081,
# |-- O f1-score 'average' decresce muito pouco: 0.9299804121180828
# |-- O log-loss aumenta muito pouco: 0.8054447684718918
def optimizacao_melhores_variaveis(trial):
    k = trial.suggest_int('k', 1, 8)

    model_optuna = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature-selection', SelectKBest(score_func=f_classif, k = k)),
        ('model', LogisticRegression(solver='liblinear'))
    ])

    model_optuna.fit(X=X_train, y=y_train)
    y_pred_optuna = model_optuna.predict(X=X_test)
    y_results_optuna = model_optuna.decision_function(X=X_test)

    f1score_optuna = f1_score(y_pred=y_pred_optuna, y_true=y_test, average='macro')

    fpr_opt, tpr_opt, _ = roc_curve(y_score=y_results_optuna, y_true=y_test)
    area_opt = auc(x=fpr_opt, y=tpr_opt)

    log_loss_opt = log_loss(y_pred=y_pred_optuna, y_true=y_test)

    return k, area_opt, f1score_optuna, log_loss_opt

search_space = {
    'k': range(1, 9),
}
sampler = optuna.samplers.GridSampler(search_space=search_space)
estudo_opt = optuna.create_study(
  sampler=sampler,
  directions=['minimize', 'maximize', 'maximize', 'minimize'],
)
estudo_opt.optimize(
  optimizacao_melhores_variaveis, # type: ignore
  show_progress_bar=True
)

melhor_trial = max(
  estudo_opt.best_trials,
  key= lambda t: t.values[1]
)

print("Trial com menor k, melhor área sob curva e f1-score, e menor log_loss")
print(f"\tIteração melhor trial: {melhor_trial.number}")
print(f"\tMelhores params: {melhor_trial.params}")
print(f"\tMelhores valores: {melhor_trial.values}")

## Salvar modelo
## Apesar do ganho de explicação do modelo com somente 5 variáveis
## |-- Irei manter o modelo baseline pois ele acerta ligeiramente mais.
## |-- Porém seria bem válido usar o modelo obtido pelo Optuna, já que as classes
## |-- ficam melhor explicadas
joblib.dump(model, "./model_lr.pkl")