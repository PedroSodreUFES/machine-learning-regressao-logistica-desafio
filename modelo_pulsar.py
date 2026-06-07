import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1: Ler dados
# Não há dados nulos
df_pulsar = pd.read_csv("./dataset/pulsar.csv")
print(df_pulsar.head(3))
print(df_pulsar.info())
print(df_pulsar.describe())

# 2: Análise Exploratória dos Dados (EDA)
sns.lineplot(
  x=list(range(-5,6)),
  y=[x*x*x for x in range(-5, 6)],
  color='#ff4000',
)
plt.savefig("./dataviz/test-image.png")
plt.close()