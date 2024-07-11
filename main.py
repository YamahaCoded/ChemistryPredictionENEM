import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./ENEM_data.csv')

years = np.array([2020, 2021, 2022, 2023]).reshape(-1, 1) 

plt.figure(figsize=(10, 6))

for index, row in df.iterrows():
    topic = row['Topic']
    topic_data = np.array(row[1:]).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(years, topic_data)
    prevision = model.predict(np.array([[2024]]))
    
    plt.scatter(topic, prevision, label=topic)

plt.title('Previsões das Questões de Química do ENEM em 2024')
plt.xlabel('Tópicos')
plt.ylabel('Questões')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()