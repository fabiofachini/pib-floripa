import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Carregar o arquivo CSV para o DataFrame
file_path = 'indicadores_tri2.csv'
df = pd.read_csv(file_path, decimal=",")

# Ajustar a coluna de índice de tempo, tratando possíveis formatos inconsistentes
df['Ano'] = pd.to_datetime(df['Ano'], format='%b/%y', errors='coerce')

# Verificar se há valores 'NaT' na coluna 'Ano' (isso indica que a conversão falhou para algumas datas)
print(df['Ano'].isna().sum(), "datas inválidas encontradas.")

# Eliminar as linhas com datas inválidas
df = df.dropna(subset=['Ano'])

# Definir as variáveis dependentes e independentes
variaveis_modelo = ['Passageiros', 'Consumo MWh', 'Emplacamentos', 'Exportações', 'Importações',
                    'ICV', 'Receitas Correntes', 'Venda Residencial', 
                    'Venda Comercial', 'Taxa Desemprego', 'PIB Corrente Fpolis']

# Garantir que não existam valores ausentes
df_modelo = df[variaveis_modelo].dropna()

# Definir X e y
X = df_modelo.drop(columns=['PIB Corrente Fpolis'])  # Supondo que 'PIB Corrente Fpolis' seja a variável alvo
y = df_modelo['PIB Corrente Fpolis']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modelo de Random Forest e LOOCV
loo = LeaveOneOut()
model = RandomForestRegressor()
mse_scores = []

for train_index, test_index in loo.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

# Média dos erros quadráticos médios
mean_mse = sum(mse_scores) / len(mse_scores)
print("Mean MSE (Random Forest LOOCV):", mean_mse)
