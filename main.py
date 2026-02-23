print("Le marché des processeurs graphiques (GPU) a connu de véritables montagnes russes au niveau des prix, particulièrement avec l'essor du minage de cryptomonnaies et les perturbations de la chaîne d'approvisionnement mondiale. Ce notebook explore en profondeur l'historique des prix des GPU NVIDIA et AMD, dans le but de révéler des tendances et des schémas susceptibles d'éclairer les futures décisions d'achat ou les prévisions du marché.\n")
print("Chargement des données \n Commençons par charger les fichiers de données contenant l'historique des prix des GPU ainsi que les métadonnées.\n")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Load the datasets
price_data = pd.read_csv('gpu_price_history.csv')
metadata = pd.read_csv('gpu_metadata.csv')

# Display the first few rows of each dataset
price_data.head(), metadata.head()


print("Nettoyage et Prétraitement des Données \n Avant de plonger dans l'analyse, il est crucial de nettoyer et de prétraiter les données. Cela inclut la gestion des valeurs manquantes, la conversion des types de données et la fusion des jeux de données si nécessaire.\n")

# Convert 'Date' to datetime format
price_data['Date'] = pd.to_datetime(price_data['Date'], errors='coerce')

# Check for missing values
price_data.isnull().sum(), metadata.isnull().sum()

# Drop rows with missing 'Date' values
price_data.dropna(subset=['Date'], inplace=True)

# Merge datasets on 'Name'
merged_data = pd.merge(price_data, metadata, on='Name', how='inner')

# Display the first few rows of the merged dataset
merged_data.head()


print("Analyse Exploratoire des Données (EDA) \n Explorons les données pour révéler les tendances et les schémas d'évolution des prix des GPU au fil du temps.\n")

# Plotting the distribution of retail and used prices
sns.histplot(merged_data['Retail Price'], kde=True, color='blue', label='Retail Price')
sns.histplot(merged_data['Used Price'], kde=True, color='red', label='Used Price')
plt.legend()
plt.title('Distribution of Retail and Used Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap for numeric features
numeric_df = merged_data.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


print("Modélisation Prédictive \n À partir des données historiques, tentons de prédire le prix d'occasion des GPU en fonction de leur prix de vente conseillé (MSRP) et d'autres caractéristiques.\n")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define features and target variable
X = merged_data[['Retail Price', '3DMARK']]
y = merged_data['Used Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse, r2



