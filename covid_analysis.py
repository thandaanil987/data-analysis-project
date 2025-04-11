# COVID-19 Data Analysis using Pandas, NumPy, and Matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load dataset
url = "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv"
df = pd.read_csv(url)

# Step 2: Basic overview
print("Dataset shape:", df.shape)
print("Columns:", df.columns)
print(df.head())

# Step 3: Data preprocessing
df['Date'] = pd.to_datetime(df['Date'])
countries = ['India', 'United States', 'Brazil', 'Russia']
df = df[df['Country'].isin(countries)]

# Step 4: Total cases by country
total_cases = df.groupby('Country')['Confirmed'].max()
print("\nTotal Confirmed Cases by Country:\n", total_cases)

# Step 5: Plotting confirmed cases over time
plt.figure(figsize=(10,6))
for country in countries:
    country_data = df[df['Country'] == country]
    plt.plot(country_data['Date'], country_data['Confirmed'], label=country)

plt.title("COVID-19 Confirmed Cases Over Time")
plt.xlabel("Date")
plt.ylabel("Confirmed Cases")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Calculating death rate
df['Death Rate (%)'] = np.where(df['Confirmed'] > 0, (df['Deaths'] / df['Confirmed']) * 100, 0)

# Step 7: Average death rate by country
avg_death_rate = df.groupby('Country')['Death Rate (%)'].mean()
print("\nAverage Death Rate (%) by Country:\n", avg_death_rate.round(2))

# Step 8: Bar chart for average death rate
avg_death_rate.plot(kind='bar', color='orange', title='Average COVID-19 Death Rate by Country', figsize=(8,5))
plt.ylabel("Death Rate (%)")
plt.grid(axis='y')
plt.show()
