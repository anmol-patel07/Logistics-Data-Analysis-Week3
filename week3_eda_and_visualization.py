import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
np.random.seed(42)
n_samples = 200

# ==============================================================================
# STEP 1: Simulate Logistics Dataset
# ==============================================================================
data = {
    'Shipment_ID': [f'LOG_{2000+i}' for i in range(n_samples)],
    'Distance_KM': np.random.uniform(5, 300, n_samples),
    'Shipment_Volume_m3': np.random.uniform(0.1, 5.0, n_samples),
    'Transport_Mode': np.random.choice(['Truck', 'Van', 'Express Bike'], n_samples, p=[0.4, 0.4, 0.2]),
    'Traffic_Level': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2])
}

df = pd.DataFrame(data)

# Derive dependent metrics
df['Transportation_Cost_USD'] = (df['Distance_KM'] * 0.8) + (df['Shipment_Volume_m3'] * 15) + np.random.normal(10, 5, n_samples)
df['Delivery_Time_Min'] = (df['Distance_KM'] * 1.2) + np.where(df['Traffic_Level'] == 'High', 45, 10) + np.random.normal(0, 10, n_samples)

print("--- Central Tendencies & Summary ---")
print(df[['Delivery_Time_Min', 'Transportation_Cost_USD', 'Shipment_Volume_m3']].describe().T[['mean', '50%', 'std']])

# ==============================================================================
# STEP 2: Generate Visualizations (Matplotlib & Seaborn)
# ==============================================================================
plt.figure(figsize=(14, 10))

# Plot 1: Delivery Time Distribution
plt.subplot(2, 2, 1)
sns.histplot(df['Delivery_Time_Min'], kde=True, color='teal')
plt.title('Distribution of Delivery Times (Minutes)')
plt.xlabel('Delivery Time (Min)')

# Plot 2: Transportation Cost by Transport Mode
plt.subplot(2, 2, 2)
sns.boxplot(x='Transport_Mode', y='Transportation_Cost_USD', data=df, palette='Set2')
plt.title('Transportation Cost by Fleet Vehicle Type')
plt.xlabel('Transport Mode')
plt.ylabel('Cost ($)')

# Plot 3: Distance vs Delivery Time with Traffic Overlay
plt.subplot(2, 2, 3)
sns.scatterplot(x='Distance_KM', y='Delivery_Time_Min', hue='Traffic_Level', data=df, palette='viridis')
plt.title('Impact of Distance and Traffic on Delivery Time')

# Plot 4: Correlation Heatmap
plt.subplot(2, 2, 4)
numeric_df = df[['Distance_KM', 'Shipment_Volume_m3', 'Transportation_Cost_USD', 'Delivery_Time_Min']]
sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title('Correlation Matrix of Key Metrics')

plt.tight_layout()
plt.savefig('logistics_visualizations.png')
print("Visualizations generated and saved as logistics_visualizations.png")
