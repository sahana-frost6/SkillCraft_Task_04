import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset (Seaborn has it built-in for easy practice)
df = sns.load_dataset('titanic')

# 2. Basic Inspection
print(df.info())
print(df.describe())

# 3. Pattern Identification: Gender vs Survival
# This is usually the strongest trend in the data
plt.figure(figsize=(8, 5))
sns.barplot(x='sex', y='survived', data=df, palette='magma')
plt.title('Survival Rate by Gender')
plt.ylabel('Proportion Survived')
plt.show()

# 4. Pattern Identification: Class vs Survival
# Did the "rich" survive more? (Socio-economic trend)
plt.figure(figsize=(8, 5))
sns.pointplot(x='pclass', y='survived', hue='sex', data=df)
plt.title('Survival Rate by Class and Gender')
plt.show()

# 5. Trend Analysis: Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='survived', kde=True, element="step", palette='husl')
plt.title('Age Distribution of Survivors vs Non-Survivors')
plt.show()

# 6. Relationship Analysis: Heatmap of Correlations
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='RdBu', fmt=".2f")
plt.title('Correlation Heatmap of Titanic Variables')
plt.show()