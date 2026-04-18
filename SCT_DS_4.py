import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load data - Using 'r' prefix to avoid the 'unicodeescape' error
path = r"C:\Users\DELL\Downloads\archive (2)\US_Accidents_March23.csv"
df = pd.read_csv(path)

# 2. Analyze Time of Day
# Use 'Start_Time' and convert to datetime objects
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['Hour'] = df['Start_Time'].dt.hour

plt.figure(figsize=(12, 6))
sns.countplot(x='Hour', data=df, palette='viridis')
plt.title('Accidents by Hour of Day')
plt.xlabel('Hour (0-23)')
plt.ylabel('Number of Accidents')
plt.show()

# 3. Analyze Weather Conditions
# Note: 'Weather_Condition' is the correct column name
# We take the top 10 conditions to keep the chart readable
top_weather = df['Weather_Condition'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_weather.values, y=top_weather.index, palette='magma')
plt.title('Top 10 Weather Conditions During Accidents')
plt.xlabel('Number of Accidents')
plt.ylabel('Weather Condition')
plt.show()

# Severity Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Severity', data=df, palette='Reds')
plt.title('Distribution of Accident Severity')
plt.xlabel('Severity Level (1=Low, 4=High)')
plt.ylabel('Number of Accidents')
plt.show()

# Accidents by State (Top 10)
top_states = df['State'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_states.index, y=top_states.values, palette='coolwarm')
plt.title('Top 10 States with Highest Number of Accidents')
plt.xlabel('State')
plt.ylabel('Number of Accidents')
plt.show()

# Selecting specific numerical columns for a cleaner heatmap
num_cols = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
correlation_matrix = df[num_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='mako', fmt=".2f")
plt.title('Correlation Heatmap of Weather Factors & Severity')
plt.show()

# List of road condition columns
road_features = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

# Calculate the sum of True values for each feature
feature_counts = df[road_features].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=feature_counts.values, y=feature_counts.index, palette='plasma')
# Identifying the pattern: Which road features are most dangerous?
plt.title('Accidents by Road Condition / Traffic Feature')
plt.xlabel('Number of Accidents')
plt.show()

# Sample the data if it's too large to plot (e.g., 100,000 points)
sample_df = df.sample(n=100000, random_state=42)

plt.figure(figsize=(12, 8))
# Use alpha=0.1 to see where points overlap (hotspots)
plt.scatter(sample_df['Start_Lng'], sample_df['Start_Lat'], alpha=0.05, s=1, color='red')
plt.title('Accident Hotspots across the US')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# Limit coordinates to mainland US for a cleaner view
plt.xlim(-125, -66)
plt.ylim(24, 50)
plt.show()

# Impact of Visibility on Severity
plt.figure(figsize=(10, 6))
sns.boxplot(x='Severity', y='Visibility(mi)', data=df[df['Visibility(mi)'] < 20])
plt.title('Impact of Visibility on Accident Severity')
plt.show()