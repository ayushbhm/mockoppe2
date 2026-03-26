import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Read iteration number from file
with open('iteration.txt', 'r') as f:
    iteration = int(f.read().strip())

# Augment data based on iteration
augmented = df.sample(frac=0.2 * iteration, replace=True, random_state=iteration)
final_df = pd.concat([df, augmented]).reset_index(drop=True)

# Save data
final_df.to_csv('data/iris.csv', index=False)
print(f"Iteration {iteration}: Dataset size = {len(final_df)}")

# Train model
X = final_df[iris.feature_names]
y = final_df['target']
model = RandomForestClassifier(n_estimators=10 * iteration, random_state=42)
model.fit(X, y)

# Save model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved with {10 * iteration} estimators")
