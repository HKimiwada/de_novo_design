import matplotlib.pyplot as plt
import pandas as pd

df_valid = pd.read_csv("tensile_strength_predictions.csv")
df_valid = df_valid.dropna(subset=["tensile_strength", "predicted_tensile_strength"])

actuals = df_valid["tensile_strength"].values
preds = df_valid["predicted_tensile_strength"].values

plt.figure(figsize=(6,6))
plt.scatter(actuals, preds, alpha=0.6)
mn, mx = actuals.min(), actuals.max()
plt.plot([mn, mx], [mn, mx], 'k--', lw=1)
plt.xlabel('Actual tensile strength')
plt.ylabel('Predicted tensile strength')
plt.title('Predicted vs Actual — PandevModel')
plt.tight_layout()
plt.show()