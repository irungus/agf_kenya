
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load the dataset
file_path = "acacia_df_cleaned.csv"
acacia_df = pd.read_csv(file_path)

# Step 1: Clean and Inspect the Data
acacia_df = acacia_df[['dbh', 'height_m', 'genus_species']]  # Select relevant columns
acacia_df = acacia_df.dropna()  # Drop rows with missing values

# Step 2: Define Traditional AGB Models
def agb_eq1(dbh):
    """Model 1: AGB = 0.091 * dbh^2.472"""
    return 0.091 * (dbh ** 2.472)

def agb_chave_2005(dbh, height):
    """Chave et al. (2005) dry forests: AGB = exp(2.187 + 0.16 * ln(DBH^2 * H))"""
    return np.exp(2.187 + 0.16 * np.log((dbh ** 2) * height))

def agb_brown(dbh):
    """Brown (1997) for wet forests: AGB = 21.297 - 6.53 * dbh + 0.74 * dbh^2."""
    return 21.297 - 6.53 * dbh + 0.74 * (dbh ** 2)

def agb_henry(dbh, height):
    """Henry et al. (2009): AGB = 0.051 * (dbh^2 * height)^0.930."""
    return 0.051 * ((dbh ** 2) * height) ** 0.930

# Step 3: Apply Traditional Models
acacia_df['AGB_Eq1'] = acacia_df['dbh'].apply(agb_eq1)
acacia_df['AGB_Chave2005'] = acacia_df.apply(lambda x: agb_chave_2005(x['dbh'], x['height_m']), axis=1)
acacia_df['AGB_Brown'] = acacia_df['dbh'].apply(agb_brown)
acacia_df['AGB_Henry'] = acacia_df.apply(lambda x: agb_henry(x['dbh'], x['height_m']), axis=1)

# Step 4: Prepare Data for ML Models
X = acacia_df[['dbh', 'height_m']]  # Predictors
y = acacia_df['AGB_Eq1']  # Target: AGB from Eq. (1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ML Models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Predict using ML Models
acacia_df['AGB_RandomForest'] = rf_model.predict(X)
acacia_df['AGB_GradientBoosting'] = gb_model.predict(X)

# Compare Models Using RMSE and R²
metrics = {
    "Model": ["Chave et al. (2005)", "Brown (1997)", "Henry et al. (2009)", "Random Forest", "Gradient Boosting"],
    "RMSE": [
        np.sqrt(mean_squared_error(y, acacia_df['AGB_Chave2005'])),
        np.sqrt(mean_squared_error(y, acacia_df['AGB_Brown'])),
        np.sqrt(mean_squared_error(y, acacia_df['AGB_Henry'])),
        np.sqrt(mean_squared_error(y, acacia_df['AGB_RandomForest'])),
        np.sqrt(mean_squared_error(y, acacia_df['AGB_GradientBoosting'])),
    ],
    "R²": [
        r2_score(y, acacia_df['AGB_Chave2005']),
        r2_score(y, acacia_df['AGB_Brown']),
        r2_score(y, acacia_df['AGB_Henry']),
        r2_score(y, acacia_df['AGB_RandomForest']),
        r2_score(y, acacia_df['AGB_GradientBoosting']),
    ],
}

metrics_df = pd.DataFrame(metrics)
print(metrics_df)

# Save the updated dataset
acacia_df.to_csv("acacia_with_ml_agb_models.csv", index=False)
