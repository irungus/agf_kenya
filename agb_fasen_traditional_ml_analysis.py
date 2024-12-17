
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Define the AGB_Fasen formula
def agb_fasen(dbh, height):
    dbh_inches = dbh / 2.54  # Convert cm to inches
    height_feet = height * 3.281  # Convert meters to feet
    return 0.25 * (dbh_inches ** 2) * height_feet

# Define traditional models
def agb_chave_2005(dbh, height): 
    return np.exp(2.187 + 0.16 * np.log((dbh ** 2) * height))

def agb_brown(dbh): 
    return 21.297 - 6.53 * dbh + 0.74 * (dbh ** 2)

def agb_henry(dbh, height): 
    return 0.051 * ((dbh ** 2) * height) ** 0.930

# Load the dataset
acacia_df = pd.read_csv("acacia_df_cleaned.csv")

# Initialize machine learning models
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
ann_model = MLPRegressor(hidden_layer_sizes=(20,), max_iter=200, random_state=42)

# Storage for results and predictions
species_results = []
all_predictions = []

# Loop through each species
for species, group in acacia_df.groupby('genus_species'):
    X_species = group[['dbh', 'height_m']]

    # Calculate AGB_Fasen and traditional model outputs
    group['AGB_Fasen'] = group.apply(lambda x: agb_fasen(x['dbh'], x['height_m']), axis=1)
    group['AGB_Chave2005'] = group.apply(lambda x: agb_chave_2005(x['dbh'], x['height_m']), axis=1)
    group['AGB_Brown'] = group['dbh'].apply(agb_brown)
    group['AGB_Henry'] = group.apply(lambda x: agb_henry(x['dbh'], x['height_m']), axis=1)

    # Machine learning target
    y_species = group['AGB_Fasen']
    X_train, X_test, y_train, y_test = train_test_split(X_species, y_species, test_size=0.2, random_state=42)

    # Train machine learning models
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    ann_model.fit(X_train, y_train)

    # Predict and collect RMSE and R²
    rf_preds = rf_model.predict(X_test)
    gb_preds = gb_model.predict(X_test)
    ann_preds = ann_model.predict(X_test)

    metrics = {
        "Species": species,
        "AGB_Fasen RMSE": np.sqrt(mean_squared_error(y_test, group.loc[X_test.index, 'AGB_Fasen'])),
        "Chave et al. (2005) RMSE": np.sqrt(mean_squared_error(y_test, group.loc[X_test.index, 'AGB_Chave2005'])),
        "Brown (1997) RMSE": np.sqrt(mean_squared_error(y_test, group.loc[X_test.index, 'AGB_Brown'])),
        "Henry et al. (2009) RMSE": np.sqrt(mean_squared_error(y_test, group.loc[X_test.index, 'AGB_Henry'])),
        "Random Forest RMSE": np.sqrt(mean_squared_error(y_test, rf_preds)),
        "Gradient Boosting RMSE": np.sqrt(mean_squared_error(y_test, gb_preds)),
        "ANN RMSE": np.sqrt(mean_squared_error(y_test, ann_preds)),
        "Random Forest R²": r2_score(y_test, rf_preds),
        "Gradient Boosting R²": r2_score(y_test, gb_preds),
        "ANN R²": r2_score(y_test, ann_preds),
    }
    species_results.append(metrics)

    # Predict for all rows in the species
    group['AGB_RandomForest'] = rf_model.predict(X_species)
    group['AGB_GradientBoosting'] = gb_model.predict(X_species)
    group['AGB_ANN'] = ann_model.predict(X_species)
    all_predictions.append(group)

# Combine and save results
all_predictions_df = pd.concat(all_predictions, axis=0)
pd.DataFrame(species_results).to_csv("species_results_fasen_traditional_ml.csv", index=False)
all_predictions_df.to_csv("agb_predictions_fasen_traditional_ml.csv", index=False)
