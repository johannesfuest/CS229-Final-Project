### CS229 Code Starts Here ###

import lib
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score
from scipy.stats import mode

#### NOTE: The generation of corruption patterns for MCAR, MNAR, and MAR is adapted from the jenga package,
#### credit to https://github.com/schelterlabs/jenga

def mask_for_imputing(data, col_to_impute, type, prop):
    data = pd.DataFrame(data)     
    n = data.shape[0]
    d = data.shape[1]  
    if type == "MCAR":
        mask = np.random.permutation(n)[:int(n*prop)]
    else:
        n_vals_to_discard = int(n * prop)
        perc_start = np.random.randint(0, n - n_vals_to_discard)
        perc_indices = range(perc_start, perc_start + n_vals_to_discard)
        if type == "MNAR":
            mask = data[col_to_impute].sort_values().iloc[perc_indices].index
        elif type == "MAR":
            depends_on_col = np.random.choice(list(set(range(d)) - {col_to_impute}))
            mask = data[depends_on_col].sort_values().iloc[perc_indices].index
        
    return np.sort(mask)

def save_results(result, parent_dir, col_name, exp_type, exp_prop, method="tddpm"):
    results_path = Path(parent_dir + "/imp_exp_results")
    results_path.mkdir(exist_ok=True, parents=True)
    prop = str(exp_prop)
    save_path = results_path / f"results.json"
    if not save_path.is_file():
        results = {f"{col_name}": {f"{exp_type}": {f"{prop}": {f"{method}"  : result}}}}
    else:
        results = lib.load_json(save_path)
        if col_name in results.keys():
            if exp_type in results[col_name].keys():
                if prop in results[col_name][exp_type].keys():
                    results[col_name][exp_type][prop].update({method : result})
                else:
                    results[col_name][exp_type][prop] = {f"{method}"  : result}
            else:
                results[col_name][exp_type] = {f"{prop}": {f"{method}"  : result}}
        else:
            results[col_name] = {f"{exp_type}" : {f"{prop}": {f"{method}"  : result}}}

    lib.dump_json(results, save_path)

def mean_mode_impute(train_data, test_data, is_cat, index, mask):
    if is_cat:
        imputed_values = np.repeat(mode(train_data[:, index])[0][0], len(mask))
        true_values = test_data[mask, index]
        result = f1_score(true_values, imputed_values, average="macro")
    else:
        imputed_values = np.repeat(np.mean(train_data[:, index]), len(mask))
        true_values = test_data[mask, index]
        result = lib.calculate_rmse(true_values, imputed_values, None)

    return result

def random_forest_impute(train_data, test_data, is_cat, index, mask):
    if is_cat:
        X_train = train_data[:, np.arange(train_data.shape[1]) != index]
        y_train = train_data[:, index]
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)

        X_test = test_data[:, np.arange(test_data.shape[1]) != index]
        
        imputed_values = rf_model.predict(X_test[mask, :])
        true_values = test_data[mask, index]
        result = f1_score(true_values, imputed_values, average="macro")
    else:
        X_train = train_data[:, np.arange(train_data.shape[1]) != index]
        y_train = train_data[:, index]
        rf_model = RandomForestRegressor()
        rf_model.fit(X_train, y_train)

        X_test = test_data[:, np.arange(test_data.shape[1]) != index]

        imputed_values = rf_model.predict(X_test[mask, :])
        true_values = test_data[mask, index]
        result = lib.calculate_rmse(true_values, imputed_values, None)
    
    return result

### CS229 Code Ends Here ###
