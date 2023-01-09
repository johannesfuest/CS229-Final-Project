This github repository contains all the code and information on our final project for Stanford class CS 229 Machine Learning. Created by Johannes Fuest, Alex Thiesmeyer and Linus Hein. 


# CS229 Final Project: Diffusion Models for Data Imputation

### Attribution:
Most of the above code is due to Kotelnikov et al., 2022 in association with their paper [TabDDPM: Modeling Tabular Data with Diffusion Models](https://arxiv.org/pdf/2209.15421.pdf). The repo for that project can be found [here](https://github.com/rotot0/tab-ddpm).

The files that we added for our data imputation method are listed below:  

```scripts/import_data.py```  

```scripts/run_exps.py```  

```scripts/lib/impute_utils.py```  

```scripts/demo_noise_addition.py```  


In addition, we edited many existing files, with particular attention paid to ```scripts/sample.py``` and ```scripts/tab_ddpm/gaussian_multinomial_diffsuion.py```. 
Wherever we have edited the original TabDDPM authors' code, we have tried to clearly comment to that effect. The implementation of missingness patterns is adapted from the [jenga package](https://pypi.org/project/jenga/), due to Sebastian Schelter.  All other code should be assumed to be due to Kotelnikov et al.


### Guidelines to Run Code:
See the [original repo](https://github.com/rotot0/tab-ddpm) for instructions on how to set up your environment and download pretuned datasets. Whenever attempting to use the TabDDPM model, the desired device to use (cpu, cuda:0 etc.) must be specified in the config.toml file associated with the type of experiment you are trying to run.

The following example commands walk through a typical use of our method.

Imports a dataset from a csv file. Splits the data into training, validation and test sets and splits numerical features from categorical features, saving as .npy files. Saves column names in a dictionary for future use. Creates an info.json file with dataset information. Additionally splits data into five folds for cross-validation. 
```
python scripts/import_data.py [path] [ds_name] [task_type] [target] [--cat_features] [--to_drop]
python scripts/import_data.py "datasets/abalone.csv" abalone regression Rings --cat_features Sex --to_drop id
```

Tunes hyperparameters for the catboost model on the chosen dataset with cross-validation.
```
python scripts/tune_evaluation_model.py [ds_name] [model] [tune_type] [device]
python scripts/tune_evaluation_model.py abalone catboost cv cuda:0
```

Tunes parameters of the TabDDPM model on the chosen dataset.
```
python scripts/tune_ddpm.py [ds_name] [train_size] [eval_type] [eval_model] [prefix]
python scripts/tune_ddpm.py abalone 2672 synthetic catboost ddpm_cb
```

Trains the TabDDPM model given a specific set of hyperparameters.
```
python scripts/pipeline.py [--config] [--train]
python scripts/pipeline.py --config exp/abalone/ddpm_cb_best/config.toml --train
```

Runs a single imputation experiment on chosen features with missingness pattern --exp_type and proportion of missing data --exp_prop. Compares to mean/mode and random forest baselines.
```
python scripts/pipeline.py [--config] [--sample_partial] [--to_impute] [exp_type] [--exp_prop] [--compare]
python scripts/pipeline.py --config exp/abalone/ddpm_cb_best/config.toml --sample_partial --to_impute Length --exp_type MCAR --exp_prop 0.1 --compare
```

For our results generation, this runs all desired experiments on a particular feature, with comparison enabled.
```
python scripts/run_exps.py [ds_name] [to_impute]
python scripts/run_exps.py abalone Length
```
Results are stored in ```exp/{ds_name}/ddpm_cb_best/imp_exp_results```
