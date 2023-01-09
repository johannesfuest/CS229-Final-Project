import torch
import numpy as np
from pathlib import Path
import zero
import os
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from tab_ddpm.utils import FoundNANsError, index_to_log_onehot
from utils_train import get_model, make_dataset, concat_y_to_X
from lib import round_columns
import lib
from sklearn.metrics import f1_score

def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)


def sample(
        parent_dir,
        real_data_path='data/higgs-small',
        batch_size=2000,
        num_samples=0,
        model_type='mlp',
        model_params=None,
        model_path=None,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        T_dict=None,
        num_numerical_features=0,
        disbalance=None,
        device=torch.device('cuda:1'),
        seed=0,
        change_val=False
):
    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)
    D = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(D.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes('train')
    )

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model, num_timesteps=num_timesteps,
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
    )

    diffusion.to(device)
    diffusion.eval()

    _, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)
    # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
    if disbalance == 'fix':
        empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], \
                                                           empirical_class_dist[0]
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(),
                                            ddim=False)

    elif disbalance == 'fill':
        ix_major = empirical_class_dist.argmax().item()
        val_major = empirical_class_dist[ix_major].item()
        x_gen, y_gen = [], []
        for i in range(empirical_class_dist.shape[0]):
            if i == ix_major:
                continue
            distrib = torch.zeros_like(empirical_class_dist)
            distrib[i] = 1
            num_samples = val_major - empirical_class_dist[i].item()
            x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, distrib.float(),
                                                  ddim=False)
            x_gen.append(x_temp)
            y_gen.append(y_temp)

        x_gen = torch.cat(x_gen, dim=0)
        y_gen = torch.cat(y_gen, dim=0)

    else:
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(),
                                            ddim=False)

    # try:
    # except FoundNANsError as ex:
    #     print("Found NaNs during sampling!")
    #     loader = lib.prepare_fast_dataloader(D, 'train', 8)
    #     x_gen = next(loader)[0]
    #     y_gen = torch.multinomial(
    #         empirical_class_dist.float(),
    #         num_samples=8,
    #         replacement=True
    #     )
    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()
    print(X_gen[:, 0])
    ###
    # X_num_unnorm = X_gen[:, :num_numerical_features]
    # lo = np.percentile(X_num_unnorm, 2.5, axis=0)
    # hi = np.percentile(X_num_unnorm, 97.5, axis=0)
    # idx = (lo < X_num_unnorm) & (hi > X_num_unnorm)
    # X_gen = X_gen[np.all(idx, axis=1)]
    # y_gen = y_gen[np.all(idx, axis=1)]
    ###

    num_numerical_features = num_numerical_features + int(
        D.is_regression and not model_params["is_y_cond"])

    X_num_ = X_gen
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:])
        # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
        if T_dict['cat_encoding'] == 'one-hot':
            X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1],
                                                            X_num_[:, num_numerical_features:])

        X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

    if num_numerical_features_ != 0:
        # _, normalize = lib.normalize({'train' : X_num_real}, T_dict['normalization'], T_dict['seed'], True)
        np.save(os.path.join(parent_dir, 'X_num_unnorm'), X_gen[:, :num_numerical_features])
        X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
        X_num = X_num_[:, :num_numerical_features]

        X_num_real = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True)
        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print("Discrete cols:", disc_cols)
        if model_params['num_classes'] == 0:
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if len(disc_cols):
            X_num = round_columns(X_num_real, X_num, disc_cols)

    if num_numerical_features != 0:
        print("Num shape: ", X_num.shape)
        np.save(os.path.join(parent_dir, 'X_num_train'), X_num)
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)
    np.save(os.path.join(parent_dir, 'y_train'), y_gen)

### CS229 Code Starts Here ###
# Parts of the sample_partial function are borrowed from the above sample function
def sample_partial(
        parent_dir,
        real_data_path='data/higgs-small',
        batch_size=2000,
        num_samples=0,
        model_type='mlp',
        model_params=None,
        model_path=None,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        T_dict=None,
        num_numerical_features=0,
        disbalance=None,
        device=torch.device('cuda:1'),
        seed=0,
        change_val=False,
        exp_type="MCAR",
        exp_prop="0.1",
        to_impute=[],
        compare=False
):
    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)
    D = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(D.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes('train')
    )

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model, num_timesteps=num_timesteps,
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
    )

    diffusion.to(device)
    diffusion.eval()

    _, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)

    #### Get test data and one hot encode categorical features ####
    has_num = D.X_num is not None
    has_cat = D.X_cat is not None
    is_reg = D.is_regression

    y_test = torch.Tensor(D.y['test']).to(device)

    # X_true is the original test data matrix (with categorical features label encoded)
    # X_test is the test data matrix with transformed features

    if has_num and not has_cat:
        X_test = torch.Tensor(D.X_num['test']).to(device)
        X_num_true = np.load(os.path.join(real_data_path, "X_num_test.npy"), allow_pickle=True)
        if is_reg:
            y_true = np.load(os.path.join(real_data_path, "y_test.npy"), allow_pickle=True)
            X_num_true=concat_y_to_X(X_num_true, y_true)
        X_true = X_num_true
    elif has_cat and not has_num:
        X_cat_test = torch.Tensor(D.X_cat['test']).to(device).long()
        for i in range(X_cat_test.shape[1]):
            n_classes = diffusion.num_classes[i]
            rows = X_cat_test[:, i] < n_classes
            X_cat_test = X_cat_test[rows, :]
            y_test = y_test[rows]
        X_test = index_to_log_onehot(X_cat_test, diffusion.num_classes)

        X_cat_true = X_cat_test.cpu().numpy()
        X_true = X_cat_true
    else: # has_num and has_cat
        X_num_true = np.load(os.path.join(real_data_path, "X_num_test.npy"), allow_pickle=True)

        X_num_test = torch.Tensor(D.X_num['test']).to(device)
        X_cat_test = torch.Tensor(D.X_cat['test']).to(device).long()
        for i in range(X_cat_test.shape[1]):
            n_classes = diffusion.num_classes[i]
            rows = X_cat_test[:, i] < n_classes
            X_cat_test = X_cat_test[rows, :]
            X_num_test = X_num_test[rows, :]
            X_num_true = X_num_true[rows.cpu().numpy(), :]
            y_test = y_test[rows]
        X_cat_test_ohe = index_to_log_onehot(X_cat_test, diffusion.num_classes)
        X_test = torch.cat([X_num_test, X_cat_test_ohe], dim=1)

        X_cat_true = X_cat_test.cpu().numpy()
        if is_reg:
            y_true = np.load(os.path.join(real_data_path, "y_test.npy"), allow_pickle=True)
            X_num_true = concat_y_to_X(X_num_true, y_true)
        X_true = np.concatenate([X_num_true, X_cat_true], axis=1)

    #### Run selected experiments ####

    col_name_dict = lib.load_json(os.path.join(real_data_path, "info.json"))["col_name_dict"]
    list_of_masks = []
    for i, col_name in enumerate(to_impute):
        index = col_name_dict[col_name][0]
        is_cat = col_name_dict[col_name][1]

        if is_cat:
            mask = lib.mask_for_imputing(X_true, is_reg + index + num_numerical_features, exp_type[i], exp_prop[i])
            for j in range(diffusion.num_classes[index]):
                X_test[mask, num_numerical_features + index + is_reg + j] = torch.nan
        else:
            mask = lib.mask_for_imputing(X_true, is_reg + index, exp_type[i], exp_prop[i])
            X_test[mask, index + is_reg] = torch.nan
        list_of_masks += [mask]

    X_gen = diffusion.sample_from_known(X_test, y_test).numpy()

    #### Evaluate Imputation Performance ####

    if has_num:
        X_gen[:, :(num_numerical_features + is_reg)] = D.num_transform.inverse_transform(X_gen[:, :(num_numerical_features + is_reg)])

        X_num_real = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True)
        if is_reg:
            y_train = np.load(os.path.join(real_data_path, "y_train.npy"), allow_pickle=True)
            X_num_real = concat_y_to_X(X_num_real, y_train)
        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)

        if len(disc_cols):
            X_gen[:, :(num_numerical_features + is_reg)] = round_columns(X_num_real, X_gen[:, :(num_numerical_features + is_reg)], disc_cols)

    for i, col_name in enumerate(to_impute):
        index = col_name_dict[col_name][0]
        is_cat = col_name_dict[col_name][1]

        mask = list_of_masks[i]
        if is_cat:
            imputed_values = X_gen[mask, num_numerical_features + index + is_reg]
            true_values = X_true[mask, num_numerical_features + index + is_reg]

            result = f1_score(true_values, imputed_values, average="macro")
        else:
            imputed_values = X_gen[mask, index + is_reg]
            true_values = X_true[mask, index + is_reg]
            
            result = lib.calculate_rmse(true_values, imputed_values, None)

        lib.save_results(result, parent_dir, col_name, exp_type[i], exp_prop[i])

        if compare:
            if has_num and not has_cat:
                X_num_train =  np.load(os.path.join(real_data_path, "X_num_train.npy"))
                if is_reg:
                    y_train = np.load(os.path.join(real_data_path, "y_train.npy"))
                    X_num_train = concat_y_to_X(X_num_train, y_train)
                X_train = X_num_train
            elif has_cat and not has_num:
                X_cat_train = D.X_cat["train"]
                X_train = X_cat_train
            else:
                X_num_train = np.load(os.path.join(real_data_path, "X_num_train.npy"))
                if is_reg:
                    y_train = np.load(os.path.join(real_data_path, "y_train.npy"))
                    X_num_train = concat_y_to_X(X_num_train, y_train)
                X_cat_train = D.X_cat["train"]
                X_train = np.concatenate([X_num_train, X_cat_train], axis=1)

            if is_cat:
                mean_mode_result = lib.mean_mode_impute(X_cat_train, X_cat_true, is_cat, index + is_reg, mask)
                random_forest_result = lib.random_forest_impute(X_train, X_true, is_cat, is_reg + index + num_numerical_features, mask)
            else:
                mean_mode_result = lib.mean_mode_impute(X_num_train, X_num_true, is_cat, is_reg + index, mask)
                random_forest_result = lib.random_forest_impute(X_train, X_true, is_cat, is_reg + index, mask)

            lib.save_results(mean_mode_result, parent_dir, col_name, exp_type[i], exp_prop[i], method="mean_mode")
            lib.save_results(random_forest_result, parent_dir, col_name, exp_type[i], exp_prop[i], method="random forest")

### CS229 Code Ends Here ###
