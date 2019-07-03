import os
import pandas as pd
import pickle
from behavenet.data.utils import get_region_list
from behavenet.fitting.utils import get_output_dirs
from behavenet.fitting.utils import get_output_session_dir
from behavenet.fitting.utils import get_subdirs


def get_dataset_str(hparams):
    return os.path.join(hparams['expt'], hparams['animal'], hparams['session'])


def get_r2s_by_trial(hparams, model_types):
    """
    For a given session, load R^2 metrics from all decoders defined by hparams
    (n_ae_latents, experiment_name)

    Args:
        hparams (dict)
        model_types (list of strs): 'ff' | 'linear'

    Returns
        (pd.DataFrame)
    """

    dataset = get_dataset_str(hparams)
    region_names = get_region_list(hparams)

    metrics = []
    model_indx = 0
    model_counter = 0
    for region in region_names:
        hparams['region'] = region
        for model_type in model_types:

            hparams['session_dir'], _ = get_output_session_dir(hparams)
            _, expt_dir = get_output_dirs(
                hparams,
                model_type=model_type,
                model_class=hparams['model_class'],
                expt_name=hparams['experiment_name'])

            # gather all versions
            try:
                versions = get_subdirs(expt_dir)
            except Exception:
                print('No models in %s; skipping' % expt_dir)

            # load csv files with model metrics (saved out from test tube)
            for i, version in enumerate(versions):
                # read metrics csv file
                model_dir = os.path.join(expt_dir, version)
                try:
                    metric = pd.read_csv(
                        os.path.join(model_dir, 'metrics.csv'))
                    model_counter += 1
                except:
                    continue
                with open(os.path.join(model_dir, 'meta_tags.pkl'), 'rb') as f:
                    hparams = pickle.load(f)
                # append model info to metrics ()
                version_num = version[8:]
                metric['version'] = str('version_%i' % model_indx + version_num)
                metric['region'] = region
                metric['dataset'] = dataset
                metric['model_type'] = model_type
                for key, val in hparams.items():
                    if isinstance(val, (str, int, float)):
                        metric[key] = val
                metrics.append(metric)

            model_indx += 10000  # assumes no more than 10k model versions/expt
    # put everything in pandas dataframe
    metrics_df = pd.concat(metrics, sort=False)
    return metrics_df


def get_best_models(metrics_df):
    """
    Find best decoder over l2 regularization and learning rate (per dataset,
    region, n_lags, and n_hid_layers). Returns a dataframe with test R^2s for
    each batch, for the best decoder in each category.

    Args:
        metrics_df (pd.DataFrame): output of get_r2s_by_trial

    Returns:
        (pd.DataFrame)
    """
    # for each version, only keep rows where test_loss is not nan
    data_queried = metrics_df[pd.notna(metrics_df.test_loss)]
    best_models_list = []

    # take min over val losses
    loss_mins = metrics_df.groupby(
        ['dataset', 'n_lags', 'n_hid_layers',
         'learning_rate', 'l2_reg', 'version', 'region']) \
        .min().reset_index()
    datasets = metrics_df.dataset.unique()
    datasets.sort()
    regions = metrics_df.region.unique()
    regions.sort()
    n_lags = metrics_df.n_lags.unique()
    n_lags.sort()
    n_hid_layers = metrics_df.n_hid_layers.unique()
    n_hid_layers.sort()
    for dataset in datasets:
        for region in regions:
            for lag in n_lags:
                for layer in n_hid_layers:
                    # get all models with this number of lags
                    single_hp = loss_mins[
                        (loss_mins.n_lags == lag)
                        & (loss_mins.n_hid_layers == layer)
                        & (loss_mins.region == region)
                        & (loss_mins.dataset == dataset)]
                    # find best version from these models
                    best_version = loss_mins.iloc[
                        single_hp.val_loss.idxmin()].version
                    # index back into original data to grab test loss on all
                    # batches
                    best_models_list.append(
                        data_queried[data_queried.version == best_version])

    return pd.concat(best_models_list)


def get_r2s_across_trials(hparams, best_models_df):
    """
    Calculate R^2 across all test trials (rather than on a trial-by-trial
    basis)

    Args:
        hparams (dict)
        best_models_df (pd.DataFrame): output of get_best_models

    Returns:
        (pd.DataFrame)
    """

    from behavenet.fitting.eval import get_test_r2

    dataset = get_dataset_str(hparams)
    versions = best_models_df.version.unique()

    all_test_r2s = []
    for version in versions:
        model_version = str(int(version[8:]) % 10000)
        hparams['model_type'] = best_models_df[
            best_models_df.version == version].model_type.unique()[0]
        hparams['region'] = best_models_df[
            best_models_df.version == version].region.unique()[0]
        hparams_, r2 = get_test_r2(hparams, model_version)
        all_test_r2s.append(pd.DataFrame({
            'dataset': dataset,
            'region': hparams['region'],
            'n_hid_layers': hparams_['n_hid_layers'],
            'n_lags': hparams_['n_lags'],
            'model_type': hparams['model_type'],
            'r2': r2}, index=[0]))
    return pd.concat(all_test_r2s)
