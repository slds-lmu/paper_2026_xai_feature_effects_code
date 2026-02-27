"""
Script to run the main simulation study on feature effect error decomposition.
"""

import argparse
from configparser import ConfigParser
from pathlib import Path
import warnings
import logging
import os
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import KFold, train_test_split

from current_research_feature_effects.data_generating.data_generation import generate_data
from current_research_feature_effects.model_training import initialize_model
from current_research_feature_effects.model_eval import eval_model, empty_dict
from current_research_feature_effects.utils import (
    create_parameter_space,
    create_and_set_sim_dir,
    setup_logger,
    configure_worker_logger,
    save_model_results,
    save_fe_aggregated_results,
    save_fe_results,
    SimulationParameter,
)
from current_research_feature_effects.feature_effects import (
    compute_pdps,
    compute_ales,
    compute_feature_effect_metrics,
    compute_variance,
)


def simulate(
    params: SimulationParameter,
):
    np.random.seed(42)
    logging.info(f"Starting simulation with parameters: {params.groundtruth}, {params.model_name}, {params.sample_size}")

    # create directories
    os.makedirs(str(params.groundtruth), exist_ok=True)
    os.makedirs(Path(str(params.groundtruth)) / "results", exist_ok=True)

    # create databases for results
    engine_model_results = create_engine(
        f"sqlite:///{str(params.groundtruth)}{params.config.get('storage', 'model_results')}"
    )
    engine_effects_results = create_engine(
        f"sqlite:///{str(params.groundtruth)}{params.config.get('storage', 'effects_results')}"
    )

    feature_names = params.groundtruth.feature_names
    grid_points = params.config.getint("feature_effects", "grid_points")
    quantiles = np.linspace(0.0001, 0.9999, grid_points, endpoint=True)
    grid_values = [params.groundtruth.get_theoretical_quantiles(feature, quantiles) for feature in feature_names]
    center_curves = params.config["feature_effects"].getboolean("centered")
    remove_first_last = params.config["feature_effects"].getboolean("remove_first_last")
    k_cv = params.config.getint("simulation_metadata", "k_cv")

    if params.do_var_decomp:
        n_mc_datasets = params.config.getint("simulation_metadata", "n_mc_datasets")
        dataset_base_seed = params.config.getint("simulation_metadata", "mc_dataset_base_seed")

    # sample data for MC approximation of groundtruth feature effects
    X_mc_groundtruth, _, _, _ = generate_data(
        groundtruth=params.groundtruth,
        n_train=params.config.getint("simulation_metadata", "n_groundtruth_mc"),
        n_test=1,
        snr=0,
        seed=params.config.getint("simulation_metadata", "groundtruth_mc_data_seed"),
    )

    # estimate groundtruth feature effects
    pdp_groundtruth = compute_pdps(
        params.groundtruth,
        X_mc_groundtruth,
        feature_names,
        grid_values,
        center_curves=center_curves,
        remove_first_last=remove_first_last,
    )
    ale_groundtruth = compute_ales(
        params.groundtruth,
        X_mc_groundtruth,
        feature_names,
        grid_values,
        center_curves=center_curves,
        remove_first_last=remove_first_last,
    )

    pdps = {}
    ales = {}
    if params.do_var_decomp:
        pdp_variances = {}
        ale_variances = {}

    n_sim = params.config.getint("simulation_params", "n_sim")
    failed_runs = 0
    for sim_no in range(n_sim):
        logging.info(
            f"Starting simulation {sim_no+1}/{n_sim} "
            + f"for {params.groundtruth} {params.model_name} {params.sample_size}."
        )
        try:
            # generate data
            X_, y_, X_test, y_test = generate_data(
                groundtruth=params.groundtruth,
                n_train=params.sample_size,
                n_test=params.config.getint("simulation_metadata", "n_test"),
                snr=params.snr,
                seed=sim_no,
            )

            ### A: estimation on training data ###

            # initialize model
            model_a = initialize_model(
                params.model_config,
                params.model_name,
                params.groundtruth,
                params.sample_size,
                params.snr,
                params.config,
            )

            # full data used as training set
            X_train, y_train = X_, y_

            # try to train and evaluate model_a
            try:
                model_a.fit(X_train, y_train)
                model_a_metrics = eval_model(model_a, X_train, y_train, X_test, y_test)
                save_model_results(
                    model_a_metrics, table="model_a_results", conn=engine_model_results, params=params, sim_no=sim_no
                )
            except Exception as e:
                model_a_metrics = empty_dict()
                save_model_results(
                    model_a_metrics, table="model_a_results", conn=engine_model_results, params=params, sim_no=sim_no
                )
                warnings.warn(
                    f"Training of model A {params.model_name} {sim_no+1} {params.sample_size} failed with error:\n{e}"
                )
                raise e

            # estimate pdp
            pdp_train = compute_pdps(model_a, X_train, feature_names, grid_values, center_curves, remove_first_last)
            ale_train = compute_ales(model_a, X_train, feature_names, grid_values, center_curves, remove_first_last)

            ### B: estimation on validation data ###

            # initialize model
            model_b = initialize_model(
                params.model_config,
                params.model_name,
                params.groundtruth,
                params.sample_size,
                params.snr,
                params.config,
            )

            # split data into training and validation set
            X_train, X_val, y_train, _ = train_test_split(X_, y_, test_size=params.val_share, random_state=sim_no)

            # try to train and evaluate model_b
            try:
                model_b.fit(X_train, y_train)
                model_b_metrics = eval_model(model_b, X_train, y_train, X_test, y_test)
                save_model_results(
                    model_b_metrics, table="model_b_results", conn=engine_model_results, params=params, sim_no=sim_no
                )
            except Exception as e:
                model_b_metrics = empty_dict()
                save_model_results(
                    model_b_metrics, table="model_b_results", conn=engine_model_results, params=params, sim_no=sim_no
                )
                warnings.warn(
                    f"Training of model B {params.model_name} {sim_no+1} {params.sample_size} failed with error:\n{e}"
                )
                raise e

            # estimate pdp
            pdp_val = compute_pdps(model_b, X_val, feature_names, grid_values, center_curves, remove_first_last)
            ale_val = compute_ales(model_b, X_val, feature_names, grid_values, center_curves, remove_first_last)

            # C: estimation with CV

            # initialize model / learner
            model_c = initialize_model(
                params.model_config,
                params.model_name,
                params.groundtruth,
                params.sample_size,
                params.snr,
                params.config,
            )

            cv = KFold(n_splits=k_cv, shuffle=True, random_state=42)
            cv_splits = list(cv.split(X=X_, y=y_))
            cv_models = []
            cv_holdout_sets = []

            try:
                for train_index, test_index in cv_splits:
                    X_train_cv, y_train_cv = X_[train_index], y_[train_index]
                    model_fold = deepcopy(model_c)
                    model_fold.fit(X_train_cv, y_train_cv)
                    cv_models.append(model_fold)
                    cv_holdout_sets.append(test_index)
            except Exception as e:
                warnings.warn(
                    f"Training of CV models {params.model_name} {sim_no+1} {params.sample_size} failed with error:\n{e}"
                )
                raise e

            pdp_cv = (
                sum(
                    compute_pdps(
                        model_fold,
                        X_[test_index],
                        feature_names,
                        grid_values,
                        center_curves,
                        remove_first_last,
                    )
                    for test_index, model_fold in zip(cv_holdout_sets, cv_models)
                )
                / k_cv
            )
            ale_cv = (
                sum(
                    compute_ales(
                        model_fold,
                        X_[test_index],
                        feature_names,
                        grid_values,
                        center_curves,
                        remove_first_last,
                    )
                    for test_index, model_fold in zip(cv_holdout_sets, cv_models)
                )
                / k_cv
            )

            for split, pdp, ale in zip(
                ["train", "val", "cv"], [pdp_train, pdp_val, pdp_cv], [ale_train, ale_val, ale_cv]
            ):
                pdps[split] = [pdp] if split not in pdps else pdps[split] + [pdp]
                ales[split] = [ale] if split not in ales else ales[split] + [ale]

            if not params.do_var_decomp:
                continue

            pdps_mc = {}
            ales_mc = {}
            for k in range(n_mc_datasets):
                X_mc, y_mc, _, _ = generate_data(
                    groundtruth=params.groundtruth,
                    n_train=params.sample_size,
                    n_test=1,
                    snr=params.snr,
                    seed=dataset_base_seed + sim_no * n_mc_datasets + k,
                )
                _, X_mc_val, _, _ = train_test_split(X_mc, y_mc, test_size=params.val_share, random_state=sim_no)

                pdp_train_mc = compute_pdps(model_a, X_mc, feature_names, grid_values, center_curves, remove_first_last)
                ale_train_mc = compute_ales(model_a, X_mc, feature_names, grid_values, center_curves, remove_first_last)
                pdp_val_mc = compute_pdps(
                    model_b, X_mc_val, feature_names, grid_values, center_curves, remove_first_last
                )
                ale_val_mc = compute_ales(
                    model_b, X_mc_val, feature_names, grid_values, center_curves, remove_first_last
                )
                pdp_cv_mc = (
                    sum(
                        compute_pdps(
                            model_fold,
                            X_mc[test_index],
                            feature_names,
                            grid_values,
                            center_curves,
                            remove_first_last,
                        )
                        for (_, test_index), model_fold in zip(cv_splits, cv_models)
                    )
                    / k_cv
                )
                ale_cv_mc = (
                    sum(
                        compute_ales(
                            model_fold,
                            X_mc[test_index],
                            feature_names,
                            grid_values,
                            center_curves,
                            remove_first_last,
                        )
                        for (_, test_index), model_fold in zip(cv_splits, cv_models)
                    )
                    / k_cv
                )

                for split, pdp, ale in zip(
                    ["train", "val", "cv"],
                    [pdp_train_mc, pdp_val_mc, pdp_cv_mc],
                    [ale_train_mc, ale_val_mc, ale_cv_mc],
                ):
                    pdps_mc[split] = [pdp] if split not in pdps_mc else pdps_mc[split] + [pdp]
                    ales_mc[split] = [ale] if split not in ales_mc else ales_mc[split] + [ale]

            for split in ["train", "val", "cv"]:
                pdp_variances[split] = (
                    [compute_variance(pdps_mc[split])]
                    if split not in pdp_variances
                    else pdp_variances[split] + [compute_variance(pdps_mc[split])]
                )
                ale_variances[split] = (
                    [compute_variance(ales_mc[split])]
                    if split not in ale_variances
                    else ale_variances[split] + [compute_variance(ales_mc[split])]
                )
        except Exception as e:
            failed_runs += 1
            logging.error(
                "Simulation %s/%s failed for %s %s %s with error: %s",
                sim_no + 1,
                n_sim,
                params.groundtruth,
                params.model_name,
                params.sample_size,
                e,
            )
            continue

    logging.info("Failed simulations: %s/%s", failed_runs, n_sim)
    print(f"Failed simulations: {failed_runs}/{n_sim}")

    # compute MSE, Bias^2, Variance for pdp and ale estimates
    pdp_metrics_base = {
        split: compute_feature_effect_metrics(pdps_split, pdp_groundtruth) for split, pdps_split in pdps.items()
    }
    ale_metrics_base = {
        split: compute_feature_effect_metrics(ales_split, ale_groundtruth) for split, ales_split in ales.items()
    }

    # save metrics
    save_fe_results(pdp_metrics_base, params, "pdp")
    save_fe_results(ale_metrics_base, params, "ale")

    # aggregate metrics
    pdp_metrics_agg = {
        split: {metric: pdp_metrics_base[split][metric].mean() for metric in pdp_metrics_base[split].keys()}
        for split in pdp_metrics_base.keys()
    }
    ale_metrics_agg = {
        split: {metric: ale_metrics_base[split][metric].mean() for metric in ale_metrics_base[split].keys()}
        for split in ale_metrics_base.keys()
    }

    # save aggregated metrics
    save_fe_aggregated_results(pdp_metrics_agg, engine_effects_results, params, "pdp")
    save_fe_aggregated_results(ale_metrics_agg, engine_effects_results, params, "ale")

    # compute variance decomposition
    if params.do_var_decomp:
        pdp_var_metrics = {
            split: {"Variance": metrics["Variance"]} for split, metrics in pdp_metrics_base.items()
        }
        ale_var_metrics = {
            split: {"Variance": metrics["Variance"]} for split, metrics in ale_metrics_base.items()
        }
        pdp_mc_metrics = {
            split: {"MC Variance": sum(pdp_variances[split]) / len(pdp_variances[split])}
            for split in pdp_variances.keys()
        }
        ale_mc_metrics = {
            split: {"MC Variance": sum(ale_variances[split]) / len(ale_variances[split])}
            for split in ale_variances.keys()
        }
        for split, metrics in pdp_mc_metrics.items():
            pdp_var_metrics.setdefault(split, {}).update(metrics)
        for split, metrics in ale_mc_metrics.items():
            ale_var_metrics.setdefault(split, {}).update(metrics)

        for split, metrics in pdp_var_metrics.items():
            if "Variance" in metrics and "MC Variance" in metrics:
                metrics["Model Variance"] = metrics["Variance"] - metrics["MC Variance"]
        for split, metrics in ale_var_metrics.items():
            if "Variance" in metrics and "MC Variance" in metrics:
                metrics["Model Variance"] = metrics["Variance"] - metrics["MC Variance"]

        save_fe_results(pdp_var_metrics, params, "pdp_var")
        save_fe_results(ale_var_metrics, params, "ale_var")

        pdp_var_metrics_agg = {
            split: {metric: pdp_var_metrics[split][metric].mean() for metric in pdp_var_metrics[split].keys()}
            for split in pdp_var_metrics.keys()
        }
        ale_var_metrics_agg = {
            split: {metric: ale_var_metrics[split][metric].mean() for metric in ale_var_metrics[split].keys()}
            for split in ale_var_metrics.keys()
        }

        save_fe_aggregated_results(pdp_var_metrics_agg, engine_effects_results, params, "pdp_var")
        save_fe_aggregated_results(ale_var_metrics_agg, engine_effects_results, params, "ale_var")


if __name__ == "__main__":
    # parse arguments and read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config.ini file")
    args = parser.parse_args()
    sim_config = ConfigParser()
    sim_config.read(Path(args.config))

    # setup logging
    log_queue, log_listener = setup_logger(Path(sim_config.get("storage", "log_dir")))

    # create parameter space
    param_space = create_parameter_space(sim_config)
    logging.info(f"Created parameter space with {len(param_space)} simulation parameters.")

    # create directories and processes
    create_and_set_sim_dir(sim_config, config_path=Path(args.config))
    num_processes = min(len(param_space), cpu_count())

    # run simulations
    with Pool(processes=num_processes, initializer=configure_worker_logger, initargs=(log_queue,)) as pool:
        pool.map(
            simulate,
            param_space,
        )

    log_listener.stop()
