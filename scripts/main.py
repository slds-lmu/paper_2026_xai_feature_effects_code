"""
Script to run the main simulation study on feature effect error decomposition.
"""

import argparse
from configparser import ConfigParser
from pathlib import Path
import warnings
import logging
import os
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
    compute_cv_feature_effect,
    compute_feature_effect_metrics,
)


def simulate(
    params: SimulationParameter,
):
    np.random.seed(42)
    logging.info(f"Starting simulation with parameters: {params.groundtruth}, {params.model_name}, {params.n_train}")

    # create directories
    os.makedirs(str(params.groundtruth), exist_ok=True)
    os.makedirs(Path(str(params.groundtruth)) / "results", exist_ok=True)

    # create databases for results
    engine_model_a_results = create_engine(
        f"sqlite:///{str(params.groundtruth)}{params.config.get('storage', 'model_a_results')}"
    )
    engine_model_b_results = create_engine(
        f"sqlite:///{str(params.groundtruth)}{params.config.get('storage', 'model_b_results')}"
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

    # sample data for MC approximation of groundtruth feature effects
    X_mc, _, _, _ = generate_data(
        groundtruth=params.groundtruth,
        n_train=params.config.getint("simulation_metadata", "n_mc"),
        n_test=1,
        snr=0,
        seed=params.config.getint("simulation_metadata", "mc_data_seed"),
    )

    # estimate groundtruth feature effects
    pdp_groundtruth = compute_pdps(
        params.groundtruth,
        X_mc,
        feature_names,
        grid_values,
        center_curves=center_curves,
        remove_first_last=remove_first_last,
    )
    ale_groundtruth = compute_ales(
        params.groundtruth,
        X_mc,
        feature_names,
        grid_values,
        center_curves=center_curves,
        remove_first_last=remove_first_last,
    )

    pdps = {}
    ales = {}

    for sim_no in range(params.config.getint("simulation_params", "n_sim")):
        logging.info(
            f"Starting simulation {sim_no+1}/{params.config.getint('simulation_params', 'n_sim')} "
            + f"for {params.groundtruth} {params.model_name} {params.sample_size}."
        )
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
        except Exception as e:
            model_a_metrics = empty_dict()
            warnings.warn(f"Training of model A {params.model_name} {sim_no+1} {params.sample_size} failed with error:\n{e}")

        # save model results
        save_model_results(model_a_metrics, conn=engine_model_a_results, params=params, sim_no=sim_no)

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
            model_b_metrics = eval_model(model_b, X_train, y_train, X_val, y_test)
        except Exception as e:
            model_b_metrics = empty_dict()
            warnings.warn(f"Training of model B {params.model_name} {sim_no+1} {params.sample_size} failed with error:\n{e}")

        # save model results
        save_model_results(model_b_metrics, conn=engine_model_b_results, params=params, sim_no=sim_no)

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
        pdp_cv = compute_cv_feature_effect(
            model_c,
            X_,
            y_,
            cv,
            feature_names,
            [grid_values] * k_cv,
            compute_pdps,
            center_curves,
            remove_first_last,
        )
        ale_cv = compute_cv_feature_effect(
            model_c,
            X_,
            y_,
            cv,
            feature_names,
            [grid_values] * k_cv,
            compute_ales,
            center_curves,
            remove_first_last,
        )

        for split, pdp, ale in zip(["train", "val", "cv"], [pdp_train, pdp_val, pdp_cv], [ale_train, ale_val, ale_cv]):
            pdps[split] = [pdp] if split not in pdps else pdps[split] + [pdp]
            ales[split] = [ale] if split not in ales else ales[split] + [ale]

    # compute MSE, Bias^2, Variance for pdp and ale estimates
    pdp_metrics = {
        split: compute_feature_effect_metrics(pdps_split, pdp_groundtruth) for split, pdps_split in pdps.items()
    }
    ale_metrics = {
        split: compute_feature_effect_metrics(ales_split, ale_groundtruth) for split, ales_split in ales.items()
    }

    # save metrics
    save_fe_results(pdp_metrics, params, "pdp")
    save_fe_results(ale_metrics, params, "ale")

    # aggregate metrics
    pdp_metrics_agg = {
        split: {metric: pdp_metrics[split][metric].mean() for metric in pdp_metrics[split].keys()}
        for split in pdp_metrics.keys()
    }
    ale_metrics_agg = {
        split: {metric: ale_metrics[split][metric].mean() for metric in ale_metrics[split].keys()}
        for split in ale_metrics.keys()
    }

    # save aggregated metrics
    save_fe_aggregated_results(pdp_metrics_agg, engine_effects_results, params, "pdp")
    save_fe_aggregated_results(ale_metrics_agg, engine_effects_results, params, "ale")


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
