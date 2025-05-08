#!/usr/bin/env python3
# evaluate_pipeline.py

import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from Pandev_Model_Test import (
    CONFIG,
    AMINO_ACIDS,
    DEVICE,
    one_hot_encoding_batch,
    b_factor_calculation,
    rc_map_calculation,
    feature_engineering,
    RNN,
    network,
    load_model,
)

def predict_tensile_for_sequence(
    sequence: str,
    protein_type: str,
    b_model: torch.nn.Module,
    p_model: torch.nn.Module,
    all_feature_names: np.ndarray,
) -> float:
    """
    Reimplements the logic of run_prediction_pipeline for a single sample:
      - One-hot encode its sequence if protein_type matches one of the accepted types
      - Predict per-residue B-factors
      - Compute rc‐map
      - Extract features for that type, plus zero‐pad for all other types
      - Concatenate into a (1 x TOTAL_FEATURES) vector
      - Run the property MLP
    """

    # 1) Prepare a dict of sequences exactly as run_prediction_pipeline does:
    seqs = {t: None for t in CONFIG["accepted_spidroin_types"]}
    if sequence and protein_type in seqs:
        seqs[protein_type] = sequence.strip()

    # figure out max length across any one sequence (just one here)
    max_len = max((len(s) for s in seqs.values() if s), default=1)

    # We'll accumulate each‐type feature blocks into this list
    feature_blocks = []
    for sp_type in CONFIG["accepted_spidroin_types"]:
        seq_str = seqs[sp_type]
        # --- if we have a real sequence, compute its block; else all zeros ---
        if seq_str:
            L = len(seq_str)
            # build padded input array
            arr = np.full((1, 1, max_len), "", dtype=object)
            arr[0, 0, :L] = list(seq_str)
            num_files = np.array([1])
            seq_lens  = np.array([[L]])

            # one‐hot, b‐factor, rc‐map
            ohe      = one_hot_encoding_batch(arr, num_files, seq_lens, AMINO_ACIDS)
            bf       = b_factor_calculation(ohe, num_files, seq_lens, b_model, DEVICE, AMINO_ACIDS)
            rcmap    = rc_map_calculation(
                         bf, ohe, arr, num_files, seq_lens,
                         CONFIG["hyperparameters"]["rc_cal_filter"],
                         CONFIG["hyperparameters"]["rc_cal_stride"],
                         AMINO_ACIDS
                       )
            feats, _ = feature_engineering(rcmap, all_feature_names, sp_type, AMINO_ACIDS)

        else:
            # count how many features this type contributed in training
            n_feats = sum(1 for name in all_feature_names if name.startswith(sp_type + "_"))
            feats = np.zeros((1, n_feats))

        feature_blocks.append(feats)

    # 2) concatenate all blocks into (1 x TOTAL_FEATURES)
    X = np.concatenate(feature_blocks, axis=1)
    if X.shape[1] != CONFIG["hyperparameters"]["property_nn_input_features"]:
        # sanity check
        raise RuntimeError(
            f"Feature‐vector length mismatch: got {X.shape[1]}, "
            f"expected {CONFIG['hyperparameters']['property_nn_input_features']}"
        )

    # 3) final model
    x_tensor = torch.from_numpy(X).float().to(DEVICE)
    with torch.no_grad():
        return p_model(x_tensor).item()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--input_csv",
        default="Pandev_Model_Files/PandevModel_evaluation_dataset.csv",
        help="CSV with columns: sequence, protein_type, tensile_strength"
    )
    p.add_argument(
        "--output_csv",
        default="tensile_strength_predictions.csv",
        help="where to save the results"
    )
    args = p.parse_args()

    # 1) Load data
    df = pd.read_csv(args.input_csv)

    # 2) Load models once
    b_model = load_model(
        RNN,
        CONFIG["paths"]["b_factor_model"],
        DEVICE,
        input_size=CONFIG["hyperparameters"]["rnn_input_size"],
        hidden_size1=CONFIG["hyperparameters"]["rnn_hidden_size1"],
        hidden_size2=CONFIG["hyperparameters"]["rnn_hidden_size2"],
        num_layers=CONFIG["hyperparameters"]["rnn_num_layers"],
        seq_len=CONFIG["hyperparameters"]["rnn_seq_len"],
    )
    p_model = load_model(
        network,
        CONFIG["paths"]["property_prediction_model"],
        DEVICE,
    )

    # 3) Load learned feature names
    all_feats = np.load(CONFIG["paths"]["feature_names"], allow_pickle=True)

    # 4) Predict for each row
    preds = []
    for _, row in df.iterrows():
        seq   = row.get("sequence", "")
        ptype = row.get("protein_type", "")
        try:
            pred = predict_tensile_for_sequence(seq, ptype, b_model, p_model, all_feats)
        except Exception as e:
            print(f"Error on row type={ptype!r}, len={len(seq)}: {e}")
            pred = float("nan")
        preds.append(pred)
    df["predicted_tensile_strength"] = preds

    # 5) Drop any rows with NaNs in actual or predicted
    n_total = len(df)
    df_valid = df.dropna(subset=["tensile_strength", "predicted_tensile_strength"])
    n_valid = len(df_valid)
    print(f"Evaluating on {n_valid}/{n_total} rows (dropped {n_total-n_valid} with NaNs)")

    # 6) Compute and print metrics
    mse = mean_squared_error(df_valid["tensile_strength"],
                             df_valid["predicted_tensile_strength"])
    mae = mean_absolute_error(df_valid["tensile_strength"],
                             df_valid["predicted_tensile_strength"])
    r2  = r2_score(df_valid["tensile_strength"],
                   df_valid["predicted_tensile_strength"])
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²:  {r2:.4f}")

    # 7) Save results
    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
