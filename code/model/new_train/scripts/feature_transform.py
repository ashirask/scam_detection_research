import numpy as np


def fit_feature_transform(X_train, epsilon=1e-4, min_mad=1e-6):
    """Learn a per-column log-or-zscore transform from TRAINING data only.

    Non-negative columns -> log(x + epsilon).
    Columns with any negative value -> robust z-standardization:
        (x - median) / (1.4826 * MAD)
    using median/MAD (not mean/std) so extreme bot values don't distort the fit.

    Returns a dict: {column_name: {"method": "log"|"zscore", ...params}}
    Columns that are entirely NaN/empty in the training split are skipped
    (left untouched by apply_feature_transform, so downstream imputers still
    see them as NaN and handle them as before).
    """
    transform_params = {}
    for col in X_train.select_dtypes(include=[np.number]).columns:
        col_data = X_train[col].replace([np.inf, -np.inf], np.nan).dropna()
        if col_data.empty:
            continue

        has_negative = (col_data < 0).any()
        if not has_negative:
            transform_params[col] = {"method": "log", "epsilon": epsilon}
        else:
            median = col_data.median()
            mad = (col_data - median).abs().median() * 1.4826
            if mad < min_mad:
                # near-zero variance column: fall back to sample std to avoid
                # dividing by ~0
                mad = col_data.std()
                if not np.isfinite(mad) or mad < min_mad:
                    mad = 1.0
                print(f"  [warn] {col}: near-zero MAD, using fallback scale")
            transform_params[col] = {"method": "zscore", "median": median, "mad": mad}
    return transform_params


def apply_feature_transform(X, transform_params):
    """Apply an already-fitted transform (see fit_feature_transform) to any split."""
    X_out = X.replace([np.inf, -np.inf], np.nan).copy()
    for col, params in transform_params.items():
        if col not in X_out.columns:
            continue
        if params["method"] == "log":
            # clip(lower=0) guards against a handful of negative values leaking
            # into a column that was judged non-negative on the train split
            X_out[col] = np.log(X_out[col].clip(lower=0) + params["epsilon"])
        else:  # zscore
            X_out[col] = (X_out[col] - params["median"]) / params["mad"]
    return X_out


