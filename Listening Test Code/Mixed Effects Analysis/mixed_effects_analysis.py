#!/usr/bin/env python3
"""Run the final Text2Groove mixed-effects analysis used in the report."""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import patsy
from scipy import stats
import statsmodels.formula.api as smf

METHODS = ["baseline", "proposed", "nearest_neighbour", "quantized"]
GENRES = ["funk", "hiphop", "jazz", "rock"]
FEELS = ["tight", "loose", "swung"]
DYNAMICS = ["medium", "soft", "loud"]
MARKERS = ["audio_1", "audio_2", "audio_3", "audio_4"]
OUTCOMES = ["prompt_match", "human_likeness"]
FACTOR_LEVELS = {"genre": GENRES, "feel": FEELS, "dynamic": DYNAMICS}
EXPECTED_RESULT_COLUMNS = [
    "session_test_id",
    "age",
    "gender",
    "musical_experience_years",
    "trial_id",
    "stimuli_rating",
    "stimuli",
    "rating_time",
    "genre",
    "feel",
    "dynamic",
    "method",
]
BASE_FORMULA = (
    "rating ~ C(method, Treatment(reference='baseline'))"
    " + C(genre, Treatment(reference='funk'))"
    " + C(feel, Treatment(reference='tight'))"
    " + C(dynamic, Treatment(reference='medium'))"
    " + trial_order_z"
    " + C(marker, Treatment(reference='audio_1'))"
)


@dataclass
class FittedModel:
    outcome: str
    specification: str
    formula: str
    result: object
    warnings: list[str]


def participant_from_result_path(path: Path) -> str:
    match = re.fullmatch(r"participant_(\d{3})_results\.csv", path.name)
    if match is None:
        raise ValueError(f"Unexpected participant result path: {path}")
    return match.group(1)


def load_analysis_data(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    result_files = sorted(data_dir.glob("participant_*_results.csv"))
    if not result_files:
        raise FileNotFoundError(f"No participant result files found under {data_dir}")

    frames: list[pd.DataFrame] = []
    key_columns = [
        "trial_id",
        "stimulus_id",
        "pattern_id",
        "prompt_id",
        "prompt_text",
        "key_method",
        "filename",
    ]
    for result_file in result_files:
        participant_id = participant_from_result_path(result_file)
        ratings = pd.read_csv(result_file, dtype={"session_test_id": "string"})
        if list(ratings.columns) != EXPECTED_RESULT_COLUMNS:
            raise ValueError(f"Unexpected result columns in {result_file}")
        if len(ratings) != 96:
            raise ValueError(
                f"Participant {participant_id} has {len(ratings)} rows; expected 96"
            )
        split = ratings["stimuli"].str.extract(
            r"^(?P<stimulus_id>audio_[1-4])__(?P<outcome>prompt_match|human_likeness)$"
        )
        if split.isna().any().any():
            raise ValueError(f"Unrecognized stimulus response name in {result_file}")
        ratings = pd.concat([ratings, split], axis=1)
        ratings["participant_id"] = participant_id

        key_file = data_dir / f"participant_{participant_id}_key.csv"
        if not key_file.exists():
            raise FileNotFoundError(f"Missing private decoding key: {key_file}")
        key = pd.read_csv(
            key_file,
            dtype={"page_id": "string", "pattern_id": "string", "prompt_id": "string"},
        ).rename(columns={"page_id": "trial_id", "method": "key_method"})
        ratings = ratings.merge(
            key[key_columns],
            on=["trial_id", "stimulus_id"],
            how="left",
            validate="many_to_one",
        )
        frames.append(ratings)

    data = pd.concat(frames, ignore_index=True)
    if data[key_columns].isna().any().any():
        raise ValueError("At least one result row could not be decoded using its key")
    if not (data["method"] == data["key_method"]).all():
        raise ValueError("Result method metadata does not agree with the decoding key")

    data["rating"] = pd.to_numeric(data["stimuli_rating"], errors="raise")
    if not data["rating"].between(0, 100).all():
        raise ValueError("At least one rating is outside the expected 0-100 range")
    data["trial_order"] = data["trial_id"].str.extract(r"(\d+)$").astype(int)
    order_mean = float(data["trial_order"].mean())
    order_sd = float(data["trial_order"].std(ddof=0))
    data["trial_order_z"] = (data["trial_order"] - order_mean) / order_sd
    data["trial_uid"] = data["participant_id"] + "__" + data["trial_id"]
    data["all_group"] = "all_observations"
    data["method"] = pd.Categorical(data["method"], METHODS, ordered=False)
    data["genre"] = pd.Categorical(data["genre"], GENRES, ordered=False)
    data["feel"] = pd.Categorical(data["feel"], FEELS, ordered=False)
    data["dynamic"] = pd.Categorical(data["dynamic"], DYNAMICS, ordered=False)
    data["marker"] = pd.Categorical(data["stimulus_id"], MARKERS, ordered=False)

    expected_per_outcome = data["participant_id"].nunique() * 12 * 4
    outcome_counts = data["outcome"].value_counts().to_dict()
    if any(outcome_counts.get(outcome, 0) != expected_per_outcome for outcome in OUTCOMES):
        raise ValueError(f"Outcome row counts are incomplete: {outcome_counts}")

    trial_table = data[
        [
            "participant_id",
            "trial_id",
            "pattern_id",
            "prompt_id",
            "genre",
            "feel",
            "dynamic",
        ]
    ].drop_duplicates()
    prompt_repetitions = trial_table.groupby("prompt_id", observed=True).size()
    metadata = {
        "data_directory": str(data_dir),
        "participants": sorted(data["participant_id"].unique().tolist()),
        "participant_count": int(data["participant_id"].nunique()),
        "rating_rows": int(len(data)),
        "rows_per_outcome": {key: int(value) for key, value in outcome_counts.items()},
        "trial_count": int(len(trial_table)),
        "pattern_count": int(trial_table["pattern_id"].nunique()),
        "prompt_count": int(trial_table["prompt_id"].nunique()),
        "prompts_observed_once": int((prompt_repetitions == 1).sum()),
        "trial_order_mean": order_mean,
        "trial_order_population_sd": order_sd,
    }
    return data, metadata


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(values[valid])]
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        candidate = (total - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted


def average_design_vector(
    model: FittedModel, data: pd.DataFrame, **overrides: str
) -> np.ndarray:
    subset = data.loc[data["outcome"] == model.outcome].copy()
    for column, value in overrides.items():
        subset[column] = value
    design = patsy.build_design_matrices(
        [model.result.model.data.design_info], subset, return_type="dataframe"
    )[0]
    return np.asarray(design, dtype=float).mean(axis=0)


def linear_estimate(model: FittedModel, vector: np.ndarray) -> dict[str, float]:
    names = list(model.result.fe_params.index)
    beta = model.result.fe_params.to_numpy()
    covariance = model.result.cov_params().loc[names, names].to_numpy()
    estimate = float(vector @ beta)
    variance = float(vector @ covariance @ vector)
    standard_error = math.sqrt(max(variance, 0.0))
    z_value = estimate / standard_error if standard_error > 0 else math.nan
    p_value = 2 * stats.norm.sf(abs(z_value)) if math.isfinite(z_value) else math.nan
    return {
        "estimate": estimate,
        "std_error": standard_error,
        "z_value": z_value,
        "p_value": p_value,
        "ci95_low": estimate - 1.96 * standard_error,
        "ci95_high": estimate + 1.96 * standard_error,
    }


def marginal_method_tables(
    model: FittedModel, data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return overall method means and the three planned proposed contrasts."""
    vectors = {
        method: average_design_vector(model, data, method=method)
        for method in METHODS
    }
    means_rows = []
    for method, vector in vectors.items():
        row = linear_estimate(model, vector)
        row.update(
            {
                "outcome": model.outcome,
                "specification": model.specification,
                "method": method,
            }
        )
        means_rows.append(row)

    contrast_rows = []
    for comparator in ["baseline", "nearest_neighbour", "quantized"]:
        row = linear_estimate(model, vectors["proposed"] - vectors[comparator])
        row.update(
            {
                "outcome": model.outcome,
                "specification": model.specification,
                "contrast": f"proposed - {comparator}",
            }
        )
        contrast_rows.append(row)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["p_value_holm"] = holm_adjust(contrasts["p_value"])
    return pd.DataFrame(means_rows), contrasts


def interaction_tables(
    model: FittedModel, data: pd.DataFrame, factor: str, levels: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    means_rows = []
    contrast_rows = []
    for level in levels:
        vectors = {
            method: average_design_vector(model, data, method=method, **{factor: level})
            for method in METHODS
        }
        for method, vector in vectors.items():
            row = linear_estimate(model, vector)
            row.update(
                {
                    "outcome": model.outcome,
                    "factor": factor,
                    "level": level,
                    "method": method,
                }
            )
            means_rows.append(row)
        for comparator in ["baseline", "nearest_neighbour", "quantized"]:
            row = linear_estimate(model, vectors["proposed"] - vectors[comparator])
            row.update(
                {
                    "outcome": model.outcome,
                    "factor": factor,
                    "level": level,
                    "contrast": f"proposed - {comparator}",
                }
            )
            contrast_rows.append(row)

    contrasts = pd.DataFrame(contrast_rows)
    contrasts["p_value_holm"] = np.nan
    for (_, level), indices in contrasts.groupby(["outcome", "level"]).groups.items():
        contrasts.loc[indices, "p_value_holm"] = holm_adjust(
            contrasts.loc[indices, "p_value"]
        )
    return pd.DataFrame(means_rows), contrasts


def likelihood_ratio_test(base: FittedModel, interaction: FittedModel) -> dict:
    statistic = max(0.0, 2 * (interaction.result.llf - base.result.llf))
    degrees_freedom = len(interaction.result.fe_params) - len(base.result.fe_params)
    return {
        "outcome": base.outcome,
        "comparison": f"base vs {interaction.specification}",
        "lr_statistic": statistic,
        "df": degrees_freedom,
        "p_value": stats.chi2.sf(statistic, degrees_freedom),
        "base_aic": base.result.aic,
        "interaction_aic": interaction.result.aic,
    }


def model_diagnostic_row(model: FittedModel, data: pd.DataFrame) -> dict:
    result = model.result
    subset = data.loc[data["outcome"] == model.outcome]
    residuals = np.asarray(result.resid, dtype=float)
    fitted = np.asarray(result.fittedvalues, dtype=float)
    jarque_bera = stats.jarque_bera(residuals)
    return {
        "outcome": model.outcome,
        "specification": model.specification,
        "converged": bool(result.converged),
        "n_observations": int(result.nobs),
        "fixed_effect_count": int(len(result.fe_params)),
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "residual_variance": float(result.scale),
        "residual_sd": math.sqrt(float(result.scale)),
        "residual_mean": float(residuals.mean()),
        "residual_skew": float(stats.skew(residuals)),
        "residual_excess_kurtosis": float(stats.kurtosis(residuals)),
        "jarque_bera_p": float(jarque_bera.pvalue),
        "observed_fitted_correlation": float(
            np.corrcoef(subset["rating"], fitted)[0, 1]
        ),
        "ratings_at_0": int((subset["rating"] == 0).sum()),
        "ratings_at_100": int((subset["rating"] == 100).sum()),
        "warnings": " | ".join(model.warnings),
    }


def variance_component_table(model: FittedModel) -> pd.DataFrame:
    names = list(model.result.model.exog_vc.names)
    values = np.asarray(model.result.vcomp, dtype=float)
    return pd.DataFrame(
        {
            "outcome": model.outcome,
            "specification": model.specification,
            "component": names,
            "variance": values,
            "sd": np.sqrt(np.maximum(values, 0.0)),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "Flat directory containing participant_*_results.csv, matching key "
            "files. Defaults to the directory containing this script."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated files; defaults to the flat data directory or project results directory.",
    )
    return parser.parse_args()


def formula_for(factor: str | None = None) -> str:
    if factor is None:
        return BASE_FORMULA
    references = {"genre": "funk", "feel": "tight", "dynamic": "medium"}
    return (
        BASE_FORMULA
        + f" + C(method, Treatment(reference='baseline')):"
        + f"C({factor}, Treatment(reference='{references[factor]}'))"
    )


def fit(data: pd.DataFrame, outcome: str, specification: str, formula: str) -> FittedModel:
    subset = data.loc[data["outcome"] == outcome].copy()
    model = smf.mixedlm(
        formula,
        subset,
        groups="all_group",
        re_formula="0",
        vc_formula={
            "participant": "0 + C(participant_id)",
            "pattern": "0 + C(pattern_id)",
        },
    )
    candidates = []
    messages: list[str] = []
    for optimizer in ["lbfgs", "powell"]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit(
                reml=False,
                method=optimizer,
                maxiter=5000,
                full_output=True,
                disp=False,
            )
        candidates.append((result, optimizer))
        messages.extend(f"{optimizer}: {item.message}" for item in caught)
    converged = [item for item in candidates if item[0].converged]
    result, optimizer = max(converged or candidates, key=lambda item: item[0].llf)
    messages.append(f"selected optimizer: {optimizer}")
    return FittedModel(outcome, specification, formula, result, sorted(set(messages)))


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir.resolve() if args.data_dir else script_dir
    output = args.output_dir.resolve() if args.output_dir else data_dir
    output.mkdir(parents=True, exist_ok=True)
    data, metadata = load_analysis_data(data_dir)

    models: dict[tuple[str, str], FittedModel] = {}
    for outcome in OUTCOMES:
        models[(outcome, "base")] = fit(data, outcome, "base", formula_for())
        for factor in FACTOR_LEVELS:
            print(f"Fitting {outcome} method-by-{factor}", flush=True)
            models[(outcome, factor)] = fit(
                data,
                outcome,
                f"method_by_{factor}",
                formula_for(factor),
            )

    tests = []
    means = []
    contrasts = []
    method_means = []
    method_contrasts = []
    for outcome in OUTCOMES:
        outcome_means, outcome_contrasts = marginal_method_tables(
            models[(outcome, "base")], data
        )
        method_means.append(outcome_means)
        method_contrasts.append(outcome_contrasts)
        for factor, levels in FACTOR_LEVELS.items():
            interaction = models[(outcome, factor)]
            row = likelihood_ratio_test(models[(outcome, "base")], interaction)
            row["factor"] = factor
            tests.append(row)
            factor_means, factor_contrasts = interaction_tables(
                interaction, data, factor, levels
            )
            means.append(factor_means)
            contrasts.append(factor_contrasts)

    tests_frame = pd.DataFrame(tests)
    tests_frame["p_value_holm"] = np.nan
    for outcome, indices in tests_frame.groupby("outcome").groups.items():
        tests_frame.loc[indices, "p_value_holm"] = holm_adjust(
            tests_frame.loc[indices, "p_value"]
        )
    tests_frame["significant_05_holm"] = tests_frame["p_value_holm"] < 0.05
    method_means_frame = pd.concat(method_means, ignore_index=True)
    method_contrasts_frame = pd.concat(method_contrasts, ignore_index=True)
    means_frame = pd.concat(means, ignore_index=True)
    contrasts_frame = pd.concat(contrasts, ignore_index=True)

    diagnostics = pd.DataFrame(
        [model_diagnostic_row(model, data) for model in models.values()]
    )
    variances = pd.concat(
        [variance_component_table(model) for model in models.values()],
        ignore_index=True,
    )

    tests_frame.to_csv(output / "mixed_effects_interaction_tests.csv", index=False)
    method_means_frame.to_csv(
        output / "mixed_effects_method_estimated_means.csv", index=False
    )
    method_contrasts_frame.to_csv(
        output / "mixed_effects_method_contrasts.csv", index=False
    )
    means_frame.to_csv(
        output / "mixed_effects_interaction_estimated_means.csv", index=False
    )
    contrasts_frame.to_csv(
        output / "mixed_effects_interaction_contrasts.csv", index=False
    )
    diagnostics.to_csv(output / "mixed_effects_model_diagnostics.csv", index=False)
    variances.to_csv(output / "mixed_effects_variance_components.csv", index=False)

    summaries = []
    for model in models.values():
        summaries.extend(
            [
                f"OUTCOME: {model.outcome}",
                f"SPECIFICATION: {model.specification}",
                f"FORMULA: {model.formula}",
                f"WARNINGS: {' | '.join(model.warnings) if model.warnings else 'None'}",
                str(model.result.summary()),
                "",
            ]
        )
    (output / "mixed_effects_model_summaries.txt").write_text(
        "\n".join(summaries), encoding="utf-8"
    )
    metadata.update(
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "analysis": "final Text2Groove mixed-effects model",
            "random_effects": ["participant", "pattern"],
            "removed_random_effect": "participant_trial",
            "estimation": "maximum likelihood",
            "interaction_test": "likelihood-ratio test against reduced base model",
            "interaction_p_adjustment": "Holm within outcome across three factors",
            "conditional_contrast_p_adjustment": (
                "Holm within outcome and factor level across proposed-vs-comparator contrasts"
            ),
        }
    )
    (output / "mixed_effects_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Mixed-effects analysis complete: {output}")


if __name__ == "__main__":
    main()
