"""Create SHAP plots for the tuned no-direct-online CatBoost model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool

from train_baseline import (
    BaselineConfig,
    build_feature_lists,
    load_column_config,
    load_split_data,
    validate_no_leakage,
)


DEFAULT_PROCESSED_DIR = Path("outputs/processed")
DEFAULT_MODEL_PATH = Path("outputs/models/tuned_no_direct_online/best_tuned_model.joblib")
DEFAULT_OUTPUT_DIR = Path("outputs/models/tuned_no_direct_online/shap")
DEFAULT_STRICT_MODEL_PATH = Path("outputs/models/tuned_strict_no_online/best_tuned_model.joblib")
DEFAULT_STRICT_OUTPUT_DIR = Path("outputs/models/tuned_strict_no_online/shap")


def configure_matplotlib_font() -> None:
    plt.switch_backend("Agg")
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False


def load_sample(processed_dir: Path, split: str, sample_size: int, random_seed: int, strict_no_online: bool):
    column_config = load_column_config(processed_dir)
    config = BaselineConfig(
        exclude_direct_online_features=True,
        exclude_strict_online_features=strict_no_online,
    )
    features, categorical, numeric = build_feature_lists(column_config, config)
    validate_no_leakage(features, column_config, config.target)

    split_data = dict(zip(["train", "valid", "test"], load_split_data(processed_dir)))
    df = split_data[split].copy()
    if sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_seed).sort_index()

    x = df[features].copy()
    for col in categorical:
        x[col] = x[col].astype("object").fillna("미상").astype(str)
    for col in numeric:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    cat_idx = [features.index(col) for col in categorical]
    y = df[config.target].astype(int)
    return x, y, features, categorical, cat_idx


def save_plot(top_df: pd.DataFrame, output_path: Path, title: str, value_col: str) -> None:
    plot_df = top_df.sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(9, 7))
    plt.barh(plot_df["feature"], plot_df[value_col], color="#396b54")
    plt.xlabel(value_col)
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def summarize_effect_direction(x: pd.DataFrame, shap_array: np.ndarray, features: list[str], top_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    shap_df = pd.DataFrame(shap_array, columns=features, index=x.index)
    for feature in top_df["feature"]:
        values = pd.to_numeric(x[feature], errors="coerce")
        shap_values = shap_df[feature]
        unique_values = values.dropna().nunique()
        if unique_values <= 1:
            continue
        if unique_values <= 2:
            low_mask = values == values.min()
            high_mask = values == values.max()
        else:
            low_cut = values.quantile(0.25)
            high_cut = values.quantile(0.75)
            low_mask = values <= low_cut
            high_mask = values >= high_cut

        low_shap = float(shap_values[low_mask].mean())
        high_shap = float(shap_values[high_mask].mean())
        delta = high_shap - low_shap
        rows.append(
            {
                "feature": feature,
                "low_value_mean": float(values[low_mask].mean()),
                "high_value_mean": float(values[high_mask].mean()),
                "low_mean_shap": low_shap,
                "high_mean_shap": high_shap,
                "high_minus_low_shap": delta,
                "direction_when_high": "increase_prediction" if delta > 0 else "decrease_prediction",
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    top_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    split: str,
    sample_size: int,
    positive_rate: float,
    model_path: Path,
    strict_no_online: bool,
) -> None:
    rows = "\n".join(
        f"| {idx + 1} | `{row.feature}` | {row.mean_abs_shap:.6f} | {row.mean_shap:.6f} |"
        for idx, row in enumerate(top_df.itertuples(index=False))
    )
    direction_rows = "\n".join(
        f"| `{row.feature}` | {row.low_value_mean:.4f} | {row.high_value_mean:.4f} | "
        f"{row.low_mean_shap:.6f} | {row.high_mean_shap:.6f} | {row.high_minus_low_shap:.6f} | {row.direction_when_high} |"
        for row in direction_df.head(20).itertuples(index=False)
    )
    condition_label = "Strict No Online" if strict_no_online else "No Direct Online"
    caveat = (
        "- 이 결과는 온라인 원본/파생/상호작용 변수를 최대한 제거한 strict no-online tuned CatBoost 모델 기준입니다."
        if strict_no_online
        else "- 이 결과는 온라인 직접 변수를 제외한 tuned CatBoost 모델 기준입니다.\n"
        "- 여기서 온라인 직접 변수 제외는 원본 온라인 금액/건수 중심 제외이며, `online_diversity`, `online_x_grade` 같은 온라인 파생/상호작용 변수는 포함되어 있습니다."
    )
    text = f"""# Tuned {condition_label} SHAP 리포트

## 기준

- 모델: `{model_path}`
- 타깃: `target_online_share_up_5pp`
- 실험 조건: `{condition_label}`
- 설명 데이터: `{split}` split에서 `{sample_size}`개 샘플
- 샘플 양성 비율: {positive_rate:.2%}
- 중요도 기준: 평균 절대 SHAP 값

## 상위 20개 변수

| rank | feature | mean_abs_shap | mean_shap |
|---:|---|---:|---:|
{rows}

## 상위 변수 방향성 요약

각 변수의 낮은 구간과 높은 구간에서 평균 SHAP 값을 비교했습니다.
`high_minus_low_shap`이 양수이면 해당 변수 값이 높을수록 타깃 1 예측을 키우는 방향이고, 음수이면 낮추는 방향입니다.

| feature | low_value_mean | high_value_mean | low_mean_shap | high_mean_shap | high_minus_low_shap | direction_when_high |
|---|---:|---:|---:|---:|---:|---|
{direction_rows}

## 해석 주의

{caveat}
- `mean_abs_shap`이 클수록 모델 예측을 많이 움직인 변수입니다.
- SHAP은 인과관계가 아니라 모델 예측 기여도입니다.
"""
    (output_dir / "shap_top20_report.md").write_text(text, encoding="utf-8")


def save_native_feature_importance(model, features: list[str], output_dir: Path) -> pd.DataFrame:
    values = model.get_feature_importance(type="PredictionValuesChange")
    importance = (
        pd.DataFrame({"feature": features, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    top20 = importance.head(20)
    importance.to_csv(output_dir / "catboost_feature_importance_all.csv", index=False, encoding="utf-8-sig")
    top20.to_csv(output_dir / "catboost_feature_importance_top20.csv", index=False, encoding="utf-8-sig")

    plot_df = top20.sort_values("importance", ascending=True)
    plt.figure(figsize=(9, 7))
    plt.barh(plot_df["feature"], plot_df["importance"], color="#5f6fa8")
    plt.xlabel("PredictionValuesChange")
    plt.ylabel("Feature")
    plt.title("Top 20 CatBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "catboost_feature_importance_top20.png", dpi=180)
    plt.close()
    return top20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain tuned no-direct-online CatBoost model with SHAP.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--strict-no-online", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.strict_no_online:
        if args.model_path == DEFAULT_MODEL_PATH:
            args.model_path = DEFAULT_STRICT_MODEL_PATH
        if args.output_dir == DEFAULT_OUTPUT_DIR:
            args.output_dir = DEFAULT_STRICT_OUTPUT_DIR
    configure_matplotlib_font()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model_path)
    x, y, features, categorical, cat_idx = load_sample(
        args.processed_dir,
        args.split,
        args.sample_size,
        args.random_seed,
        args.strict_no_online,
    )
    native_top20 = save_native_feature_importance(model, features, args.output_dir)
    pool = Pool(x, label=y, cat_features=cat_idx, feature_names=features)
    shap_values = model.get_feature_importance(pool, type="ShapValues")
    shap_array = np.asarray(shap_values)[:, :-1]

    importance = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap": np.abs(shap_array).mean(axis=0),
            "mean_shap": shap_array.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    top20 = importance.head(20).reset_index(drop=True)
    direction = summarize_effect_direction(x, shap_array, features, top20)

    importance.to_csv(args.output_dir / "shap_all_features.csv", index=False, encoding="utf-8-sig")
    top20.to_csv(args.output_dir / "shap_top20.csv", index=False, encoding="utf-8-sig")
    direction.to_csv(args.output_dir / "shap_effect_direction_top20.csv", index=False, encoding="utf-8-sig")
    save_plot(
        top20,
        args.output_dir / "shap_top20.png",
        "Top 20 SHAP Features - Tuned No Direct Online CatBoost",
        "mean_abs_shap",
    )
    write_report(args.output_dir, top20, direction, args.split, len(x), float(y.mean()), args.model_path, args.strict_no_online)

    summary = {
        "target": "target_online_share_up_5pp",
        "condition": "strict_no_online" if args.strict_no_online else "no_direct_online",
        "split": args.split,
        "sample_size": int(len(x)),
        "positive_rate_in_sample": float(y.mean()),
        "native_feature_importance_top20": native_top20.to_dict(orient="records"),
        "top20": top20.to_dict(orient="records"),
        "effect_direction_top20": direction.to_dict(orient="records"),
    }
    (args.output_dir / "shap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
