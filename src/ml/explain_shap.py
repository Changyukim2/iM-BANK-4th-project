"""Create SHAP explanations for the main online-share baseline model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from train_baseline import BaselineConfig, build_feature_lists, load_column_config, validate_no_leakage


DEFAULT_PROCESSED_DIR = Path("outputs/processed")
DEFAULT_MODEL_PATH = Path("outputs/models/baseline/best_baseline_model.joblib")
DEFAULT_OUTPUT_DIR = Path("outputs/models/baseline/shap")


def configure_matplotlib_font() -> None:
    for font_name in ["AppleGothic", "NanumGothic", "Malgun Gothic"]:
        try:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue


def load_feature_data(processed_dir: Path, split: str, sample_size: int, random_seed: int) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    column_config = load_column_config(processed_dir)
    config = BaselineConfig()
    features, _, _ = build_feature_lists(column_config, config)
    validate_no_leakage(features, column_config, config.target)

    path = processed_dir / f"dx_panel_{split}_model_input.csv.gz"
    df = pd.read_csv(path)
    if sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_seed).sort_index()
    return df[features], df[config.target].astype(int), features


def get_transformed_feature_names(model, features: list[str]) -> list[str]:
    preprocessor = model.named_steps["preprocess"]
    try:
        return preprocessor.get_feature_names_out(features).tolist()
    except TypeError:
        return preprocessor.get_feature_names_out().tolist()


def group_encoded_feature_name(encoded_name: str, categorical_features: list[str]) -> str:
    if encoded_name.startswith("num__"):
        return encoded_name.replace("num__", "", 1)
    if encoded_name.startswith("cat__"):
        raw = encoded_name.replace("cat__", "", 1)
        for col in categorical_features:
            if raw == col or raw.startswith(f"{col}_"):
                return col
        return raw
    return encoded_name


def calculate_shap(model, x: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    x_transformed = preprocessor.transform(x)
    if hasattr(x_transformed, "toarray"):
        x_transformed = x_transformed.toarray()

    encoded_names = get_transformed_feature_names(model, x.columns.tolist())
    categorical_features = [
        col
        for col in x.columns
        if col in preprocessor.named_transformers_["cat"].feature_names_in_
    ]
    grouped_names = [group_encoded_feature_name(name, categorical_features) for name in encoded_names]

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(x_transformed, check_additivity=False)
    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]
    elif isinstance(shap_values, list):
        shap_array = np.asarray(shap_values[1])

    return shap_array, encoded_names, grouped_names


def summarize_shap(shap_array: np.ndarray, encoded_names: list[str], grouped_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoded_importance = pd.DataFrame(
        {
            "feature": encoded_names,
            "mean_abs_shap": np.abs(shap_array).mean(axis=0),
            "mean_shap": shap_array.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    grouped = (
        pd.DataFrame(
            {
                "feature": grouped_names,
                "mean_abs_shap_component": np.abs(shap_array).mean(axis=0),
            }
        )
        .groupby("feature", as_index=False)["mean_abs_shap_component"]
        .sum()
        .rename(columns={"mean_abs_shap_component": "mean_abs_shap"})
        .sort_values("mean_abs_shap", ascending=False)
    )
    return encoded_importance, grouped


def save_plot(top_grouped: pd.DataFrame, output_path: Path) -> None:
    plot_df = top_grouped.sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(9, 7))
    plt.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#2f6f8f")
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title("Top 20 SHAP Features - Online Share Up Baseline")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_report(
    output_dir: Path,
    top_grouped: pd.DataFrame,
    top_encoded: pd.DataFrame,
    split: str,
    sample_size: int,
    model_path: Path,
) -> None:
    grouped_rows = "\n".join(
        f"| {idx + 1} | `{row.feature}` | {row.mean_abs_shap:.6f} |"
        for idx, row in enumerate(top_grouped.itertuples(index=False))
    )
    encoded_rows = "\n".join(
        f"| {idx + 1} | `{row.feature}` | {row.mean_abs_shap:.6f} | {row.mean_shap:.6f} |"
        for idx, row in enumerate(top_encoded.itertuples(index=False))
    )
    text = f"""# SHAP 상위 변수 리포트

## 기준

- 모델: `{model_path}`
- 타깃: `target_online_share_up_5pp`
- 의미: 다음 달 온라인 채널 점유율이 현재 월보다 5%p 이상 상승할지 예측
- 설명 데이터: `{split}` split에서 최대 `{sample_size}`개 샘플
- 중요도 기준: 평균 절대 SHAP 값

## 상위 20개 변수

범주형 one-hot 컬럼은 원래 변수 단위로 합산했습니다.

| rank | feature | mean_abs_shap |
|---:|---|---:|
{grouped_rows}

## 참고: 인코딩된 컬럼 기준 상위 20개

| rank | encoded_feature | mean_abs_shap | mean_shap |
|---:|---|---:|---:|
{encoded_rows}

## 해석 주의

- SHAP 값은 모델 예측에 대한 기여도입니다. 인과관계를 의미하지는 않습니다.
- `mean_abs_shap`이 클수록 해당 변수가 모델 예측을 많이 흔든다는 뜻입니다.
- 이 결과는 기존 메인 타깃 `target_online_share_up_5pp` 기준이며, Y1 회귀 타깃 기준 결과가 아닙니다.
"""
    (output_dir / "shap_top20_report.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain best baseline model with SHAP.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib_font()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(args.model_path)
    x, y, _ = load_feature_data(args.processed_dir, args.split, args.sample_size, args.random_seed)
    shap_array, encoded_names, grouped_names = calculate_shap(model, x)
    encoded_importance, grouped_importance = summarize_shap(shap_array, encoded_names, grouped_names)

    top_grouped = grouped_importance.head(20).reset_index(drop=True)
    top_encoded = encoded_importance.head(20).reset_index(drop=True)
    top_grouped.to_csv(args.output_dir / "shap_top20_grouped.csv", index=False, encoding="utf-8-sig")
    top_encoded.to_csv(args.output_dir / "shap_top20_encoded.csv", index=False, encoding="utf-8-sig")
    save_plot(top_grouped, args.output_dir / "shap_top20_grouped.png")
    write_report(args.output_dir, top_grouped, top_encoded, args.split, len(x), args.model_path)

    summary = {
        "target": "target_online_share_up_5pp",
        "split": args.split,
        "sample_size": int(len(x)),
        "positive_rate_in_sample": float(y.mean()),
        "top20_grouped": top_grouped.to_dict(orient="records"),
    }
    (args.output_dir / "shap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
