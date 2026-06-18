"""Tune boosted models for the no-direct-online ablation setting.

Target remains target_online_share_up_5pp. The feature set excludes:
- future columns: next_*
- target columns: target_*
- key/split columns
- eda_priority_score
- direct online amount/share/count features
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train_baseline import (
    BaselineConfig,
    build_feature_lists,
    evaluate_split,
    load_column_config,
    load_split_data,
    make_feature_audit,
    validate_no_leakage,
)


DEFAULT_PROCESSED_DIR = Path("outputs/processed")
DEFAULT_OUTPUT_DIR = Path("outputs/models/tuned_no_direct_online")
DEFAULT_STRICT_OUTPUT_DIR = Path("outputs/models/tuned_strict_no_online")
DEFAULT_BASIC_OUTPUT_DIR = Path("outputs/models/tuned_basic_only")
RANDOM_SEED = 42


@dataclass(frozen=True)
class TuneConfig:
    target: str = "target_online_share_up_5pp"
    random_seed: int = RANDOM_SEED
    top_k_rates: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)


def positive_class_weight(y: pd.Series) -> float:
    positives = max(int(y.sum()), 1)
    negatives = max(int((1 - y).sum()), 1)
    return negatives / positives


def fill_catboost_inputs(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int]]:
    features = categorical + numeric
    x_train = train[features].copy()
    x_valid = valid[features].copy()
    x_test = test[features].copy()
    for df in [x_train, x_valid, x_test]:
        for col in categorical:
            df[col] = df[col].astype("object").fillna("미상").astype(str)
        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    cat_idx = [features.index(col) for col in categorical]
    return x_train, x_valid, x_test, cat_idx


def make_xgb_pipeline(params: dict[str, Any], categorical: list[str], numeric: list[str]) -> Pipeline:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", XGBClassifier(**params)),
        ]
    )


def predict_score(model: Any, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def evaluate_model(
    model: Any,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    test_x: pd.DataFrame,
    test_y: pd.Series,
    rates: tuple[float, ...],
) -> dict[str, dict[str, float]]:
    # Keep threshold fixed from validation F1 for comparability with baseline.
    from train_baseline import best_f1_threshold

    valid_score = predict_score(model, valid_x)
    threshold, _ = best_f1_threshold(valid_y.to_numpy(), valid_score)
    return {
        "train": evaluate_split(model, train_x, train_y, threshold, rates),
        "valid": evaluate_split(model, valid_x, valid_y, threshold, rates),
        "test": evaluate_split(model, test_x, test_y, threshold, rates),
    }


def tune_catboost(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    config: TuneConfig,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, Any]]:
    x_train, x_valid, x_test, cat_idx = fill_catboost_inputs(train, valid, test, categorical, numeric)
    y_train = train[config.target].astype(int)
    y_valid = valid[config.target].astype(int)
    y_test = test[config.target].astype(int)
    scale_pos_weight = positive_class_weight(y_train)

    candidates = {
        "catboost_d6_lr005_l2_3": {
            "iterations": 900,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3,
        },
        "catboost_d7_lr004_l2_6": {
            "iterations": 1100,
            "depth": 7,
            "learning_rate": 0.04,
            "l2_leaf_reg": 6,
        },
        "catboost_d8_lr003_l2_8": {
            "iterations": 1300,
            "depth": 8,
            "learning_rate": 0.03,
            "l2_leaf_reg": 8,
        },
        "catboost_d6_lr003_l2_10": {
            "iterations": 1200,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 10,
        },
    }
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    models: dict[str, Any] = {}
    for name, params in candidates.items():
        model = CatBoostClassifier(
            **params,
            loss_function="Logloss",
            eval_metric="PRAUC",
            cat_features=cat_idx,
            scale_pos_weight=scale_pos_weight,
            random_seed=config.random_seed,
            od_type="Iter",
            od_wait=80,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True)
        metrics[name] = evaluate_model(model, x_train, y_train, x_valid, y_valid, x_test, y_test, config.top_k_rates)
        metrics[name]["best_iteration"] = {"value": int(model.get_best_iteration() or params["iterations"])}
        models[name] = model
    return metrics, models


def tune_xgboost(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    config: TuneConfig,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, Any]]:
    features = categorical + numeric
    x_train = train[features]
    y_train = train[config.target].astype(int)
    x_valid = valid[features]
    y_valid = valid[config.target].astype(int)
    x_test = test[features]
    y_test = test[config.target].astype(int)
    scale_pos_weight = positive_class_weight(y_train)

    candidates = {
        "xgboost_d4_lr004": {"max_depth": 4, "learning_rate": 0.04, "n_estimators": 700, "min_child_weight": 30},
        "xgboost_d5_lr003": {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 900, "min_child_weight": 40},
        "xgboost_d6_lr0025": {"max_depth": 6, "learning_rate": 0.025, "n_estimators": 1000, "min_child_weight": 60},
    }
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    models: dict[str, Any] = {}
    for name, params in candidates.items():
        model_params = {
            **params,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 5.0,
            "reg_alpha": 0.1,
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": config.random_seed,
        }
        model = make_xgb_pipeline(model_params, categorical, numeric)
        model.fit(x_train, y_train)
        metrics[name] = evaluate_model(model, x_train, y_train, x_valid, y_valid, x_test, y_test, config.top_k_rates)
        models[name] = model
    return metrics, models


def write_report(
    output_dir: Path,
    metrics: dict[str, dict[str, Any]],
    feature_count: int,
    categorical_count: int,
    numeric_count: int,
    best_name: str,
    config: TuneConfig,
    direct_online_excluded: bool,
    strict_no_online: bool,
    leakage_duplicate_excluded: bool,
    basic_only: bool,
    feature_audit: pd.DataFrame,
) -> None:
    rows = []
    for model_name, split_metrics in metrics.items():
        for split in ["train", "valid", "test"]:
            m = split_metrics[split]
            rows.append({"model": model_name, "split": split, **m})
    pd.DataFrame(rows).to_csv(output_dir / "tuned_metrics.csv", index=False, encoding="utf-8-sig")
    (output_dir / "tuned_metrics.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "feature_count": feature_count,
                "categorical_feature_count": categorical_count,
                "numeric_feature_count": numeric_count,
                "excluded_direct_online_features": direct_online_excluded,
                "excluded_strict_online_features": strict_no_online,
                "excluded_leakage_duplicate_candidates": leakage_duplicate_excluded,
                "excluded_reason_counts": (
                    feature_audit[feature_audit["status"].eq("excluded")]["reason"]
                    .value_counts()
                    .sort_index()
                    .to_dict()
                ),
                "basic_only_features": basic_only,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    feature_audit.to_csv(output_dir / "feature_audit.csv", index=False, encoding="utf-8-sig")

    table = "\n".join(
        f"| {name} | {m['valid']['pr_auc']:.4f} | {m['valid']['roc_auc']:.4f} | "
        f"{m['test']['pr_auc']:.4f} | {m['test']['roc_auc']:.4f} | "
        f"{m['test']['top_10pct_recall']:.4f} | {m['test']['top_10pct_lift']:.4f} |"
        for name, m in sorted(metrics.items(), key=lambda item: item[1]["valid"]["pr_auc"], reverse=True)
    )
    best = metrics[best_name]
    text = f"""# No Direct Online Tuned Model Report

## 누수 방지

- 타깃: `target_online_share_up_5pp`
- 기준: 현재 월 `t` 정보로 다음 달 `t+1` 온라인 점유율 5%p 이상 상승 여부 예측
- 분할: 기존 시간 분할 train/valid/test 유지
- 제외: `next_*`, `target_*`, `기준년월`, `법인ID`, `split`, `eda_priority_score`
- 온라인 직접 변수 제외 여부: `{direct_online_excluded}`
- strict no-online 여부: `{strict_no_online}`
- 누수/중복 1차 후보 동시 제거 여부: `{leakage_duplicate_excluded}`
- basic only 여부: `{basic_only}`
- 피처 감사표: `feature_audit.csv`

## 피처 수

- 전체 피처: {feature_count}
- 범주형 피처: {categorical_count}
- 숫자형 피처: {numeric_count}

## 튜닝 결과

| model | Valid PR-AUC | Valid ROC-AUC | Test PR-AUC | Test ROC-AUC | Test Top 10% Recall | Test Top 10% Lift |
|---|---:|---:|---:|---:|---:|---:|
{table}

## 최고 모델

- Valid PR-AUC 기준 최고 모델: `{best_name}`
- Valid PR-AUC: {best['valid']['pr_auc']:.4f}
- Test PR-AUC: {best['test']['pr_auc']:.4f}
- Test Top 10% Recall: {best['test']['top_10pct_recall']:.4f}
- Test Top 10% Lift: {best['test']['top_10pct_lift']:.4f}

## 해석

온라인 직접 변수를 제외한 조건에서 valid PR-AUC 0.65 달성 여부를 확인하기 위한 튜닝 결과입니다.
성능이 0.65에 미달한다면 현재 제외 조건에서는 온라인 직접 변수 없이 확보할 수 있는 신호가 제한적이라는 근거로 해석합니다.
"""
    (output_dir / "tuned_report.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune no-direct-online boosted models.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", choices=["catboost", "xgboost"], default=["catboost", "xgboost"])
    parser.add_argument("--strict-no-online", action="store_true")
    parser.add_argument("--exclude-leakage-duplicate-candidates", action="store_true")
    parser.add_argument("--basic-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        if args.basic_only:
            args.output_dir = DEFAULT_BASIC_OUTPUT_DIR
        elif args.strict_no_online:
            args.output_dir = DEFAULT_STRICT_OUTPUT_DIR
        else:
            args.output_dir = DEFAULT_OUTPUT_DIR
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = TuneConfig()
    column_config = load_column_config(args.processed_dir)
    baseline_config = BaselineConfig(
        exclude_direct_online_features=not args.basic_only,
        exclude_strict_online_features=args.strict_no_online,
        exclude_leakage_duplicate_candidates=args.exclude_leakage_duplicate_candidates,
        basic_only_features=args.basic_only,
    )
    features, categorical, numeric = build_feature_lists(column_config, baseline_config)
    validate_no_leakage(features, column_config, config.target)
    feature_audit = make_feature_audit(column_config, baseline_config)

    train, valid, test = load_split_data(args.processed_dir)
    all_metrics: dict[str, dict[str, Any]] = {}
    all_models: dict[str, Any] = {}

    if "catboost" in args.models:
        metrics, models = tune_catboost(train, valid, test, categorical, numeric, config)
        all_metrics.update(metrics)
        all_models.update(models)
    if "xgboost" in args.models:
        metrics, models = tune_xgboost(train, valid, test, categorical, numeric, config)
        all_metrics.update(metrics)
        all_models.update(models)

    best_name = max(all_metrics, key=lambda name: all_metrics[name]["valid"]["pr_auc"])
    joblib.dump(all_models[best_name], args.output_dir / "best_tuned_model.joblib")
    write_report(
        args.output_dir,
        all_metrics,
        len(features),
        len(categorical),
        len(numeric),
        best_name,
        config,
        not args.basic_only,
        args.strict_no_online,
        args.exclude_leakage_duplicate_candidates,
        args.basic_only,
        feature_audit,
    )
    print(json.dumps({"best_model": best_name, "best_valid_pr_auc": all_metrics[best_name]["valid"]["pr_auc"], "metrics": all_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
