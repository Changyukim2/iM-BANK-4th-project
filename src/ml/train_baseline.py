"""Train leakage-safe baseline models before AutoML.

This script uses only the preprocessed train/valid/test files and the feature
definition saved by preprocess_dx.py. Future-month columns are explicitly
blocked even if they are present in the data files.
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_SEED = 42
DEFAULT_PROCESSED_DIR = Path("outputs/processed")
DEFAULT_MODEL_DIR = Path("outputs/models/baseline")


@dataclass(frozen=True)
class BaselineConfig:
    target: str = "target_online_share_up_5pp"
    exclude_heuristic_score: bool = True
    exclude_direct_online_features: bool = False
    exclude_strict_online_features: bool = False
    exclude_leakage_duplicate_candidates: bool = False
    basic_only_features: bool = False
    top_k_rates: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    random_seed: int = RANDOM_SEED


ALWAYS_BLOCKED_COLUMNS = {
    "기준년월",
    "법인ID",
    "split",
    "Y_t1",
    "Y_t3",
    "target_online_share_up_5pp",
    "target_new_online_activation",
    "next_month",
    "expected_next_month",
    "next_online_share",
    "next_online_amount",
    "next_online_share_delta",
}
ALWAYS_BLOCKED_PREFIXES = ("next_", "target_")


DIRECT_ONLINE_FEATURES = {
    "인터넷뱅킹거래금액",
    "스마트뱅킹거래금액",
    "인터넷뱅킹거래건수",
    "스마트뱅킹거래건수",
    "online_amount",
    "online_count",
    "online_share",
    "internet_share_in_online",
    "smart_share_in_online",
    "internet_share_within_online",
    "smart_share_within_online",
    "online_count_share",
    "온라인이용여부",
    "Y2_온라인거래건수",
    "is_online_active",
    "is_internet_active",
    "is_smart_active",
    "log1p_online_amount",
    "log1p_online_count",
    "online_roll3_mean",
    "online_amount_roll3_mean",
    "online_roll6_mean",
    "online_amount_roll6_mean",
    "online_cnt_roll3",
    "online_use_roll3",
    "online_share_roll3_mean",
    "online_share_roll6_mean",
    "online_count_roll3_mean",
    "online_count_roll6_mean",
    "is_online_active_months_roll3",
    "is_online_active_months_roll6",
    "is_smart_active_months_roll3",
    "is_smart_active_months_roll6",
    "online_share_diff1",
    "online_share_lag1",
    "online_amount_diff1",
    "online_amount_lag1",
    "online_amount_growth1",
    "prev_online_amount_is_zero",
    "prev_month_zero_online",
    "온라인거래건수_lag1",
    "온라인거래건수_lag2",
    "온라인거래건수_lag3",
    "온라인거래_최근3개월평균",
    "온라인거래_최근3개월표준편차",
    "온라인거래_3개월변화",
    "디지털거래금액_lag1",
    "디지털거래비중_lag1",
}


LEAKAGE_DUPLICATE_CANDIDATE_FEATURES = {
    *DIRECT_ONLINE_FEATURES,
    "active_months_3",
    "active_months_6",
    "active_ratio_6m",
    "online_diversity",
    "online_x_grade",
    "online_share_lag1_x_grade",
    "card_x_online",
    "fx_x_online",
    "deposit_x_online_share",
    "offline_amount",
    "total_channel_amount",
    "offline_count",
    "total_channel_count",
    "branch_share",
    "atm_share",
    "phone_share",
    "branch_count_share",
    "is_no_channel",
    "is_branch_active",
    "is_atm_active",
    "log1p_offline_amount",
    "log1p_total_channel",
    "offline_roll3_mean",
    "offline_roll6_mean",
    "offline_amount_lag1",
    "prev_month_zero_offline",
    "오프라인거래금액_lag1",
    "전체채널거래금액_lag1",
    "오프라인거래비중_lag1",
    "loan_total",
    "총대출잔액_lag1",
    "대출한도소진율_lag1",
    "순입출금_lag1",
    "card_total",
    "credit_share",
    "check_share",
    "fx_total",
    "log1p_card",
    "log1p_fx",
    "log1p_deposit",
    "log1p_loan",
    "전담_등급_interact",
    "auto_tendency_index",
    "months_observed",
}


STRICT_ONLINE_FEATURES = {
    *DIRECT_ONLINE_FEATURES,
    "온라인이용여부",
    "Y2_온라인거래건수",
    "internet_share_in_online",
    "smart_share_in_online",
    "online_diversity",
    "online_x_grade",
    "online_share_lag1_x_grade",
    "active_months_3",
    "active_months_6",
    "active_ratio_6m",
    "auto_tendency_index",
    "온라인거래건수_lag1",
    "온라인거래건수_lag2",
    "온라인거래건수_lag3",
    "온라인거래_최근3개월평균",
    "온라인거래_최근3개월표준편차",
    "온라인거래_3개월변화",
    "디지털거래금액_lag1",
    "디지털거래비중_lag1",
}

STRICT_ONLINE_NAME_TOKENS = (
    "online",
    "온라인",
    "인터넷뱅킹",
    "스마트뱅킹",
    "디지털",
)


BASIC_FEATURES = {
    "요구불예금잔액",
    "거치식예금잔액",
    "적립식예금잔액",
    "수익증권잔액",
    "신탁잔액",
    "퇴직연금잔액",
    "여신한도금액",
    "여신_운전자금대출잔액",
    "운전_할인어음잔액",
    "운전_당좌대출잔액",
    "운전_일반자금대출잔액",
    "운전_무역금융잔액",
    "운전_주택자금대출잔액",
    "운전_기업구매자금대출잔액",
    "운전_외상매출채권담보대출잔액",
    "운전_기타운전자금대출잔액",
    "여신_시설자금대출잔액",
    "시설_일반자금대출잔액",
    "시설_에너지절약시설대출잔액",
    "시설_주택자금대출잔액",
    "시설_기타시설자금대출잔액",
    "외환_수출실적금액",
    "외환_수입실적금액",
    "신용카드사용금액",
    "체크카드사용금액",
    "창구거래금액",
    "인터넷뱅킹거래금액",
    "스마트뱅킹거래금액",
    "폰뱅킹거래금액",
    "ATM거래금액",
    "자동이체금액",
    "요구불입금금액",
    "요구불출금금액",
    "요구불예금좌수",
    "거치식예금좌수",
    "적립식예금좌수",
    "수익증권좌수",
    "신탁좌수",
    "퇴직연금좌수",
    "여신_운전자금대출좌수",
    "운전_할인어음좌수",
    "운전_당좌대출좌수",
    "운전_일반자금대출좌수",
    "운전_무역금융좌수",
    "운전_주택자금대출좌수",
    "운전_기업구매자금대출좌수",
    "운전_외상매출채권담보대출좌수",
    "운전_기타운전자금대출좌수",
    "여신_시설자금대출좌수",
    "시설_일반자금대출좌수",
    "시설_에너지절약시설대출좌수",
    "시설_주택자금대출좌수",
    "시설_기타시설자금대출좌수",
    "신용카드개수",
    "외환_수출실적거래건수",
    "외환_수입실적거래건수",
    "창구거래건수",
    "인터넷뱅킹거래건수",
    "스마트뱅킹거래건수",
    "폰뱅킹거래건수",
    "ATM거래건수",
    "자동이체거래건수",
    "법인_고객등급_enc",
    "전담고객여부_enc",
    "업종_대분류_enc",
    "업종_중분류_enc",
    "사업장_시도_enc",
    "사업장_시군구_enc",
}


def load_column_config(processed_dir: Path) -> dict[str, Any]:
    config_path = processed_dir / "modeling_columns.json"
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_split_data(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    notebook_output = processed_dir / "dx_processed.csv"
    if notebook_output.exists():
        df = pd.read_csv(notebook_output)
        df["기준년월"] = pd.to_datetime(df["기준년월"])
        if "next_month" in df.columns and "expected_next_month" in df.columns:
            next_month = pd.to_datetime(df["next_month"], errors="coerce")
            expected_next_month = pd.to_datetime(df["expected_next_month"], errors="coerce")
            df = df[next_month.eq(expected_next_month)].copy()
        df = df[df["기준년월"] <= pd.Timestamp("2025-11-01")].copy()
        train = df[df["기준년월"] <= pd.Timestamp("2024-12-01")].copy()
        valid = df[df["기준년월"].between(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-01"))].copy()
        test = df[df["기준년월"].between(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-11-01"))].copy()
        return train, valid, test

    def read_split(split: str) -> pd.DataFrame:
        model_input = processed_dir / f"dx_panel_{split}_model_input.csv.gz"
        fallback = processed_dir / f"dx_panel_{split}.csv.gz"
        return pd.read_csv(model_input if model_input.exists() else fallback)

    train = read_split("train")
    valid = read_split("valid")
    test = read_split("test")
    return train, valid, test


def feature_exclusion_reason(col: str, column_config: dict[str, Any], config: BaselineConfig) -> str | None:
    always_blocked = set(ALWAYS_BLOCKED_COLUMNS)
    always_blocked.update(
        {
            column_config.get("id_col", "법인ID"),
            column_config.get("date_col", "기준년월"),
            column_config.get("split_col", "split"),
            column_config.get("target_t1", "Y_t1"),
            column_config.get("target_t3", "Y_t3"),
            config.target,
            column_config.get("secondary_target", "target_new_online_activation"),
        }
    )
    if col in always_blocked or col.startswith(ALWAYS_BLOCKED_PREFIXES):
        return "temporal_leakage_or_non_feature"
    if config.exclude_heuristic_score and col == "eda_priority_score":
        return "manual_heuristic_score"
    if config.exclude_strict_online_features and (
        col in STRICT_ONLINE_FEATURES or any(token in col for token in STRICT_ONLINE_NAME_TOKENS)
    ):
        return "strict_online_target_proxy"
    if config.exclude_direct_online_features and col in DIRECT_ONLINE_FEATURES:
        return "direct_online_target_proxy"
    if config.exclude_leakage_duplicate_candidates and col in LEAKAGE_DUPLICATE_CANDIDATE_FEATURES:
        return "leakage_or_duplicate_candidate"
    return None


def make_feature_audit(column_config: dict[str, Any], config: BaselineConfig) -> pd.DataFrame:
    rows = []
    for col in column_config.get("feature_cols", []):
        reason = feature_exclusion_reason(col, column_config, config)
        rows.append(
            {
                "feature": col,
                "status": "excluded" if reason else "used",
                "reason": reason or "",
            }
        )
    return pd.DataFrame(rows)


def build_feature_lists(column_config: dict[str, Any], config: BaselineConfig) -> tuple[list[str], list[str], list[str]]:
    if "feature_cols" in column_config:
        if config.basic_only_features:
            numeric = [col for col in column_config["feature_cols"] if col in BASIC_FEATURES]
            return numeric, [], numeric

        numeric = [
            col
            for col in column_config["feature_cols"]
            if feature_exclusion_reason(col, column_config, config) is None
        ]
        return numeric, [], numeric

    if config.basic_only_features:
        categorical = [col for col in column_config["categorical_features"] if col in BASIC_FEATURES]
        numeric = [col for col in column_config["numeric_features"] if col in BASIC_FEATURES]
        return categorical + numeric, categorical, numeric

    categorical = [
        col
        for col in column_config["categorical_features"]
        if feature_exclusion_reason(col, column_config, config) is None
    ]
    numeric = [
        col
        for col in column_config["numeric_features"]
        if feature_exclusion_reason(col, column_config, config) is None
    ]
    features = categorical + numeric
    return features, categorical, numeric


def validate_no_leakage(features: list[str], column_config: dict[str, Any], target: str) -> None:
    blocked = set(column_config.get("excluded_leakage_columns", []))
    blocked.add(target)
    blocked.add(column_config.get("secondary_target", "target_new_online_activation"))
    leakage = [
        col
        for col in features
        if col in blocked or col.startswith("next_") or col.startswith("target_")
    ]
    if leakage:
        raise ValueError(f"Potential leakage columns found in features: {leakage}")


def make_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def make_models(config: BaselineConfig, categorical: list[str], numeric: list[str]) -> dict[str, Pipeline]:
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
            random_state=config.random_seed,
        ),
        "decision_tree_depth6": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=100,
            class_weight="balanced",
            random_state=config.random_seed,
        ),
        "random_forest_depth8": RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=80,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.random_seed,
        ),
    }
    return {
        name: Pipeline(
            steps=[
                ("preprocess", make_preprocessor(categorical, numeric)),
                ("model", model),
            ]
        )
        for name, model in models.items()
    }


def predict_score(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    decision = model.decision_function(x)
    return 1 / (1 + np.exp(-decision))


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_idx = int(np.nanargmax(f1_values))
    return float(thresholds[best_idx]), float(f1_values[best_idx])


def top_k_metrics(y_true: np.ndarray, y_score: np.ndarray, rates: tuple[float, ...]) -> dict[str, float]:
    order = np.argsort(-y_score)
    positives = max(float(y_true.sum()), 1.0)
    base_rate = float(y_true.mean()) if len(y_true) else 0.0
    result: dict[str, float] = {}
    for rate in rates:
        k = max(1, int(np.ceil(len(y_true) * rate)))
        selected = y_true[order[:k]]
        selected_positive_rate = float(selected.mean()) if k else 0.0
        recall = float(selected.sum() / positives)
        lift = float(selected_positive_rate / base_rate) if base_rate > 0 else 0.0
        label = int(rate * 100)
        result[f"top_{label}pct_recall"] = recall
        result[f"top_{label}pct_lift"] = lift
        result[f"top_{label}pct_precision"] = selected_positive_rate
    return result


def evaluate_split(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    rates: tuple[float, ...],
) -> dict[str, float]:
    y_true = y.to_numpy()
    y_score = predict_score(model, x)
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "rows": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }
    metrics.update(top_k_metrics(y_true, y_score, rates))
    return metrics


def train_and_evaluate(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    numeric: list[str],
    config: BaselineConfig,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, Pipeline]]:
    x_train = train[features]
    y_train = train[config.target].astype(int)
    x_valid = valid[features]
    y_valid = valid[config.target].astype(int)
    x_test = test[features]
    y_test = test[config.target].astype(int)

    models = make_models(config, categorical, numeric)
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        valid_score = predict_score(model, x_valid)
        threshold, _ = best_f1_threshold(y_valid.to_numpy(), valid_score)
        metrics[name] = {
            "train": evaluate_split(model, x_train, y_train, threshold, config.top_k_rates),
            "valid": evaluate_split(model, x_valid, y_valid, threshold, config.top_k_rates),
            "test": evaluate_split(model, x_test, y_test, threshold, config.top_k_rates),
        }
        fitted_models[name] = model

    return metrics, fitted_models


def write_outputs(
    output_dir: Path,
    metrics: dict[str, dict[str, dict[str, float]]],
    models: dict[str, Pipeline],
    features: list[str],
    categorical: list[str],
    numeric: list[str],
    config: BaselineConfig,
    feature_audit: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_reason_counts = (
        feature_audit[feature_audit["status"].eq("excluded")]["reason"].value_counts().sort_index().to_dict()
    )
    payload = {
        "config": asdict(config),
        "feature_count": len(features),
        "categorical_feature_count": len(categorical),
        "numeric_feature_count": len(numeric),
        "excluded_by_design": ["next_*", "target_*", "key_columns", "split", "eda_priority_score"],
        "excluded_direct_online_features": sorted(DIRECT_ONLINE_FEATURES) if config.exclude_direct_online_features else [],
        "excluded_leakage_duplicate_candidate_features": (
            sorted(LEAKAGE_DUPLICATE_CANDIDATE_FEATURES)
            if config.exclude_leakage_duplicate_candidates
            else []
        ),
        "excluded_reason_counts": excluded_reason_counts,
        "metrics": metrics,
    }
    feature_audit.to_csv(output_dir / "feature_audit.csv", index=False, encoding="utf-8-sig")
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = []
    for model_name, split_metrics in metrics.items():
        for split_name, values in split_metrics.items():
            rows.append({"model": model_name, "split": split_name, **values})
    pd.DataFrame(rows).to_csv(output_dir / "baseline_metrics.csv", index=False, encoding="utf-8-sig")

    best_model_name = max(metrics, key=lambda name: metrics[name]["valid"]["pr_auc"])
    joblib.dump(models[best_model_name], output_dir / "best_baseline_model.joblib")

    report = build_report(payload, best_model_name)
    (output_dir / "baseline_report.md").write_text(report, encoding="utf-8")


def build_report(payload: dict[str, Any], best_model_name: str) -> str:
    lines = [
        "# Baseline 모델링 리포트",
        "",
        "## 누수 방지 기준",
        "",
        "- 학습 피처는 `outputs/processed/modeling_columns.json`에서 불러왔습니다.",
        "- `next_`로 시작하는 다음 달 컬럼은 모두 제외했습니다.",
        "- `target_`으로 시작하는 정답/보조정답 컬럼은 모두 제외했습니다.",
        "- `기준년월`, `법인ID`, `split`은 모델 피처에서 제외했습니다.",
        "- `eda_priority_score`는 사람이 정한 휴리스틱 점수이므로 baseline 학습 피처에서 제외했습니다.",
        f"- 온라인 직접 변수 제외 ablation 여부: `{payload['config']['exclude_direct_online_features']}`",
        f"- 누수/중복 1차 후보 동시 제거 여부: `{payload['config']['exclude_leakage_duplicate_candidates']}`",
        f"- 제외 사유별 개수: `{payload['excluded_reason_counts']}`",
        "- 피처 감사표는 `feature_audit.csv`에 저장했습니다.",
        "",
        "## 피처 수",
        "",
        f"- 전체 사용 피처: {payload['feature_count']}",
        f"- 범주형 피처: {payload['categorical_feature_count']}",
        f"- 숫자형 피처: {payload['numeric_feature_count']}",
        "",
        "## 모델 비교",
        "",
        "| model | split | ROC-AUC | PR-AUC | F1 | Top 10% Recall | Top 10% Lift |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name, split_metrics in payload["metrics"].items():
        for split_name in ["train", "valid", "test"]:
            m = split_metrics[split_name]
            lines.append(
                f"| {model_name} | {split_name} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | "
                f"{m['f1']:.4f} | {m['top_10pct_recall']:.4f} | {m['top_10pct_lift']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## 선택 기준",
            "",
            f"- 검증 데이터 PR-AUC 기준 최고 baseline 모델: `{best_model_name}`",
            "- 캠페인 타깃팅 목적이므로 이후 모델 비교에서는 PR-AUC, Top-K Recall, Lift@K를 우선 확인합니다.",
            "- Logistic Regression은 baseline 참고 모델입니다. 데이터 크기와 one-hot 피처 수 때문에 수렴 경고가 발생할 수 있어, 최종 모델 후보로 볼 때는 반복 수나 정규화 설정을 다시 조정해야 합니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train leakage-safe baseline models.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--include-heuristic-score", action="store_true")
    parser.add_argument("--exclude-direct-online-features", action="store_true")
    parser.add_argument("--exclude-strict-online-features", action="store_true")
    parser.add_argument("--exclude-leakage-duplicate-candidates", action="store_true")
    parser.add_argument("--basic-only-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BaselineConfig(
        exclude_heuristic_score=not args.include_heuristic_score,
        exclude_direct_online_features=args.exclude_direct_online_features,
        exclude_strict_online_features=args.exclude_strict_online_features,
        exclude_leakage_duplicate_candidates=args.exclude_leakage_duplicate_candidates,
        basic_only_features=args.basic_only_features,
    )
    column_config = load_column_config(args.processed_dir)
    features, categorical, numeric = build_feature_lists(column_config, config)
    validate_no_leakage(features, column_config, config.target)
    feature_audit = make_feature_audit(column_config, config)
    train, valid, test = load_split_data(args.processed_dir)
    metrics, models = train_and_evaluate(train, valid, test, features, categorical, numeric, config)
    write_outputs(args.output_dir, metrics, models, features, categorical, numeric, config, feature_audit)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
