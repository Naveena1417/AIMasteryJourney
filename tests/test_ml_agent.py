from agents.ml_agent import (
    load_ml_dataset,
    analyze_ml_request_with_llm,
    train_classification_model
)


def test_load_ml_dataset():
    data = load_ml_dataset("data/ml_sample.csv")

    assert data is not None
    assert len(data) == 20
    assert "Purchased" in data.columns


def test_ml_request_analysis():
    data = load_ml_dataset("data/ml_sample.csv")

    result = analyze_ml_request_with_llm(
        "Predict whether a customer will purchase",
        data.columns.tolist()
    )

    assert result["task"] == "classification"
    assert result["target_column"] == "Purchased"


def test_ml_training():
    data = load_ml_dataset("data/ml_sample.csv")

    model, accuracy = train_classification_model(
        data,
        "Purchased",
        "logistic_regression"
    )

    assert model is not None
    assert accuracy >= 0
    assert accuracy <= 1