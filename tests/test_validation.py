import pytest

from src.validation import (
    fallback_abstentions,
    prediction_to_binary,
    source_label_to_binary,
    validate_binary,
)


def test_source_label_to_binary_maps_binary_values():
    assert source_label_to_binary(0) == 0
    assert source_label_to_binary("1") == 1
    assert source_label_to_binary("negative") == 0
    assert source_label_to_binary("positive") == 1
    assert source_label_to_binary("neutral") is None
    assert source_label_to_binary(None) is None


def test_source_label_to_binary_rejects_out_of_range_integer():
    with pytest.raises(ValueError, match="unsupported binary source label"):
        source_label_to_binary(2)


def test_prediction_to_binary_treats_neutral_as_abstention():
    assert prediction_to_binary("negative") == 0
    assert prediction_to_binary("positive") == 1
    assert prediction_to_binary("neutral") is None
    assert prediction_to_binary(None) is None


def test_validate_binary_reports_coverage_and_neutral_as_error():
    result = validate_binary(
        y_true=[0, 1, 1, 0],
        y_pred=["negative", "neutral", "positive", "positive"],
    )

    assert result["n"] == 4
    assert result["coverage_count"] == 3
    assert result["coverage"] == pytest.approx(0.75)
    assert result["covered_accuracy"] == pytest.approx(2 / 3)
    assert result["accuracy_neutral_as_error"] == pytest.approx(0.5)
    assert result["recall"] == {"negative": 0.5, "positive": 0.5}
    assert result["confusion"] == {
        "negative": {"negative": 1, "positive": 1, "abstain": 0},
        "positive": {"negative": 0, "positive": 1, "abstain": 1},
    }


def test_fallback_abstentions_only_replaces_neutral():
    result = fallback_abstentions(
        ["neutral", "positive", "negative", "neutral"],
        ["negative", "neutral", "positive", "neutral"],
    )
    assert result == ["negative", "positive", "negative", "neutral"]


def test_validate_binary_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        validate_binary([0], ["negative", "positive"])
