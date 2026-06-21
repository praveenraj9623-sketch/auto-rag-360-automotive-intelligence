from src.ml.predict import predict_complaint_severity, predict_maintenance_risk, predict_vehicle_risk


def test_predict_maintenance_risk_returns_bounded_probability() -> None:
    result = predict_maintenance_risk(
        {
            "Air temperature": 300.0,
            "Process temperature": 310.0,
            "Rotational speed": 1450,
            "Torque": 42.0,
            "Tool wear": 120,
        }
    )

    assert result["prediction"] in {0, 1}
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["risk_level"] in {"Low", "Medium", "High"}


def test_predict_complaint_severity_returns_valid_label() -> None:
    result = predict_complaint_severity("The vehicle experienced brake failure and a crash risk while driving.")

    assert result["severity"] in {"Low", "Medium", "High"}
    assert result["prediction"] == result["severity"]
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_vehicle_risk_returns_valid_category() -> None:
    result = predict_vehicle_risk(
        {
            "make": "toyota",
            "body-style": "sedan",
            "engine-type": "ohc",
            "horsepower": 110,
            "curb-weight": 2500,
            "fuel-system": "mpfi",
            "city-mpg": 24,
            "highway-mpg": 30,
        }
    )

    assert result["risk_category"] in {"Low", "Medium", "High"}
    assert result["prediction"] == result["risk_category"]
    assert 0.0 <= result["confidence"] <= 1.0
