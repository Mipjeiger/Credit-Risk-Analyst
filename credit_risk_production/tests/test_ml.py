import pandas as pd

from ml.features import add_derived_features, row_to_narrative


def test_derived_features():
    row = pd.Series(
        {"Total_TL": 10, "Unsecured_TL": 7, "num_deliq_6mts": 2, "num_deliq_12mts": 3}
    )
    r = add_derived_features(row)
    assert "unsecured_ratio" in r
    assert r["unsecured_ratio"] > 0


def test_narrative():
    row = pd.Series({"Total_TL": 5, "Unsecured_TL": 2})
    txt = row_to_narrative(row, cid=1)
    assert "CUSTOMER_PROFILE id=1" in txt
