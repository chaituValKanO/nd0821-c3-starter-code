import pytest
import pandas as pd

@pytest.fixture(scope='session')
def data(input_data_path="../data/cleaned_census_data.csv"):
    data = pd.read_csv(input_data_path)
    return data

@pytest.fixture(scope="session")
def expected_columns():
    return ['age',
            'workclass',
            'fnlgt',
            'education',
            'education_num',
            'marital_status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital_gain',
            'capital_loss',
            'hours_per_week',
            'native_country',
            'salary']
