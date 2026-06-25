# tests/conftest.py
import pytest, os, requests

SAMPLE_RECORD = {
    'Height': 1.75, 'Weight': 70.0, 'Universe': 'Earth-616',
    'Identity': 'Public', 'Gender': 'Male', 'Marital_Status': 'Single',
    'Teams': 'Avengers', 'Origin': 'Human', 'Magic': 0, 'Mutant': 0,
}

ENDPOINT_URL = os.environ.get('ENDPOINT_URL', '')
DBR_TOKEN    = os.environ.get('DBR_TOKEN', '')

def call_endpoint(record):
    response = requests.post(
        ENDPOINT_URL,
        headers={'Authorization': f'Bearer {DBR_TOKEN}'},
        json={'dataframe_records': record}, timeout=30
    )
    return response.status_code, response.text

@pytest.fixture
def sample_record(): return SAMPLE_RECORD.copy()
