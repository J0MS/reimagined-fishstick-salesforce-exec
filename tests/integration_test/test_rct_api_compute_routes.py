"""
Random Control Trial API-Integration Test cases

Unitary test.

Copyright 2023 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import sys
import uuid
import requests
import pytest
from fastapi import status
from unittest import mock
from datetime import datetime, date, timedelta
from importlib.machinery import SourceFileLoader

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

sys.path.append('../../src/')



from src.config.config import settings

INTEGRATION_TEST_URL = os.getenv('INTEGRATION_TEST_URL')
API_VERSION = "1.0"
SHARED_DATA = {}

class TestComputeRCTonSuccess:
    @pytest.mark.rct_api
    def test_compute_rct_on_success(self,payload_header_provider,
        rct_experiment_data_provider,
        rct_design_table_file_provider,
        shared_data
        ):
        """ Test response for /compute path."""
        response = requests.post(
            "{}/v{}/compute".format(INTEGRATION_TEST_URL, API_VERSION),
            headers=payload_header_provider,
            data=rct_experiment_data_provider,
            files=rct_design_table_file_provider
        )
        cwd = os.getcwd()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print("testw")
        print(cwd)
        print(dir_path)
        print(dir(settings))

        assert len(response.json()) == 4
        assert response.status_code == status.HTTP_200_OK
        assert response.json().get('state') == "SUCCESS"
        exp_id = response.json().get('exp_id')
        try:
            # Attempt to convert exp_id to an integer
            int(exp_id)
            assert True  # Passes if value is a valid experiment_id
        except ValueError:
            assert False, f"'{exp_id}' cannot be converted to an UUID."

        task_id = response.json().get('task_id')
        try:
            # Attempt to convert exp_id to an UUID
            uuid.UUID(task_id)
            assert True  # Passes if valid is a valid UUID
        except ValueError:
            assert False, f"'{exp_id}' cannot be converted to an UUID."

        response = requests.get(
            "{}/v{}/retrieve_job_results?job_id={}".format(INTEGRATION_TEST_URL, API_VERSION, task_id),
            headers=payload_header_provider,
            data={}
        )
        assert len(response.json()) == 3
        task_result = response.json().get('task_result')
        assert len(task_result) == 2
        assert list(task_result.keys()) == ["experiment_id", "rct_table"]



    @pytest.mark.skip
    def test_retrieve_job_results_on_success(self,
        payload_header_provider,
        shared_data
        ):
        """ Test response for /compute path."""
        global SHARED_DATA
        TASK_ID = SHARED_DATA.get('task_id', None)
        response = requests.get(
            "{}/v{}/retrieve_job_results?job_id={}".format(INTEGRATION_TEST_URL, API_VERSION, TASK_ID),
            headers=payload_header_provider,
            data={}
        )
        assert len(response.json()) == 3
        task_result = response.json().get('task_result')
        assert len(task_result) == 2
        assert list(task_result.keys()) == ["experiment_id", "rct_table"]

