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
import requests
import pytest
from unittest import mock
from datetime import datetime, date, timedelta
from importlib.machinery import SourceFileLoader

from fastapi import status
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append('../../src/')

from src.config.config import settings

INTEGRATION_TEST_URL = os.getenv('INTEGRATION_TEST_URL')

class TestHealthChecker:
    @pytest.mark.rct_api
    def test_health_endpoint_success(self,payload_header_provider):
        """ Test response for /health path."""
        response = requests.get(
            "{}/health".format(INTEGRATION_TEST_URL),
            headers=payload_header_provider
        )
        assert len(response.json()) == 3
        assert response.status_code == status.HTTP_200_OK
        assert response.json().get('state') == "SUCCESS"
        assert response.json().get('health') == "ok"

