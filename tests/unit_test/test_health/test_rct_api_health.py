"""
Unit test RCT API health

Copyright 2023 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.routers.health.health import HealthChecker
from src.main import rct_api
import logging

# Instanciate app
client = TestClient(rct_api)


@pytest.mark.rct_api
def test_health(payload_header_provider):
    """ Test response for /health path."""
    response = client.get(
        "/health",
        headers=payload_header_provider
    )
    assert len(response.json()) == 3
    assert response.status_code == 200
    assert response.json().get('state') == "SUCCESS"
    assert response.json().get('health') == "ok"

    
