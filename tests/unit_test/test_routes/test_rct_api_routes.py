"""
Unit test RCT API health

Copyright 2024 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Libraries-------------- 

import logging
import pytest
from fastapi.testclient import TestClient
from fastapi import APIRouter, FastAPI
from src.routers.rct.routes import Router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def client():
    app = FastAPI()
    router = Router(logger=logger)
    app.include_router(router.router)
    return TestClient(app)

@pytest.mark.skip
def test_compute_route(client):
    """
    Test the /compute route.
    """
    logger.info("Starting test_compute_route")
    response = client.post("/v1.0/compute", json={})
    assert response.status_code == 200 or response.status_code == 422  # Depending on the input, it might return 422 for validation error
    logger.info("test_compute_route passed")

@pytest.mark.skip
def test_rct_gateway_route(client):
    """
    Test the /rct-gateway route.
    """
    logger.info("Starting test_rct_gateway_route")
    response = client.post("/v1.0/rct-gateway", json={})
    assert response.status_code == 200 or response.status_code == 422  # Depending on the input, it might return 422 for validation error
    logger.info("test_rct_gateway_route passed")

@pytest.mark.skip
def test_random_control_trial_route(client):
    """
    Test the /random-control-trial route.
    """
    logger.info("Starting test_random_control_trial_route")
    response = client.post("/v1.0/random-control-trial", json={})
    assert response.status_code == 200 or response.status_code == 422  # Depending on the input, it might return 422 for validation error
    logger.info("test_random_control_trial_route passed")

@pytest.mark.skip
def test_rct_route(client):
    """
    Test the /rct route.
    """
    logger.info("Starting test_rct_route")
    response = client.post("/v1.0/rct", json={})
    assert response.status_code == 200 or response.status_code == 422  # Depending on the input, it might return 422 for validation error
    logger.info("test_rct_route passed")

@pytest.mark.skip
def test_invalid_route(client):
    """
    Test an invalid route.
    """
    logger.info("Starting test_invalid_route")
    response = client.post("/v1.0/invalid-route", json={})
    assert response.status_code == 404
    logger.info("test_invalid_route passed")

