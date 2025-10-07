"""
Random Control Trial API-Health Checker.

API routes definition.

Copyright 2023 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from fastapi import FastAPI, APIRouter, status
from dataclasses import dataclass
import logging

@dataclass
class HealthChecker:
    """Class for determine API Health."""
    logger: logging.Logger
    router: APIRouter = APIRouter(tags=["liveness"])
    state: str = "SUCCESS!"
    response_string: str = "ok"

    def __post_init__(self):
        self.router.add_api_route("/health", self.health, methods=["GET"])


    def health(self):
        self.logger.info("Health check running!")
        return {"statusCode": status.HTTP_200_OK, "state": self.state, "health":self.response_string}

