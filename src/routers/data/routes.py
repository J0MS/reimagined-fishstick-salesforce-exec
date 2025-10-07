"""
Data routes definition.

Copyright 2025 Salesforce Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from dataclasses import dataclass
import logging
from io import StringIO
#import requests

from fastapi import (
    APIRouter,
    status,
    HTTPException,
    Form,
    File,
    UploadFile,
    Request,
    Depends,
)




from ...models.responses.response_model import LeadScoringResponse
from ...config.config import settings, LoggingFormatter, APIMetadata, APIPolicies

from ..errors import Exceptions
from .retriever import DataRetriever
from mlflow.tracking import MlflowClient





@dataclass
class DataRouter:
    """
    Class to interact with Snowflake instance

    Attributes
    ----------
    logger : logging.Logger
        Logger objects
    router: APIRouter
        FastAPI APIRouter object

    """

    logger: logging.Logger
    router: APIRouter = APIRouter()

    def __post_init__(self):

        self.router.add_api_route("/v{}.{}/retrieve".format(APIMetadata.api_major, APIMetadata.api_minor),
                                  DataRetriever.retrieve,
                                  methods=["GET"],
                                  #response_model=LeadScoringResponse,
                                  response_description="Lead scoring report for each lead",
                                  summary="Endpoint to get lead scoring report",
                                  tags=["core"]
                                  # dependencies=[Depends(verify_access)],
                                  )



