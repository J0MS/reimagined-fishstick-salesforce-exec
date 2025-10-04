"""
API routes definition.

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


##from ...models.eor_model import ExperimentObject
#from ...models.rct_model import ExperimentData
#from ...models.responses import RCTGatewayResponse, WriteToDBResponse
#from ...models.validators.response_validator import RCTResponseValidator, RCTOutputSchema
##from ...utils.rct_utils import RCTTools
#from .worker import run_rct, celery
from ...config.config import settings, LoggingFormatter, APIMetadata, APIPolicies

from ..errors import Exceptions


#import pandas as pd
#import numpy as np

# testops_tools package



from .compute import ComputeLeads

@dataclass
class ComputeRouter:
    """
    Class to compute RCT

    Attributes
    ----------
    spark_session : SparkSession
        SparkSession to perform operations
    logger : logging.Logger
        Logger objects
    router: APIRouter
        FastAPI APIRouter object

    """
    #spark_session: SparkSession
    logger: logging.Logger
    router: APIRouter = APIRouter()

    def __post_init__(self):

        self.router.add_api_route("/v{}.{}/compute".format(APIMetadata.api_major, APIMetadata.api_minor),
                                  ComputeLeads.compute,
                                  methods=["POST"],
                                  #response_model=RCTGatewayResponse,
                                  response_description="Design table: Table with experimental units randomized",
                                  summary="Endpoint to generate randomized experimental units",
                                  tags=["core"]
                                  # dependencies=[Depends(verify_access)],
                                  )



