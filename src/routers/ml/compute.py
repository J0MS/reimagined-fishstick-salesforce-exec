"""
Compute Salesforce XGBoost score leads results.

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
import requests

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

#import pandas as pd
#import numpy as np

#from ...models.validators.response_validator import RCTResponseValidator, RCTOutputSchema
#from ...models.rct_model import ExperimentData
from ...config.config import settings, LoggingFormatter, APIMetadata, APIPolicies
from ..errors import Exceptions

#from ..data.data_access import DataAccess as DeltaRepository

from ...config.logger.factory import LoggingFactory


logger: logging.Logger = LoggingFactory.get_logger()

@dataclass
class ComputeLeads:
    """
    Class to compute leads scoring from request

    Attributes
    ----------
    logger : logging.Logger
        Logger objects
    """


    async def compute(
        request: Request = None,
    ):

        """
        Compute RCT design table

        Parameters
        ----------
        request : str
            Request object, to capture headers
        design_table_file: UploadFile
            CSV File with experimental units data
        experiment_data: Object with the following data:
            - **EXP_NAME**: Experiment Name
            - **EXP_STATUS**: Status of experiment
            - **BLOCKING_FACTORS**: Blocking factors
            - **EXPERIMENTAL_FACTORS**: Experimental factors
            - **SCOPE_SIZE**: Scope size


        Returns
        -------
        str
            Job id of RCT algorithm execution
        """
        logger.info("Starting RCT")

        try:
            # Inputs

            logger.info(f"Input exp_obj:")

            try:
                experiment_id = "RANDOM"
                response = {"experiment_id": experiment_id}
                return response

            except Exception as e:
                logger.error(Exceptions.BROKEN_PIPE.value)


        except Exception as e:
            logger.error(Exceptions.BROKEN_PIPE.value)

            properties = {"custom_dimensions": {"exception": str(e)}}
            logger.exception("Captured an exception.", extra=properties)

            response = {
                "message": Exceptions.BROKEN_PIPE,
                "error": str(e),
            }

            return response


