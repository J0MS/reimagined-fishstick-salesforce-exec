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
from ...models.request.request_model import LeadScoringRequest
from ...models.responses.response_model import LeadScoringResponse
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
        scoring_request: LeadScoringRequest,
        request: Request = None,
    ) -> LeadScoringResponse:

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

            logger.info(f"Input exp_obj:{LeadScoringRequest}")

            try:
                experiment_id = "RANDOM"

                response = {
                            "STATUS_CODE": 200,
                            "STATE": experiment_id,
                            "EXECUTION_ID": 100,
                            "INFERENCE_REPORT": {"lead_score": 5,
                                                 "confidence": 0.87,
                                                 "probabilities": {"score_1": 0.02,"score_2": 0.03,"score_3": 0.05,"score_4": 0.03,"score_5": 0.87},
                                                 "model_info": {"model_name": "lead-scoring-xgboost","version": "3","stage": "Production"},
                                                 "prediction_timestamp": "2025-10-04T12:00:00Z"
                                                 }
                            }

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


