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

import uuid
import numpy as np
import pandas as pd
import mlflow
import xgboost as xgb
import mlflow.xgboost
from datetime import datetime
from mlflow.tracking import MlflowClient

from ...ml.lead_scoring_model_builder import LeadScoringModel
from ...models.ml.lead_scoring_model import LeadScoringInput, LeadScoringOutput
from ...models.request.request_model import LeadScoringRequest
from ...models.responses.response_model import LeadScoringResponse
from ...config.config import settings, LoggingFormatter, APIMetadata, APIPolicies, InferenceStatus
from ...config.api_lifecycle import model_store as MODEL_STORE
from ...routers.data.connection_handler import SnowflakeHandler
from ..errors import Exceptions

from ...config.logger.factory import LoggingFactory

logger: logging.Logger = LoggingFactory.get_logger()

#Load model store
model_store = MODEL_STORE

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
        Compute Salesforce leads scoring values

        Parameters
        ----------
        request : str
            Request object, to capture headers
        scoring_request : LeadScoringRequest
            LeadScoringRequest object with this definition:
                LEAD_ID            Lead ID.
                MARKET             Market/Country for this lead.
                LEAD_PARAMETERS    Lead parameters, used for this inference.

        Returns
        -------
        STATUS_CODE: int
            HTTP code response
        STATE: str
            HTTP state response
        EXECUTION_ID: str
            Execution unique identifier
        INFERENCE_REPORT: Optional[LeadScoringOutput]
            Lead scoring inference report
        """
        logger.info("Starting Compute leads inference")


        execution_id = str(uuid.uuid1())
        logger.info(f"Execution id:{execution_id}")
        


        try:
            # Inputs

            logger.info(f"Input object:{LeadScoringRequest}")
            
            
            try:
                scoring_request_data = pd.DataFrame([scoring_request.model_dump()])
                # Convert to numpy array 2D
                input_data = scoring_request_data[['INFERENCE_PARAMETERS']].values[0][0]
                input_data = pd.DataFrame([input_data]).to_dict()
                input_data = pd.DataFrame(input_data).values
            except Exception as e:
                logger.error("❌ Unable load input data:", e)
                
            try:
                model = MODEL_STORE.model
                model_version = MODEL_STORE.model_version
            except Exception as e:
                logger.error("❌ Unable load model definition", e)


            try:
                                
                try:
                    # Intentar predecir directamente
                    if hasattr(model, "predict"):
                        try:
                            dmatrix = xgb.DMatrix(input_data)
                            inference = model.predict(dmatrix)
                            
                        except Exception as e:
                            logger.error("❌ Prediction failed:", e)
      
                    else:
                        logger.error("❌ Invalid model:", e)
                        raise ValueError("Invalid model")

                except Exception as e:
                    logger.error("❌ Broken model:", e)
                    return LeadScoringResponse(
                        STATUS_CODE=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        STATE=InferenceStatus.FAILED.value,
                        EXECUTION_ID=execution_id,
                        INFERENCE_REPORT=None
                    )
                #Building inference report
                inference_report = LeadScoringOutput( 
                    lead_score = inference.argmax() + 1,
                    confidence = inference.max(),
                    probabilities = {f"score_{i+1}": float(v) for i, v in enumerate(inference[0])},
                    model_info= {"model_name": "lead-scoring-xgboost","version": model_version,"stage": "Production"},
                    prediction_timestamp= datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                
                logger.info(f"Inference report:\n{inference_report.model_dump_json(indent=2)}")
                #mlflow.set_tag("inference_report", inference_json)

                
                response = LeadScoringResponse(
                    STATUS_CODE = int(status.HTTP_200_OK),
                    STATE = InferenceStatus.SUCCESS.value,
                    EXECUTION_ID = execution_id,
                    INFERENCE_REPORT = inference_report
                )
                
                try:
                    SnowflakeHandler.insert(response)
                    logger.info({
                        "status": "success",
                        "message": "Response stored successfully in Snowflake",
                        "execution_id": response.EXECUTION_ID
                        })
                except HTTPException as he:
                    logger.error(f"{Exceptions.FAILED_INSERTION.value}, execution id {execution_id}" )

                return response

            except Exception as e:
                logger.error(Exceptions.FAILED_INFERENCE.value)
                error_response = LeadScoringResponse(
                    STATUS_CODE=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    STATE=InferenceStatus.FAILED.value,
                    EXECUTION_ID=execution_id,
                    INFERENCE_REPORT=None
                )

                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=error_response.model_dump()
                )


        except Exception as e:
            logger.error(f"{Exceptions.BROKEN_PIPE.value}, execution id {execution_id}" )

            properties = {"custom_dimensions": {"exception": str(e)}}
            logger.exception("Captured an exception.", extra=properties)

            error_response = LeadScoringResponse(
                    STATUS_CODE=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    STATE=InferenceStatus.FAILED.value,
                    EXECUTION_ID=execution_id,
                    INFERENCE_REPORT=None
                    )

            raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=error_response.model_dump()
                    )



