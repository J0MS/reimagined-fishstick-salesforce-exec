"""
Retrieve data salesforce  score leads results.

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

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from ...ml.lead_scoring_model_builder import LeadScoringModel
from ...models.request.request_model import LeadScoringRequest
from ...models.responses.response_model import LeadScoringResponse
from ...config.config import settings, LoggingFormatter, APIMetadata, APIPolicies
from ..errors import Exceptions

from .connection_handler import SnowflakeHandler

from ...config.logger.factory import LoggingFactory


logger: logging.Logger = LoggingFactory.get_logger()



@dataclass
class DataRetriever:
    """
    Class to retrieve data from Snowflake database

    Attributes
    ----------
    logger : logging.Logger
        Logger objects
    """


    async def retrieve(
        limit: int = 10
    ) :

        """
        Compute RCT design table

        Parameters
        ----------
        request : str
            Request object, to capture headers

        Returns
        -------
        dict
            Set of lead scores
        """
        logger.info("Starting data loading")

        try:
            with SnowflakeHandler().get_snowflake_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("USE DATABASE SFTEST")
                cursor.execute("USE SCHEMA PUBLIC")
                
                query = f"""
                    SELECT STATE, EXECUTION_ID, LEAD_SCORE, CONFIDENCE, 
                    MODEL_NAME, PREDICTION_TIMESTAMP, CREATED_AT
                    FROM PUBLIC.LEAD_SCORING_PLATFORM
                    ORDER BY CREATED_AT DESC
                    LIMIT {limit}
                """
                cursor.execute(query)
                results = cursor.fetchall()
                cursor.close()
            
                responses = []
                for row in results:
                    responses.append({
                        "state": row[0],
                        "execution_id": row[1],
                        "lead_score": row[2],
                        "confidence": row[3],
                        "model_name": row[4],
                        "prediction_timestamp": row[5],
                        "created_at": row[6]
                    })
            
                return {"responses": responses}
        except Exception as e:
            logger.error("❌ Unable to stablish connection:", e)
            raise HTTPException(status_code=500, detail=str(e))


