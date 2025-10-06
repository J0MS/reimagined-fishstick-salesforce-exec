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

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from ...ml.lead_scoring_model_builder import LeadScoringModel
from ...models.request.request_model import LeadScoringRequest
from ...models.responses.response_model import LeadScoringResponse
from ...config.config import settings, LoggingFormatter, APIMetadata, APIPolicies
from ..errors import Exceptions

#from ..data.data_access import DataAccess as DeltaRepository

from ...config.logger.factory import LoggingFactory


logger: logging.Logger = LoggingFactory.get_logger()



class ModelStore:
    model = None
    model_version = None
    feature_names = None
    scaler_mean = None
    scaler_scale = None
    last_reload = None

model_store = ModelStore()

def load_model_from_registry():
    """Load model from MLflow Model Registry"""
    try:
        MLFLOW_TRACKING_URI = settings.MLFLOW_TRACKING_URI
        MODEL_NAME = settings.MODEL_NAME
        MODEL_STAGE = settings.MODEL_STAGE
        logger.info(f"Loading model: {MODEL_NAME} (stage: {MODEL_STAGE})")
        
        client = MlflowClient()
        model_uri =MLFLOW_TRACKING_URI
        
        # Load the model
        model = mlflow.xgboost.load_model(model_uri)
        
        # Get model version info
        model_version = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])[0]
        run_id = model_version.run_id
        
        logger.info(f"Model version: {model_version.version}, Run ID: {run_id}")
        
        # Load preprocessing info
        try:
            artifacts_path = client.download_artifacts(run_id, "preprocessing_info.json")
            with open(artifacts_path, 'r') as f:
                preprocessing_info = json.load(f)
            
            model_store.feature_names = preprocessing_info['feature_names']
            model_store.scaler_mean = np.array(preprocessing_info['scaler_mean'])
            model_store.scaler_scale = np.array(preprocessing_info['scaler_scale'])
            
            logger.info(f"Loaded preprocessing info: {len(model_store.feature_names)} features")
        except Exception as e:
            logger.warning(f"Could not load preprocessing info: {e}")
        
        # Update model store
        model_store.model = model
        model_store.model_version = model_version.version
        model_store.last_reload = datetime.utcnow()
        
        logger.info(f"✅ Model loaded successfully: {MODEL_NAME} v{model_version.version}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        raise


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


