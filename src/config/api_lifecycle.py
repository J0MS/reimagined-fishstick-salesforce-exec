"""
API Lifecycle via environment variables.

Copyright 2025 Salesforce Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import mlflow
import mlflow.xgboost
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

import boto3, tempfile
import xgboost as xgb
from datetime import datetime
from dataclasses import dataclass, field
import logging
from .config import settings
#from .config.logger.factory import LoggingFactory


# Global model storage
class ModelStore:
    model = None
    model_version = None
    last_reload = None
    
model_store = ModelStore()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@dataclass
class APILifecycle:
    """Class to define  API lifecycle operations"""

    model_name: str = settings.MODEL_NAME
    model_stage: str = settings.MODEL_STAGE
    mlflow_tracking_url: str =settings.MLFLOW_TRACKING_URI
    aws_access_key_id: str = settings.AWS_ACCESS_KEY_ID
    aws_secret_access_key: str= settings.AWS_SECRET_ACCESS_KEY
    aws_region: str = settings.AWS_REGION
    aws_bucket: str = settings.AWS_BUCKET
    bucket_prefix: str = settings.BUCKET_PREFIX

    @classmethod
    def load_model_from_registry(cls):
        """Load model from MLflow Model Registry"""
        try:
            session = boto3.Session(
                aws_access_key_id=cls.aws_access_key_id,
                aws_secret_access_key=cls.aws_secret_access_key,
                region_name=cls.aws_region
            )
        
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI) 
        
            s3 = session.client("s3")
        
            response = s3.list_objects_v2(Bucket=cls.aws_bucket, Prefix=cls.bucket_prefix)
            model_key = None

            for obj in response.get("Contents", []):
                if obj["Key"].endswith("model.xgb"): 
                    model_key = obj["Key"]
                    break

            if model_key is None:
                raise ValueError("No se encontró el archivo 'model.xgb' dentro de los artifacts de MLflow.")

            import numpy as np
            import pandas as pd
            # --- Descargar y cargar modelo XGBoost ---
            with tempfile.NamedTemporaryFile() as tmp:
                s3.download_fileobj(cls.aws_bucket, model_key, tmp)
                tmp.seek(0)
                booster = xgb.Booster()
                booster.load_model(tmp.name)
            
                f = {'email_opens': {0: 5},
                'email_clicks': {0: 0},
                'page_views': {0: 7},
                'content_downloads': {0: 3},
                'demo_requested': {0: 0},
                'pricing_page_visits': {0: 0},
                'case_study_views': {0: 2},
                'days_since_last_activity': {0: 4.555039840533683},
                'company_size': {0: 35.20217516441934},
                'annual_revenue': {0: 9512582.10732521},
                'is_decision_maker': {0: 0},
                'job_level_score': {0: 3},
                'session_duration_avg': {0: 6.961331938543087},
                'pages_per_session': {0: 4},
                'return_visitor': {0: 0}}
            
                # Convertir a DataFrame
                df = pd.DataFrame(f)

                # Convertir a numpy array 2D
                data = df.values  # shape (1, n_features)

                #data = np.array([ f ])  # convertir a array 2D
                dmatrix = xgb.DMatrix(data) 
            
                r =booster.predict(dmatrix)
                logger.info("Result")
                logger.info(r)
                logger.info(r.shape)
                logger.info( type(booster.predict(dmatrix)) )
            
            logger.info("Connected to MLflow server")
        
            # Update model store
            model_store.model = booster
            model_store.model_version = 1
            model_store.last_reload = datetime.utcnow()
        
            logger.info(f"✅ Model loaded successfully: {cls.model_name} ")
        
            return True
        
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise
