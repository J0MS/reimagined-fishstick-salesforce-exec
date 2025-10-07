"""
Compute Salesforce Snowflake connection handler.

Copyright 2025 Salesforce Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import logging
import snowflake.connector
from contextlib import contextmanager
from dataclasses import dataclass, field
from fastapi import HTTPException

from ...config.config import settings
from ...config.logger.factory import LoggingFactory

from ...models.responses.response_model import LeadScoringResponse

logger: logging.Logger = LoggingFactory.get_logger()

@dataclass
class SnowflakeHandler:
    """Class to define  Snowflake handler"""

    account: str = settings.SNOWFLAKE_ACCOUNT
    user: str = settings.SNOWFLAKE_USER
    password: str =settings.SNOWFLAKE_PASSWORD
    warehouse: str = settings.SNOWFLAKE_WAREHOUSE
    database: str= settings.SNOWFLAKE_DATABASE
    schema: str = settings.SNOWFLAKE_SCHEMA


    @classmethod
    @contextmanager
    def get_snowflake_connection(cls):
        """Context manager for Snowflake connection"""
        conn = None
        try:
            conn = snowflake.connector.connect(
                account=cls.account,
                user=cls.user,
                password=cls.password,
                warehouse=cls.warehouse,
                database=cls.database,
                schema=cls.schema,
            )

            yield conn
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Snowflake connection error: {str(e)}")
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def insert(response_data: LeadScoringResponse):
        """Insert API response into Snowflake table"""
    
        insert_query = """
        INSERT INTO PUBLIC.LEAD_SCORING_PLATFORM (
            STATE, 
            EXECUTION_ID, 
            CONFIDENCE, 
            LEAD_SCORE,
            MODEL_NAME, 
            MODEL_STAGE, 
            MODEL_VERSION,
            PREDICTION_TIMESTAMP, 
            SCORE_1, 
            SCORE_2, 
            SCORE_3, 
            SCORE_4, 
            SCORE_5
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """
        
        # Extract values from nested structure
        inference = response_data.INFERENCE_REPORT
        model_info = inference.model_info
        probs = inference.probabilities

        
        values = (
            response_data.STATE,
            response_data.EXECUTION_ID,
            inference.confidence,
            inference.lead_score,
            model_info.get("model_name"),
            model_info.get("stage"),
            model_info.get("version"),
            inference.prediction_timestamp,
            probs.get("score_1"),
            probs.get("score_2"),
            probs.get("score_3"),
            probs.get("score_4"),
            probs.get("score_5")
        )
        
        try:
            logger.info("Attemping data insertion")
            with SnowflakeHandler.get_snowflake_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("USE DATABASE SFTEST")
                cursor.execute("USE SCHEMA PUBLIC")
                cursor.execute(insert_query, values)
                conn.commit()
                cursor.close()
                logger.info("Succesful data insertion")
                return True
            
        except Exception as e:
            logger.error("Insertion error")
            logger.error(str(e))
            raise HTTPException(status_code=500, detail=f"Error inserting data: {str(e)}")
