"""
Compute Salesforce XGBoost score leads results.

Copyright 2025 Salesforce Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import snowflake.connector
from contextlib import contextmanager
from dataclasses import dataclass, field

from ...config.config import settings

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
