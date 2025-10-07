"""
API configuration via environment variables.

Copyright 2025 Salesforce Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import logging
from pydantic_settings import BaseSettings
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class APIMetadata:
    """Class to define  API metadata"""

    api_title: str = "Salesforce API"
    api_major: str = "1"
    api_minor: str = "0"
    api_patch: str = "1"
    api_version: str = "{}.{}.{}".format(api_major, api_minor, api_patch)
    api_description: str = """
        Salesforce API helps you to run ML algorithms 🚀

        ## inference

        - You can design robust,unbiased experiments from your set of experimental units, just call  **/inference** endpoint.

        ## Write to DB

        - You can save your experiments results to TestOps database, just call **/write_to_db** endpoint.

    """
    terms_of_service: str = "https://salesforceservice.aws.net/terms/"
    contact: Dict[str, str] = field(default_factory=lambda: ({
            "name": "Salesforcee MLOps Team",
            "url": "https://saalesforcewebsites.net",
            "email": "mlopsteam@salesforce.com",
          }
         )
        )

    license_info: Dict[str, str] = field(default_factory=lambda: ({
            "name": "Enterprise Licence Agreement ",
            "url": "https://salesforceservice.aws.net",
          }
         )
        )
    tags_metadata: List = field(default_factory=lambda: [
        {
            "name": "core",
            "description": "Core operations with input data. **ML** algorithm implemented here",
            "externalDocs": {
                "description": "API Architecture and ML Algorithm documentation",
                "url": "https://www.salesforce-docs.com/ml"
            }
        },
        {
            "name": "data_ops",
            "description": "Data saving  operations with output data"
        },
       ]
      )
    openapi_url: str = "/api/v{}/openapi.json".format(api_version.replace(".","_"))
    api_cloud_role_name: str = "SALESFORCE-API"


@dataclass
class APIPolicies:
    """Class to define API Policies"""

    api_allowed_host: List = field(default_factory=lambda: ["*"])
    trace_version: str = "00"
    trace_id: str = "4bf92f3577b34da6a3ce929d0e0e4736"
    parent_span_id: str = "00f067aa0ba902b7"
    trace_flags: str = "01"
    api_middlewares_exclusions: List = field(default_factory=lambda: ["/docs"])


class Settings(BaseSettings):
    """API configuration class using environment variables.

    Attributes:
        INSTRUMENTATION_KEY instrumentation key
    """
    INSTRUMENTATION_KEY: str
    JWT_SECRET: str
    ALGORITHM: str
    MLFLOW_TRACKING_URI : str
    MODEL_NAME : str
    MODEL_STAGE : str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    AWS_BUCKET: str
    BUCKET_PREFIX: str
    SNOWFLAKE_ACCOUNT: str
    SNOWFLAKE_USER: str
    SNOWFLAKE_PASSWORD: str
    SNOWFLAKE_WAREHOUSE: str
    SNOWFLAKE_DATABASE: str
    SNOWFLAKE_SCHEMA: str


    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

settings = Settings()


class LoggingFormatter(logging.Formatter):
    """
       Custom Logging Formatter
    """

    grey = "\x1b[38;21m"
    green = "\x1b[1;32m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format='%(levelname)s: %(asctime)s - %(message)s (%(filename)s:%(lineno)d)'

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

    def get_format(self, record):
        return self.FORMATS.get(record.levelno)


class RegisteredTeams(Enum):

    """
    Registered Team Codes
    """
    MARKETING = "Marketing"
    SALES_FORCE = "SalesForce"
    OTHER = "Other"

class InferenceStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


