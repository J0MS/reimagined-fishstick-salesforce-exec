"""
Random Control Trial API.

API routes definition.

Copyright 2024 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging

from .formatter import LoggingFormatter
from ..config import APIMetadata

class LoggingFactory():

  @staticmethod
  def get_logger():
    # Instantiate logger for API
    logger = logging.getLogger(APIMetadata.api_cloud_role_name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        # Instantiate console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        # Instantiate custom LogginFormater
        console_handler.setFormatter(LoggingFormatter())
        # Instantiate Azure log handler to export logging activity (Use in exporter object as well)
        # Set configured handler
        logger.addHandler(console_handler)
    return logger
