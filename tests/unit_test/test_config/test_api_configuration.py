"""
Salesforce Testcases for middleware

Copyright 2025 Salesforce Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import pytest
import random
import logging
from dataclasses import asdict
from dataclasses import fields
from typing import List

sys.path.append('src/')
from src.config.config import Settings, LoggingFormatter, APIMetadata, APIPolicies

""" Auxiliar functions """
def record_generator(level: str) -> logging.LogRecord:
    name = str(random.randint(100,999))
    int_level= random.randint(1,5)
    path_name = str(random.randint(100,999))
    line_number= random.randint(100,999)
    str_level= level
    return logging.LogRecord(name, int_level, path_name, line_number, str_level, (), False)


""" Testing APIMetadata class """
class TestAPIMetadata:
    @pytest.mark.rct_api
    def test_api_metadata(self, api_metadata_keys_provider):
        """ Test configuration for APIMetadata class."""
        
        api_metadata = APIMetadata()
        assert [*asdict(api_metadata).keys()] == api_metadata_keys_provider
        assert type(api_metadata.__repr__() ) == str

""" Testing Settings class """
class TestSettings:
    def test_settings_attributes(self):
        settings = Settings()

        assert hasattr(settings, 'TEST_OPS_API_URL')
        assert hasattr(settings, 'INSTRUMENTATION_KEY')
        assert isinstance(settings.TEST_OPS_API_URL, str)
        assert isinstance(settings.INSTRUMENTATION_KEY, str)

    def test_settings_config_attributes(self):
        assert hasattr(Settings.Config, 'env_file')
        assert hasattr(Settings.Config, 'env_file_encoding')
        assert isinstance(Settings.Config.env_file, str)
        assert isinstance(Settings.Config.env_file_encoding, str)
        assert Settings.Config.env_file == '.env'
        assert Settings.Config.env_file_encoding == 'utf-8'

""" Testing APIPolicies class """
class TestAPIPolicies:
    @pytest.mark.rct_api
    def test_api_allowed_host_default_value(self):
        policies = APIPolicies()
        assert policies.api_allowed_host == ["*"]

    @pytest.mark.rct_api
    def test_api_middlewares_exclusions_default_value(self):
        policies = APIPolicies()
        expected_exclusions = [
            "/random-control-trial",
            "/docs",
            "/redoc",
            APIMetadata.openapi_url
        ]
        assert policies.api_middlewares_exclusions == expected_exclusions

    @pytest.mark.rct_api
    def test_fields_annotations(self):
        annotations = {
            "api_allowed_host": List,
            "trace_version": str,
            "trace_id": str,
            "parent_span_id": str,
            "trace_flags": str,
            "api_middlewares_exclusions": List
        }
        for field in fields(APIPolicies):
            assert field.name in annotations
            assert field.type == annotations[field.name]

""" Testing LoggingFormatter class """
class TestLoggingFormatter:
    @pytest.mark.rct_api
    def test_format_debug(self):
        formatter = LoggingFormatter()
        record = logging.LogRecord("test_logger", logging.DEBUG, "path/to/file.py", 42, "Debug message", None, None)
        formatted = formatter.format(record)
        assert formatted == f"\x1b[38;21mDEBUG: {record.asctime} - {record.msg} ({record.filename}:{record.lineno})\x1b[0m"
  
    @pytest.mark.rct_api
    def test_format_info(self):
        formatter = LoggingFormatter()
        record = logging.LogRecord("test_logger", logging.INFO, "path/to/file.py", 42, "Info message", None, None)
        formatted = formatter.format(record)
        assert formatted == f"\x1b[1;32mINFO: {record.asctime} - {record.msg} ({record.filename}:{record.lineno})\x1b[0m"

    @pytest.mark.rct_api
    def test_format_warning(self):
        formatter = LoggingFormatter()
        record = logging.LogRecord("test_logger", logging.WARNING, "path/to/file.py", 42, "Warning message", None, None)
        formatted = formatter.format(record)
        assert formatted == f"\x1b[33;21mWARNING: {record.asctime} - {record.msg} ({record.filename}:{record.lineno})\x1b[0m"

    @pytest.mark.rct_api
    def test_format_error(self):
        formatter = LoggingFormatter()
        record = logging.LogRecord("test_logger", logging.ERROR, "path/to/file.py", 42, "Error message", None, None)
        formatted = formatter.format(record)
        assert formatted == f"\x1b[31;21mERROR: {record.asctime} - {record.msg} ({record.filename}:{record.lineno})\x1b[0m"

    @pytest.mark.rct_api
    def test_format_critical(self):
        formatter = LoggingFormatter()
        record = logging.LogRecord("test_logger", logging.CRITICAL, "path/to/file.py", 42, "Critical message", None, None)
        formatted = formatter.format(record)
        assert formatted == f"\x1b[31;1mCRITICAL: {record.asctime} - {record.msg} ({record.filename}:{record.lineno})\x1b[0m"

    @pytest.mark.rct_api
    @pytest.mark.parametrize("initial_level,final_level", [("DEBUG", 1), ("INFO", 2), ("WARNING", 3),("ERROR", 4),("CRITICAL", 5) ])
    def test_api_logging_formater(self, initial_level, final_level):
        """ Test configuration for API Logging formater class."""
    
        FORMAT='%(levelname)s: %(asctime)s - %(message)s (%(filename)s:%(lineno)d)'
        logging.basicConfig(format=FORMAT)
        
        formater = LoggingFormatter()
        assert formater.format(record_generator(initial_level)) == initial_level
        record = record_generator(initial_level)
        assert formater.get_format(record) == formater.FORMATS.get(record.levelno)
  




#    assert formater.get_format(record_generator(initial_level)) == list(formats.values())[1]
    
    #record = logging.LogRecord('bob', 1, 'foo', 23, "WARNING", (), False)
#    assert formater.format(record_generator("WARNING") ) == "WARNING"
    #list(formats.values())[1]

    #assert formater.format(record_generator("DEBUG")) == "DEBUG"
    #assert formater.format(record_generator("INFO")) == "INFO"
    #assert formater.format(record_generator("WARNING")) == "WARNING"
    #assert formater.format(record_generator("ERROR")) == "ERROR"
    #assert formater.format(record_generator("CRITICAL")) == "CRITICAL"

