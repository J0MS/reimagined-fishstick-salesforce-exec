"""
Lead scoring system cases configuration.

Copyright 2025 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Fixtures and function helpers for Test Cases.
"""

import os
import json
import random
import sys
import pytest
import pandas as pd





sys.path.append('src/')
#TEST_FILES_LOCATION = "tests/unit_test/data"
# Funciona!!:
#TEST_FILES_LOCATION = "/src/src/tests/unit_test/data"
TEST_FILES_LOCATION = "{}/src/tests/unit_test/data".format(os.getcwd())



#from src.main import rct_api
from src.models.sample import power_calulator_sample, rct_sample, experiment_data_sample


""" Helper functions """


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    # Initialize a custom namespace to store data
    config.shared_data = {}

@pytest.fixture(scope='session')
def shared_data(request):
    """Create a shared data dictionary accessible across tests."""
    if not hasattr(request.session, 'shared_data'):
        request.session.shared_data = {"task_id": "w" }
    return request.session.shared_data



"""Test Module fixtures """

@pytest.fixture
def eor_endpoint_path_provider():
    """ Provide endpoint for EOR API """
    return '/experiment/experiment-object/'

@pytest.fixture
def payload_header_provider():
    """ Provide HTTP header for telemetry """
    return {'salesforce_user': 'test@salesforce.com', 'salesforce_team':'marketing' }


"""API Configuration fixtures """
@pytest.fixture
def api_metadata_keys_provider():
    """ Provide API Metadata valid keys """
    return ['api_title', 'api_major', 'api_minor',
            'api_patch', 'api_version', 'api_description',
            'terms_of_service', 'contact', 'license_info',
            'tags_metadata', 'openapi_url', 'api_cloud_role_name'
            ]

"""Test suite results reporter"""
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Access test outcomes
    #total_tests = terminalreporter._numcollected
    passed_tests = len(terminalreporter.stats.get("passed", []))
    failed_tests = len(terminalreporter.stats.get("failed", []))
    skipped_tests = len(terminalreporter.stats.get("skipped", []))
    total_tests = passed_tests  + failed_tests + skipped_tests 
                        
    # Display the statistics
    terminalreporter.write_sep("-", f"Total tests: {total_tests}")
    terminalreporter.write_sep("-", f"Passed tests: {passed_tests}")
    terminalreporter.write_sep("-", f"Failed tests: {failed_tests}")
    terminalreporter.write_sep("-", f"Skipped tests: {skipped_tests}")
    passed_tests = str(passed_tests)

    # Add the test results to the file
    with open("/src/test_suite_results.txt", "w") as file:
        file.write(f"TOTAL_TEST={total_tests}\n")
        file.write(f"PASSED_TEST={passed_tests}\n")

