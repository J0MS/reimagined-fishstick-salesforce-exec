"""
Random Control Trial API-Tes cases configuration.

Copyright 2023 Anheuser Busch InBev Inc.

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

def create_design_table_df(rows: int) -> pd.DataFrame :
    """ Method to create a synthetic rct design table  """
    tags = ["test", "control"]
    random_tags = [random.choice(tags) for _ in range(1,rows + 1)]
    data = {
        'id': [x for x in  range(1,rows + 1)],
        'experiment_id': [x for x in  range(1,rows + 1)],
        'ex_unit_id': [x+100 for x in  range(1,rows + 1)] ,
        'ex_factor_1': ["test", "test", "control", "control", "test"],
        'ex_factor_2': ["No Level" for _ in range(rows)],
        'ex_factor_3': ["No Level" for _ in range(rows)],
        'ex_factor_4': ["No Level" for _ in range(rows)],
        'ex_factor_5': ["No Level" for _ in range(rows)],
        'block_factor_1': ["No Level" for _ in range(rows)],
        'block_factor_2': ["No Level" for _ in range(rows)],
        'block_factor_3': ["No Level" for _ in range(rows)],
        'block_factor_4': ["No Level" for _ in range(rows)],
        'block_factor_5': ["No Level" for _ in range(rows)]
    }
    df = pd.DataFrame(data)
    return df


#def pytest_namespace():
#    return {'shared': None}

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



"""RCT API Module fixtures """

@pytest.fixture
def eor_endpoint_path_provider():
    """ Provide endpoint for EOR API """
    return '/experiment/experiment-object/'

@pytest.fixture
def payload_header_provider():
    """ Provide HTTP header for telemetry """
    return {'abi_user': 'test@ab-inbev.com', 'abi_team':'Other' }

@pytest.fixture
def power_calculator_provider():
    """ Provide PC sample as dict """
    return power_calulator_sample

@pytest.fixture
def rct_design_table_file_provider(autouse=True):
    """ Provide design_table file from CSV as bytes stream """
    df = create_design_table_df(5)
    df.to_csv("{}/design_table_file_example.csv".format(TEST_FILES_LOCATION),index=False)
    table_file = "{}/design_table_file_example.csv".format(TEST_FILES_LOCATION)
    return {'design_table_file': open(table_file, 'rb')}

#@pytest.fixture
#def capture_task_id():
#    # Capture taks_id
#    response = {"task_id": 123}
#    return response


@pytest.fixture
def rct_design_table_file_path_provider():
    """ Provide path design_table file from CSV as bytes stream """
    path_table_file = "{}/design_table_file_example.csv".format(TEST_FILES_LOCATION)
    return path_table_file


@pytest.fixture
def rct_experiment_data_provider():
    """ Provide experiment_data as JSON string"""
    return {'experiment_data': json.dumps(experiment_data_sample)}


class ComputeRCTProviders:
    @pytest.fixture
    def rct_experiment_data_provider():
        """ Provide experiment_data as JSON string"""
        return {'experiment_data': json.dumps(experiment_data_sample)}
    
    @pytest.fixture
    def rct_experiment_data_provider_as_dict():
        """ Provide experiment_data as Dict string"""
        return {'experiment_data': experiment_data_sample}
    
    @pytest.fixture
    def rct_experiment_data_sample_provider():
        """ Provide experiment_data as dictionary"""
        return experiment_data_sample

@pytest.fixture
def rct_table_file_provider():
    """ Provide rct_table file from CSV as bytes stream """

    df = create_design_table_df(5)
    df.to_csv("{}/design_table_file_example.csv".format(TEST_FILES_LOCATION),index=False)
    table_file = "{}/rct_table_file_example.csv".format(TEST_FILES_LOCATION)
    return {'rct_table_file': open(table_file, 'rb')}

@pytest.fixture
def rct_table_file_bad_provider():
    """ Provide rct_table bad file from CSV as bytes stream """

    df = create_design_table_df(5)
    df.to_csv("{}/design_table_file_example.csv".format(TEST_FILES_LOCATION), index=False, header=False)
    table_file = "{}/rct_table_file_bad_example.csv".format(TEST_FILES_LOCATION)
    return {'rct_table_file': open(table_file, 'rb')}


"""RCT API Configuration fixtures """
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

