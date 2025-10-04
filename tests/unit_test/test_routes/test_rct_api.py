"""
Random Control Trial API-API Test cases

Unitary test.

Copyright 2023 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import json
import secrets
import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest import mock

from src.main import rct_api
from src.models.eor_model import ExperimentObject
from src.models.rct_model import ExperimentData
from src.models.responses import RCTGatewayResponse, WriteToDBResponse
from src.models.validators.response_validator import RCTResponseValidator, RCTOutputSchema
from src.config.config import settings, LoggingFormatter, APIMetadata, APIPolicies

client = TestClient(rct_api)

TIME_FORMAT = '%Y-%m-%d'

def rct_wrong_date_syntax(date_mutation: str) -> dict:
    return {'detail': [{'loc': ['body', 'experiment_data', '__root__'],
                        'msg': "time data '{}' does not match format '{}'".format(date_mutation, TIME_FORMAT),
                        'type': 'value_error'}]}

def rct_wrong_date_delta() -> dict:
    return {'detail': [{'loc': ['body', 'experiment_data', '__root__'],
                        'msg': 'Invalid date delta detected, fields with suffix START_DATE must be less than or equal to fields with suffix END_DATE ',
                        'type': 'value_error'}]}

def rct_wrong_date_ranges() -> dict:
    return {'detail': [{'loc': ['body', 'experiment_data', '__root__'],
                        'msg': 'Invalid date ranges detected, dates must be in range 2020-12-31 00:00:00 to {}'.format(datetime.today().strftime(TIME_FORMAT)),
                        'type': 'value_error'}]}

def rct_payload_missing_param(missing_parameter: str) -> dict:
    error_response = {'detail': [{'loc': ['body', 'experiment_data'],
                                  'msg': 'field required',
                                  'type': 'value_error.missing'}]}
    body_key = error_response.get('detail')[0].get('loc')
    body_key.insert(2, missing_parameter)
    new_detail = [{'loc': body_key,
                   'msg': error_response.get('detail')[0].get('msg'),
                   'type': error_response.get('detail')[0].get('type')}]
    error_response.update({'detail': new_detail})
    return error_response

def rct_wrong_parameter(target_param: str, bad_param: str, correct_param: str) -> dict:
    loc_list = ['body', 'experiment_data']
    loc_list.append(target_param)
    return {'detail': [{'loc': loc_list,
                        'msg': "Invalid {} value:{}, should be: {}".format(target_param, bad_param, correct_param),
                        'type': 'value_error'}]}
@pytest.mark.skip
@pytest.mark.rct_api
def test_rct_gateway_success(payload_header_provider,
                             rct_experiment_data_provider_as_dict,
                             rct_design_table_file_provider,
                             mocker):
    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider_as_dict)
    response = client.post(
        "/v{}.{}/compute".format(APIMetadata.api_major, APIMetadata.api_minor),
        data=json.dumps(rct_experiment_data_provider_as_dict),
        files=rct_design_table_file_provider,
        headers=payload_header_provider
    )
    assert response.status_code == 201
    assert response.json().get('statusCode') == 201
    assert response.json().get('state') == "SUCCESS"
    assert isinstance(response.json().get('exp_id'), int)
    assert isinstance(response.json().get('rct_table'), str)
    assert int(response.json().get('exp_id')) > 0
@pytest.mark.skip
@pytest.mark.rct_api
def test_rct_gateway_missing_exp_name(payload_header_provider,
                                      rct_design_table_file_provider,
                                      rct_experiment_data_sample_provider,
                                      rct_experiment_data_provider,
                                      mocker):
    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
    del local_rct_experiment_data_sample_provider['EXP_NAME']
    experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
    response = client.post(
        "/rct-gateway",
        data=experiment_data,
        files=rct_design_table_file_provider,
        headers=payload_header_provider
    )
    assert response.status_code == 422
    assert response.json() == rct_payload_missing_param('EXP_NAME')

@pytest.mark.skip
@pytest.mark.rct_api
def test_rct_gateway_missing_exp_status(payload_header_provider,
                                        rct_design_table_file_provider,
                                        rct_experiment_data_sample_provider,
                                        rct_experiment_data_provider,
                                        mocker):
    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
    del local_rct_experiment_data_sample_provider['EXP_STATUS']
    experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
    response = client.post(
        "/rct-gateway",
        data=experiment_data,
        files=rct_design_table_file_provider,
        headers=payload_header_provider
    )
    assert response.status_code == 422
    assert response.json() == rct_payload_missing_param('EXP_STATUS')

@pytest.mark.skip
@pytest.mark.rct_api
def test_rct_gateway_bad_exp_status(payload_header_provider,
                                    rct_design_table_file_provider,
                                    rct_experiment_data_sample_provider,
                                    rct_experiment_data_provider,
                                    mocker):
    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
    bad_exp_status = str(secrets.randbelow(999))
    target_parameter = 'EXP_STATUS'
    local_rct_experiment_data_sample_provider.update({target_parameter: bad_exp_status})
    experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
    response = client.post(
        "/rct-gateway",
        data=experiment_data,
        files=rct_design_table_file_provider,
        headers=payload_header_provider
    )
    rct_wrong_parameter_response = rct_wrong_parameter(target_parameter, bad_exp_status,
                                                       "['IN_PROGRESS', 'NOT_STARTED', 'COMPLETED', 'SUBMITTED', 'NOT_SUBMITTED', 'ERROR']")
    assert response.status_code == 422
    assert response.json() == rct_wrong_parameter_response

@pytest.mark.skip
@pytest.mark.rct_api
def test_rct_gateway_missing_exp_market(payload_header_provider,
                                        rct_design_table_file_provider,
                                        rct_experiment_data_sample_provider,
                                        rct_experiment_data_provider,
                                        mocker):
    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
    del local_rct_experiment_data_sample_provider['EXP_MARKET']
    experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
    response = client.post(
        "/rct-gateway",
        data=experiment_data,
        files=rct_design_table_file_provider,
        headers=payload_header_provider
    )
    assert response.status_code == 422
    assert response.json() == rct_payload_missing_param('EXP_MARKET')

@pytest.mark.skip
@pytest.mark.rct_api
def test_rct_gateway_bad_exp_market(payload_header_provider,
                                    rct_design_table_file_provider,
                                    rct_experiment_data_sample_provider,
                                    rct_experiment_data_provider,
                                    mocker):
    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
    bad_exp_market = str(secrets.randbelow(999))
    local_rct_experiment_data_sample_provider.update({'EXP_MARKET': bad_exp_market})
    experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
    valid_markets = ["Europe", "Mexico", "Colombia", "Peru", "Panama", "Ecuador", "USA", "Canada", "Africa", "Vietnam", "Argentina", "Honduras", "El_Salvador", "Dominican_Republic", "China", "Korea", "Brazil", "Tanzania", "Uganda", "South_Africa", "Paraguay", "Uruguay"]
    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
    response = client.post(
        "/rct-gateway",
        data=experiment_data,
        files=rct_design_table_file_provider,
        headers=payload_header_provider
    )
    assert response.status_code == 422
    assert response.json() == rct_wrong_parameter('EXP_MARKET', bad_exp_market, valid_markets)

@pytest.mark.skip
@pytest.mark.rct_api
def test_rct_gateway_missing_exp_type(payload_header_provider,
                                      rct_design_table_file_provider,
                                      rct_experiment_data_sample_provider,
                                      rct_experiment_data_provider,
                                      mocker):
    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
    del local_rct_experiment_data_sample_provider['EXP_TYPE']
    experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
    response = client.post(
        "/rct-gateway",
        data=experiment_data,
        files=rct_design_table_file_provider,
        headers=payload_header_provider
    )
    assert response.status_code == 422
    assert response.json() == rct_payload_missing_param('EXP_TYPE')

@pytest.mark.skip
@pytest.mark.rct_api
def test_rct_gateway_bad_exp_type(payload_header_provider,
                                  rct_design_table_file_provider,
                                  rct_experiment_data_sample_provider,
                                  rct_experiment_data_provider,
                                  mocker):
    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
    bad_exp_type = str(secrets.randbelow(999))
    local_rct_experiment_data_sample_provider.update({'EXP_TYPE': bad_exp_type})
    experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
    response = client.post(
        "/rct-gateway",
        data=experiment_data,
        files=rct_design_table_file_provider,
        headers=payload_header_provider
    )
    assert response.status_code == 422
    assert response.json() == rct_wrong_parameter('EXP_TYPE', bad_exp_type,
                                                  "['Promotion', 'Portfolio', 'Suggester_Order_Upsell', 'Credit_Risk_Assessment', 'BEES_Engage', 'Algo_Selling', 'Algo_Tasking', 'Adhoc', 'Test_Experiment']")

@pytest.mark.skip
@pytest.mark.rct_api
def test_rct_gateway_slash_dates_format(payload_header_provider,
                                        rct_design_table_file_provider,
                                        rct_experiment_data_sample_provider,
                                        rct_experiment_data_provider,
                                        mocker):
    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
    date_keys = [key for key in local_rct_experiment_data_sample_provider.keys() if '_DATE' in key]
    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
    for date_key in date_keys:
        date_mutation = local_rct_experiment_data_sample_provider.get(date_key).replace("-", "/")
        local_rct_experiment_data_sample_provider.update({date_key: date_mutation})
        experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
        response = client.post(
            "/rct-gateway",
            data=experiment_data,
            files=rct_design_table_file_provider,
            headers=payload_header_provider
        )
        assert response.status_code == 422
        assert response.json() == rct_wrong_date_syntax(date_mutation)
        local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()

@pytest.mark.skip
@pytest.mark.rct_api
def test_rct_gateway_only_numeric_dates_format(payload_header_provider,
                                               rct_design_table_file_provider,
                                               rct_experiment_data_sample_provider,
                                               rct_experiment_data_provider,
                                               mocker):
    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
    date_keys = [key for key in local_rct_experiment_data_sample_provider.keys() if '_DATE' in key]
    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
    for date_key in date_keys:
        date_mutation = local_rct_experiment_data_sample_provider.get(date_key).replace("-", "")
        local_rct_experiment_data_sample_provider.update({date_key: date_mutation})
        experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
        response = client.post(
            "/rct-gateway",
            data=experiment_data,
            files=rct_design_table_file_provider,
            headers=payload_header_provider
        )
        assert response.status_code == 422
        assert response.json() == rct_wrong_date_syntax(date_mutation)
        local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()

@pytest.mark.skip
@pytest.mark.rct_api
def test_rct_gateway_date_deltas(payload_header_provider,
                                 rct_design_table_file_provider,
                                 rct_experiment_data_sample_provider,
                                 rct_experiment_data_provider,
                                 mocker):
    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
    start_dates = [key for key in local_rct_experiment_data_sample_provider.keys() if 'START_DATE' in key]
    end_dates = [key for key in local_rct_experiment_data_sample_provider.keys() if 'END_DATE' in key]
    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
    for start, end in zip(start_dates, end_dates):
        end_date_object = datetime.strptime(local_rct_experiment_data_sample_provider.get(end), TIME_FORMAT)
        date_mutation = end_date_object + timedelta(days=secrets.randbelow(365 + 1))
        local_rct_experiment_data_sample_provider.update({start: date_mutation.strftime(TIME_FORMAT)})
        experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
        response = client.post(
            "/rct-gateway",
            data=experiment_data,
            files=rct_design_table_file_provider,
            headers=payload_header_provider
        )
        assert response.status_code == 422
        assert response.json() == rct_wrong_date_delta()
        local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()

@pytest.mark.skip
@pytest.mark.rct_api
def test_rct_gateway_date_range(payload_header_provider,
                                rct_design_table_file_provider,
                                rct_experiment_data_sample_provider,
                                rct_experiment_data_provider,
                                mocker):
    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
    date_keys = [key for key in local_rct_experiment_data_sample_provider.keys() if '_DATE' in key]
    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
    for date_key in date_keys:
        date_key_object = datetime.strptime(local_rct_experiment_data_sample_provider.get(date_key), TIME_FORMAT)
        delta = date_key_object - datetime.strptime("2020-12-31", TIME_FORMAT)
        date_mutation = date_key_object - timedelta(days=abs(delta.days + 1))
        local_rct_experiment_data_sample_provider.update({date_key: date_mutation.strftime(TIME_FORMAT)})
        experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
        response = client.post(
            "/rct-gateway",
            data=experiment_data,
            files=rct_design_table_file_provider,
            headers=payload_header_provider
        )
        assert response.status_code == 422
        assert response.json() == rct_wrong_date_ranges()
        local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
    for date_key in date_keys:
        date_key_object = datetime.strptime(local_rct_experiment_data_sample_provider.get(date_key), TIME_FORMAT)
        delta = datetime.today() - date_key_object
        date_mutation = datetime.today() + timedelta(days=abs(delta.days + 1))
        local_rct_experiment_data_sample_provider.update({date_key: date_mutation.strftime(TIME_FORMAT)})
        experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
        response = client.post(
            "/rct-gateway",
            data=experiment_data,
            files=rct_design_table_file_provider,
            headers=payload_header_provider
        )
        assert response.status_code == 422
        assert response.json() == rct_wrong_date_ranges()
        local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
