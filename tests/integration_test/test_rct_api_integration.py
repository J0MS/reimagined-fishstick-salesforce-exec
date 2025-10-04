"""
Random Control Trial API-Integration Test cases

Unitary test.

Copyright 2023 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import json
import os
import sys
import secrets
import requests
import pytest
from unittest import mock
from datetime import datetime, date, timedelta
from importlib.machinery import SourceFileLoader

sys.path.append('../../src/')

#from src.main import rct_api
#from src.config import settings
#from src.models.model import ExperimentObject, ExperimentData, RCTGatewayResponse, RCTResponseValidator
#
#INTEGRATION_TEST_URL = settings.INTEGRATION_TEST_URL
#
#""" Module helper functions """
#
#TIME_FORMAT='%Y-%m-%d'
## Error respose helper functions
#def rct_wrong_date_syntax(date_mutation:str) -> dict:
#    """ Provide wrong date syntax response"""
#    return {'detail': [{'loc': ['body', 'experiment_data', '__root__'],
#                        'msg': "time data '{}' does not match format '{}'".format(date_mutation,TIME_FORMAT),
#                        'type': 'value_error'}]}
#
#def rct_wrong_date_delta() -> dict:
#    """ Provide wrong date delta response"""
#    return {'detail': [{'loc': ['body', 'experiment_data', '__root__'],
#                        'msg':
#                        'Invalid date delta detected,'\
#                        'fields with suffix START_DATE must be less than or equal to fields with suffix END_DATE ',
#                        'type': 'value_error'
#                        }]}
#
#def rct_wrong_date_ranges() -> dict:
#    """ Provide wrong date ranges response"""
#    return {'detail': [{'loc': ['body', 'experiment_data', '__root__'],
#                         'msg':
#                         'Invalid date ranges detected,dates must be in range 2020-12-31 00:00:00'\
#                         'to {}'.format(datetime.today().strftime(TIME_FORMAT)),
#                         'type': 'value_error'}
#                       ]}
#
#def rct_payload_missing_param(missing_parameter:str) -> dict:
#    """ Provide missing experimental parameter response"""
#    error_response = {'detail': [{'loc': ['body', 'experiment_data'],
#                       'msg': 'field required',
#                       'type': 'value_error.missing'}]}
#    body_key =  error_response.get('detail')[0].get('loc')
#    body_key.insert(2, missing_parameter )
#    new_detail = [{'loc': body_key,
#                   'msg': error_response.get('detail')[0].get('msg'),
#                   'type': error_response.get('detail')[0].get('type')
#                 }]
#    error_response.update({'detail': new_detail})
#    return error_response
#
#
#def rct_wrong_parameter(target_param:str,bad_param:str,correct_param:str ) -> dict:
#    loc_list = ['body', 'experiment_data']
#    loc_list.append(target_param)
#    return {'detail': [{'loc': loc_list,
#                        'msg': "Invalid {} value:{}, should be: {}".format(target_param,bad_param,correct_param),
#                        'type': 'value_error'}
#                       ]}
#
#
#""" RCT API test operations"""
#
#@pytest.mark.rct_api
#def test_health(payload_header_provider):
#    """ Test response for /health path."""
#    response = requests.get(
#        "{}/health".format(INTEGRATION_TEST_URL),
#        headers=payload_header_provider
#    )
#    assert len(response.json()) == 3
#    assert response.status_code == 200
#    assert response.json().get('state') == "SUCCESS!"
#    assert response.json().get('health') == "ok"


#@pytest.mark.rct_api
#def test_rct_gateway_sucess(payload_header_provider,
#                            rct_experiment_data_provider_as_dict,
#                            rct_design_table_file_path_provider,
#                            rct_design_table_file_provider,
#                            rct_experiment_data_sample_provider,
#                            mocker
#                            ):
#    """ Test response for rct-gateway endpoint on success"""
#
#    
#    experiment_data = {'experiment_data' : json.dumps( rct_experiment_data_sample_provider   ) }
#
#    response = requests.post(
#                    "{}/rct-gateway".format(INTEGRATION_TEST_URL),
#                    data=experiment_data,
#                    files=rct_design_table_file_provider,
#                    headers=payload_header_provider
#            )
#
#    assert len(response.json()) == 4
#    assert response.json().keys() == RCTGatewayResponse.schema().get("properties").keys()
#    assert type(response.json().get('statusCode')) == int
#    assert type(response.json().get('state')) == str
#    assert type(response.json().get('exp_id')) == int
#    assert type(response.json().get('rct_table')) == str
#    assert response.json().get('statusCode') == 201
#    assert response.json().get('state') == "SUCCESS"
#    assert int(response.json().get('exp_id')) > 0
#
#
#@pytest.mark.rct_api
#def test_rct_gateway_missing_exp_name(payload_header_provider,
#                                      rct_design_table_file_provider,
#                                      rct_experiment_data_sample_provider,
#                                      rct_experiment_data_provider,
#                                      mocker
#                                      ):
#    """ Test response for /rct-gateway endpoint on error when EXP_NAME is missing """
#    #Removing EXP_NAME
#    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
#    del local_rct_experiment_data_sample_provider['EXP_NAME']
#    experiment_data = {'experiment_data' : json.dumps(local_rct_experiment_data_sample_provider)}
#
#    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
#
#    response = requests.post(
#                    "{}/rct-gateway".format(INTEGRATION_TEST_URL),
#                    data=experiment_data,
#                    files=rct_design_table_file_provider,
#                    headers=payload_header_provider
#               )
#    assert len(response.json()) == 1
#    assert response.status_code == 422
#    assert response.json() == rct_payload_missing_param('EXP_NAME')
#
#
#@pytest.mark.rct_api
#def test_rct_gateway_missing_exp_status(payload_header_provider,
#                                        rct_design_table_file_provider,
#                                        rct_experiment_data_sample_provider,
#                                        rct_experiment_data_provider,
#                                        mocker
#                                        ):
#    """ Test response for /rct-gateway endpoint on error when EXP_STATUS is missing """
#    #Removing EXP_STATUS
#    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
#    del local_rct_experiment_data_sample_provider['EXP_STATUS']
#    experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
#
#    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
#
#    response = requests.post(
#                    "{}/rct-gateway".format(INTEGRATION_TEST_URL),
#                    data=experiment_data,
#                    files=rct_design_table_file_provider,
#                    headers=payload_header_provider
#
#            )
#    assert len(response.json()) == 1
#    assert response.status_code == 422
#    assert response.json() == rct_payload_missing_param('EXP_STATUS')
#
#
#@pytest.mark.rct_api
#def test_rct_gateway_bad_exp_status(payload_header_provider,
#                                    rct_design_table_file_provider,
#                                    rct_experiment_data_sample_provider,
#                                    rct_experiment_data_provider,
#                                    mocker
#                                    ):
#    """ Test response for /rct-gateway endpoint when EXP_STATUS is bad """
#
#    #Removing EXP_STATUS
#    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
#
#
#    # Compute random string as exp_status
#    bad_exp_status = str(secrets.randbelow(999))
#    target_parameter = 'EXP_STATUS'
#    local_rct_experiment_data_sample_provider.update({target_parameter: bad_exp_status})
#    experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
#
#    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
#
#    response = requests.post(
#                    "{}/rct-gateway".format(INTEGRATION_TEST_URL),
#                    data=experiment_data,
#                    files=rct_design_table_file_provider,
#                    headers=payload_header_provider
#            )
#    rct_wrong_parameter_response = rct_wrong_parameter(target_parameter,bad_exp_status,
#                                "['IN_PROGRESS', 'NOT_STARTED', 'COMPLETED', 'SUBMITTED', 'NOT_SUBMITTED', 'ERROR']")
#
#    assert len(response.json()) == 1
#    assert response.status_code == 422
#    assert response.json() == rct_wrong_parameter_response
#
#
#@pytest.mark.rct_api
#def test_rct_gateway_missing_exp_market(payload_header_provider,
#                                        rct_design_table_file_provider,
#                                        rct_experiment_data_sample_provider,
#                                        rct_experiment_data_provider,
#                                        mocker
#                                        ):
#    """ Test response for /rct-gateway endpoint when EXP_MARKET is missing """
#    #Removing EXP_MARKET
#    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
#    del local_rct_experiment_data_sample_provider['EXP_MARKET']
#    experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
#
#    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
#
#    response = requests.post(
#                    "{}/rct-gateway".format(INTEGRATION_TEST_URL),
#                    data=experiment_data,
#                    files=rct_design_table_file_provider,
#                    headers=payload_header_provider
#
#            )
#    assert len(response.json()) == 1
#    assert response.status_code == 422
#    assert response.json() == rct_payload_missing_param('EXP_MARKET')
#
#
#@pytest.mark.rct_api
#def test_rct_gateway_bad_exp_market(payload_header_provider,
#                                    rct_design_table_file_provider,
#                                    rct_experiment_data_sample_provider,
#                                    rct_experiment_data_provider,
#                                    mocker
#                                    ):
#    """ Test response for /rct-gateway endpoint when EXP_MARKET is bad """
#    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
#    #Updating EXP_MARKET
#    bad_exp_market = str(secrets.randbelow(999))
#    local_rct_experiment_data_sample_provider.update( {'EXP_MARKET': bad_exp_market} )
#    experiment_data = {'experiment_data' : json.dumps( local_rct_experiment_data_sample_provider)}
#    valid_markets = ["Europe", "Mexico", "Colombia", "Peru", "Panama", "Ecuador", "USA", "Canada", "Africa", "Vietnam", "Argentina", "Honduras", "El_Salvador", "Dominican_Republic", "China", "Korea", "Brazil", "Tanzania", "Uganda", "South_Africa", "Paraguay", "Uruguay"]
#
#    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
#
#    response = requests.post(
#                    "{}/rct-gateway".format(INTEGRATION_TEST_URL),
#                    data=experiment_data,
#                    files=rct_design_table_file_provider,
#                    headers=payload_header_provider
#
#            )
#    assert len(response.json()) == 1
#    assert response.status_code == 422
#    assert response.json() == rct_wrong_parameter('EXP_MARKET',
#                                                  bad_exp_market,
#                                                  valid_markets
#                                                  )
#
#    #"['Europe', 'Mexico', 'Colombia', 'Peru', 'Panama', 'Ecuador', 'USA', 'Canada', 'Africa', 'Vietnam', 'Argentina', 'Honduras', 'El_Salvador', 'Dominican_Republic', 'China', 'Korea', 'Brazil' , 'Tanzania', 'Uganda', 'South_Africa', 'Paraguay', 'Uruguay']"
#
#@pytest.mark.rct_api
#def test_rct_gateway_missing_exp_type(payload_header_provider,
#                                      rct_design_table_file_provider,
#                                      rct_experiment_data_sample_provider,
#                                      rct_experiment_data_provider,
#                                      mocker
#                                      ):
#    """ Test response for /rct-gateway endpoint when EXP_TYPE is missing """
#    #Removing EXP_TYPE
#    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
#    del local_rct_experiment_data_sample_provider['EXP_TYPE']
#    experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
#
#    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
#
#    response = requests.post(
#                    "{}/rct-gateway".format(INTEGRATION_TEST_URL),
#                    data=experiment_data,
#                    files=rct_design_table_file_provider,
#                    headers=payload_header_provider
#
#            )
#    assert len(response.json()) == 1
#    assert response.status_code == 422
#    assert response.json() == rct_payload_missing_param('EXP_TYPE')
#
#
#@pytest.mark.rct_api
#def test_rct_gateway_bad_exp_type(payload_header_provider,
#                                  rct_design_table_file_provider,
#                                  rct_experiment_data_sample_provider,
#                                  rct_experiment_data_provider,
#                                  mocker
#                                  ):
#    """ Test response for /rct-gateway endpoint when EXP_TYPE is bad """
#
#    #Removing EXP_TYPE
#    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
#    bad_exp_type = str(secrets.randbelow(999))
#    local_rct_experiment_data_sample_provider.update({'EXP_TYPE': bad_exp_type})
#    experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
#
#    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
#
#    response = requests.post(
#                    "{}/rct-gateway".format(INTEGRATION_TEST_URL),
#                    data=experiment_data,
#                    files=rct_design_table_file_provider,
#                    headers=payload_header_provider
#            )
#    assert len(response.json()) == 1
#    assert response.status_code == 422
#    assert response.json() == rct_wrong_parameter('EXP_TYPE',
#                                                  bad_exp_type,
#                                                  "['Promotion', "
#                                                  "'Portfolio', "
#                                                  "'Suggester_Order_Upsell', "
#                                                  "'Credit_Risk_Assessment', "
#                                                  "'BEES_Engage', "
#                                                  "'Algo_Selling', "
#                                                  "'Algo_Tasking', "
#                                                  "'Adhoc', "
#                                                  "'Test_Experiment']"
#                                                  )
#
#
#@pytest.mark.rct_api
#def test_rct_gateway_slash_dates_format(payload_header_provider,
#                                        rct_design_table_file_provider,
#                                        rct_experiment_data_sample_provider,
#                                        rct_experiment_data_provider,
#                                        mocker
#                                        ):
#    """ Test response for /rct-gateway endpoint when some date field is bad """
#
#    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
#    experiment_data_keys = [*local_rct_experiment_data_sample_provider.keys()]
#    date_keys = [key for key in experiment_data_keys if '_DATE' in key]
#
#    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
#
#    for date_key in date_keys:
#        date_mutation = local_rct_experiment_data_sample_provider.get(date_key).replace("-","/")
#        local_rct_experiment_data_sample_provider.update({date_key: date_mutation})
#        experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
#
#        response = requests.post(
#                        "{}/rct-gateway".format(INTEGRATION_TEST_URL),
#                        data=experiment_data,
#                        files=rct_design_table_file_provider,
#                        headers=payload_header_provider
#                )
#        assert response.json() != rct_wrong_date_syntax(date_mutation)
#        local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
#
#
#@pytest.mark.rct_api
#def test_rct_gateway_only_numeric_dates_format(payload_header_provider,
#                                               rct_design_table_file_provider,
#                                               rct_experiment_data_sample_provider,
#                                               rct_experiment_data_provider,
#                                               mocker
#                                               ):
#    """ Test response for /rct-gateway endpoint when some date field is bad """
#
#    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
#    experiment_data_keys = [*local_rct_experiment_data_sample_provider.keys()]
#    date_keys = [key for key in experiment_data_keys if '_DATE' in key]
#
#    for date_key in date_keys[:1]:
#        date_mutation = local_rct_experiment_data_sample_provider.get(date_key).replace("-", "" )
#        local_rct_experiment_data_sample_provider.update({date_key: date_mutation})
#        experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
#
#        response = requests.post(
#                        "{}/rct-gateway".format(INTEGRATION_TEST_URL),
#                        data=experiment_data,
#                        files=rct_design_table_file_provider,
#                        headers=payload_header_provider
#                )
#
#        assert response.json() != rct_wrong_date_syntax(date_mutation)
#        local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
#        #rct_design_table_file_provider.get("design_table_file").close()
#
#@pytest.mark.rct_api
#def test_rct_gateway_date_deltas(payload_header_provider,
#                                 rct_design_table_file_provider,
#                                 rct_experiment_data_sample_provider,
#                                 rct_experiment_data_provider,
#                                 mocker
#                                 ):
#    """ Test response for /rct-gateway endpoint when some date field is bad """
#
#    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
#    experiment_data_keys = [*local_rct_experiment_data_sample_provider.keys()]
#    start_dates = [key for key in experiment_data_keys if 'START_DATE' in key]
#    end_dates  = [key for key in experiment_data_keys if 'END_DATE' in key]
#
#    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
#
#    for start, end in zip(start_dates[:1], end_dates[:1]):
#        end_date_object = datetime.strptime(local_rct_experiment_data_sample_provider.get(end),TIME_FORMAT)
#        date_mutation = end_date_object + timedelta(days=secrets.randbelow(365+1))
#
#        local_rct_experiment_data_sample_provider.update({start: date_mutation.strftime(TIME_FORMAT)})
#        experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
#
#        response = requests.post(
#                        "{}/rct-gateway".format(INTEGRATION_TEST_URL),
#                        data=experiment_data,
#                        files=rct_design_table_file_provider,
#                        headers=payload_header_provider
#                )
#        print("Error: ",response)
#        assert response.json() != rct_wrong_date_delta()
#        local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
#
#
#@pytest.mark.skip
#@pytest.mark.rct_api
#def test_rct_gateway_date_range(payload_header_provider,
#                                rct_design_table_file_provider,
#                                rct_experiment_data_sample_provider,
#                                rct_experiment_data_provider,
#                                mocker
#                                ):
#    """ Test response for /rct-gateway endpoint when some date field is outside of the lower limit """
#    local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
#    experiment_data_keys = [*local_rct_experiment_data_sample_provider.keys()]
#    date_keys = [key for key in experiment_data_keys if '_DATE' in key]
#
#    mocker.patch('src.main.rct_gateway', return_value=rct_experiment_data_provider)
#
#    for date_key in date_keys[:1]:
#        date_key_object = datetime.strptime(local_rct_experiment_data_sample_provider.get(date_key),TIME_FORMAT)
#        delta = date_key_object - datetime.strptime("2020-12-31",TIME_FORMAT)
#        date_mutation = date_key_object - timedelta(days=abs(delta.days + 1))
#
#        local_rct_experiment_data_sample_provider.update({date_key: date_mutation.strftime(TIME_FORMAT)})
#        experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
#
#        response = requests.post(
#                        "{}/rct-gateway".format(INTEGRATION_TEST_URL),
#                        data=experiment_data,
#                        files=rct_design_table_file_provider,
#                        headers=payload_header_provider
#                )
#        assert response.json() != rct_wrong_date_ranges() 
#        assert response.json() != rct_wrong_date_delta()
#        local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()
#
#    for date_key in date_keys[:1]:
#        date_key_object = datetime.strptime(local_rct_experiment_data_sample_provider.get(date_key),TIME_FORMAT)
#        delta = datetime.today() - date_key_object
#        date_mutation = datetime.today() + timedelta(days=abs(delta.days + 1 ))
#
#        local_rct_experiment_data_sample_provider.update({date_key: date_mutation.strftime(TIME_FORMAT)})
#        experiment_data = {'experiment_data': json.dumps(local_rct_experiment_data_sample_provider)}
#        print("frontera", delta.days, date_key_object, date_mutation, datetime.today())
#
#        response = requests.post(
#                        "{}/rct-gateway".format(INTEGRATION_TEST_URL),
#                        data=experiment_data,
#                        files=rct_design_table_file_provider,
#                        headers=payload_header_provider
#                )
#        assert response.json() != rct_wrong_date_ranges() 
#        assert response.json() != rct_wrong_date_delta()
#        local_rct_experiment_data_sample_provider = rct_experiment_data_sample_provider.copy()


