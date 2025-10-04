"""
RCT API Model definition.

Copyright 2023 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime, date, timedelta
import operator
import pydantic
import json


class ExperimentData(BaseModel):
    """
    Schema definition for ExperimentData object
    """

    EXP_ID: Optional[int]
    EXP_NAME: str
    EXP_STATUS: str
    EXP_MESSAGE: str
    EXP_MARKET: str
    EXP_TYPE: str
    EXP_UNIT: str
    EXPERIMENT_START_DATE: Optional[date]
    EXPERIMENT_END_DATE: Optional[date]
    BASELINE_START_DATE: Optional[date]
    BASELINE_END_DATE: Optional[date]
    ANALYSIS_START_DATE: Optional[date]
    ANALYSIS_END_DATE: Optional[date]
    BLOCKING_FACTORS: Optional[List[dict]]
    EXPERIMENTAL_FACTORS: dict
    SCOPE_SIZE: int
    CREATED_BY_NAME: str
    CREATED_BY_EMAIL: str
    PRIMARY_OUTCOME_NAME: Optional[str]
    is_outlier_required: bool = False

    @classmethod
    def __get_validators__(cls):
        # Parse object to JSON
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        # Check if input is a valid dictionary
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

    class Config:
        validate_assignment = True


    @validator('EXP_STATUS', allow_reuse=True)
    def check_exp_status(cls, EXP_STATUS):
        """
            Validator for EXP_STATUS field
        """
        valid_exp_status = ["IN_PROGRESS", "NOT_STARTED", "COMPLETED", "SUBMITTED", "NOT_SUBMITTED", "ERROR"]
        if EXP_STATUS in valid_exp_status:
            return EXP_STATUS
        else:
            raise ValueError("Invalid EXP_STATUS value:{}, should be: {}".format(EXP_STATUS, valid_exp_status))

    @validator('EXP_MARKET', allow_reuse=True)
    def check_exp_market(cls, EXP_MARKET):
        """
            Validator for EXP_MARKET field
        """
        valid_markets = ["Europe", "Mexico", "Colombia", "Peru", "Panama", "Ecuador", "USA", "Canada", "Africa", "Vietnam", "Argentina", "Honduras", "El_Salvador", "Dominican_Republic", "China", "Korea", "Brazil", "Tanzania", "Uganda", "South_Africa", "Paraguay", "Uruguay"]
        if EXP_MARKET in valid_markets:
            return EXP_MARKET
        else:
            raise ValueError("Invalid EXP_MARKET value:{}, should be: {}".format(EXP_MARKET, valid_markets))


    @validator('EXP_TYPE', allow_reuse=True)
    def check_exp_type(cls, EXP_TYPE):
        """
            Validator for EXP_TYPE field
        """
        valid_exp_types = ["Promotion","Portfolio","Suggester_Order_Upsell","Credit_Risk_Assessment","BEES_Engage","Algo_Selling","Algo_Tasking","Adhoc","Test_Experiment"]
        if EXP_TYPE in valid_exp_types:
            return EXP_TYPE
        else:
            raise ValueError("Invalid EXP_TYPE value:{}, should be: {}".format(EXP_TYPE, valid_exp_types))

   # @validator("EXPERIMENT_START_DATE",
   #            "EXPERIMENT_END_DATE",
   #            "BASELINE_START_DATE",
   #            "BASELINE_END_DATE",
   #            "ANALYSIS_START_DATE",
   #            "ANALYSIS_END_DATE"
   #            , pre=False, always=False, allow_reuse=True)
    def check_dates(cls, value):
        """
            Validator for correct date strings
        """
        try:
            if value != None:
                 datetime.strptime(value,'%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect data format, should be: YYYY-MM-DD")

    #TODO:
    # Validate dates correlation, precedence etc,
    #@root_validator(pre=True, allow_reuse=True)
    def check_dates_consistency(cls, values):
        """
            Validator for check dates consistency
            Criteria:
            - Date fields with suffix START_DATE must be less than or equal to fields with suffix END_DATE
            - All date fields must be in range from 2020-12-31 to TODAY(UTC)
        """
        #Set global variables to date ranges comparation
        start_date = datetime.strptime("2020-12-31",'%Y-%m-%d')
        end_date = datetime.today()  + timedelta(days=90)

        #Get payload date variables
        EXPERIMENT_START_DATE = datetime.strptime(values.get("EXPERIMENT_START_DATE"),'%Y-%m-%d') if values.get("EXPERIMENT_START_DATE") is not None else datetime.today().strftime('%Y-%m-%d')
        EXPERIMENT_END_DATE   = datetime.strptime(values.get("EXPERIMENT_END_DATE"),'%Y-%m-%d')   if values.get("EXPERIMENT_END_DATE") is not None else datetime.today().strftime('%Y-%m-%d')
        BASELINE_START_DATE   = datetime.strptime(values.get("BASELINE_START_DATE"),'%Y-%m-%d')   if values.get("BASELINE_START_DATE") is not None else datetime.today().strftime('%Y-%m-%d')
        BASELINE_END_DATE     = datetime.strptime(values.get("BASELINE_END_DATE"),'%Y-%m-%d')     if values.get("BASELINE_END_DATE")   is not None else datetime.today().strftime('%Y-%m-%d')
        ANALYSIS_START_DATE   = datetime.strptime(values.get("ANALYSIS_START_DATE"),'%Y-%m-%d')   if values.get("ANALYSIS_START_DATE") is not None else datetime.today().strftime('%Y-%m-%d')
        ANALYSIS_END_DATE     = datetime.strptime(values.get("ANALYSIS_END_DATE"),'%Y-%m-%d')     if values.get("ANALYSIS_END_DATE")   is not None else datetime.today().strftime('%Y-%m-%d')

        #Boolean flags to check dates comparations
        valid_date_delta = False
        valid_date_ranges= False
        #Validate date deltas
        if  operator.le(EXPERIMENT_START_DATE, EXPERIMENT_END_DATE) and \
            operator.le(BASELINE_START_DATE, BASELINE_END_DATE) and \
            operator.le(ANALYSIS_START_DATE, ANALYSIS_END_DATE):
            valid_date_delta = True
        #Validate date ranges
        if  start_date <= EXPERIMENT_START_DATE <= end_date and \
            start_date <= EXPERIMENT_END_DATE   <= end_date and \
            start_date <= BASELINE_START_DATE   <= end_date and \
            start_date <= BASELINE_END_DATE     <= end_date and \
            start_date <= ANALYSIS_START_DATE   <= end_date and \
            start_date <= ANALYSIS_END_DATE     <= end_date:
            valid_date_ranges = True

        if not valid_date_delta:
            raise ValueError("Invalid date delta detected,"\
                    "fields with suffix START_DATE must be less than or equal to fields with suffix END_DATE ")

        if not valid_date_ranges:
            raise ValueError("Invalid date ranges detected,"\
                    "dates must be in range {} to {}".format(start_date, end_date.strftime('%Y-%m-%d')))
        return values


    @root_validator(allow_reuse=True)
    def check_experimental_factors(cls, values ):
        """
            Validator for EXPERIMENTAL_FACTORS field
            - Sum of level_sample_allocation list must be less or equal to SCOPE_SIZE
        """
        EXPERIMENTAL_FACTORS = values.get("EXPERIMENTAL_FACTORS")
        SCOPE_SIZE = values.get("SCOPE_SIZE")
        TOTAL_ALLOCATION = sum([*EXPERIMENTAL_FACTORS.values() ][0][1])

        if type(EXPERIMENTAL_FACTORS) == dict:
            for key in EXPERIMENTAL_FACTORS.keys():
                level_sample_allocation_sum_values = sum(EXPERIMENTAL_FACTORS.get(key)[1])
                # Ajust according needs, max value would be 1.0 or SCOPE_SIZE
                is_weight_type_int = all(isinstance(x, int) for x in EXPERIMENTAL_FACTORS.get(key)[1] )

                if is_weight_type_int:
                    if not(level_sample_allocation_sum_values <= SCOPE_SIZE):
                        raise ValueError("Invalid level_sample_allocation list of EXPERIMENTAL_FACTORS, sum:{},"\
                                         "must be less or equals than SCOPE_SIZE: {}".format(level_sample_allocation_sum_values, SCOPE_SIZE ))
                else:
                    if not(level_sample_allocation_sum_values == 1.0 ):
                        #raise ValueError("Invalid level_sample_allocation list of EXPERIMENTAL_FACTORS, sum:{},"\
                                #                 "must be less or equal than SCOPE_SIZE: {}".format(level_sample_allocation_sum_values, SCOPE_SIZE))
                        raise ValueError("Invalid level_sample_allocation list of EXPERIMENTAL_FACTORS, sum:{},"\
                                         "must be equal than: {}".format(level_sample_allocation_sum_values, 1.0))


                #if not(level_sample_allocation_sum_values == TOTAL_ALLOCATION):
                #    raise ValueError("Invalid level_sample_allocation list of EXPERIMENTAL_FACTORS, sum:{},"\
                #                     "must equals to TOTAL_ALLOCATION: {}".format(level_sample_allocation_sum_values, TOTAL_ALLOCATION))
            return values
        else:
            raise ValueError("Invalid EXPERIMENTAL_FACTORS type:{}, should be: {}".format(type(EXPERIMENTAL_FACTORS) , type({})) )


    @validator('SCOPE_SIZE', allow_reuse=True)
    def check_scope_size(cls, SCOPE_SIZE):
        """
            Validator for SCOPE_SIZE field
        """
        if type(SCOPE_SIZE) == int:
            return SCOPE_SIZE
        else:
            raise ValueError("Invalid SCOPE_SIZE type:{}, should be: {}".format(type(SCOPE_SIZE),type(int(0))))

    # Consider use Microsoft Active Directory to validate identity
    @validator('CREATED_BY_EMAIL', allow_reuse=True)
    def check_created_by_email(cls, CREATED_BY_EMAIL):
        """
            Validator for CREATED_BY_EMAIL field
        """
        authorized_domains = ["ab-inbev.com"]
        CREATED_BY_EMAIL_PIECES = CREATED_BY_EMAIL.split("@")

        if len(CREATED_BY_EMAIL_PIECES) == 2 and CREATED_BY_EMAIL_PIECES[1] in authorized_domains:
            return CREATED_BY_EMAIL
        else:
            raise ValueError("Invalid CREATED_BY_EMAIL value:{}, should be: someone@{}".format(CREATED_BY_EMAIL, authorized_domains ))


class RCTExperimentData(BaseModel):
    """
    Schema definition for ExperimentData object
    """

    EXP_ID: int
    EXPERIMENTAL_FACTORS: dict

    geographical_filters: List[dict]
    product_filters: List[dict]
    blocking_factors: List[str]
    metric: str
    geographical_granularity: str
    zone_code: str
    country_code: str
    base_line_start_date: date
    base_line_end_date: date
    experiment_unit: str
    experiment_market: str
    metric: str
    is_outlier_required: bool = False


class ObservationExperimentData(BaseModel):
    """
    Schema definition for Observational study payload
    """
    EXP_ID: int
    BLOCKING_FACTORS: Optional[List[dict]]
    EXPERIMENTAL_FACTORS: dict
    SCOPE_SIZE: int
    PRIMARY_OUTCOME_NAME: Optional[str]
    file_name: str
    column_mappings: List[dict]