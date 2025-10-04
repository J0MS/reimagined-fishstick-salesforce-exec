"""
Unit test RCT API health

Copyright 2024 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Libraries-------------- 

import logging
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType
from src.utils.rct_utils import RCTTools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.master("local[1]").appName("test").getOrCreate()

@pytest.fixture
def sample_dataframe(spark):
    schema = StructType([
        StructField("ex_factor_1", StringType(), True),
        StructField("ex_factor_2", StringType(), True),
        StructField("ex_factor_3", StringType(), True),
        StructField("ex_factor_4", StringType(), True),
        StructField("ex_factor_5", StringType(), True),
        StructField("block_factor_1", StringType(), True),
        StructField("block_factor_2", StringType(), True),
        StructField("block_factor_3", StringType(), True),
        StructField("block_factor_4", StringType(), True),
        StructField("block_factor_5", StringType(), True)
    ])
    data = [("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")]
    return spark.createDataFrame(data, schema)

def test_check_input_layout(sample_dataframe):
    """
    Test the check_input_layout method.
    """
    logger.info("Starting test_check_input_layout")
    try:
        result = RCTTools.check_input_layout(sample_dataframe)
        logger.info(f"Result: {result}")
        assert isinstance(result, DataFrame)
    except Exception as e:
        logger.error(f"Error in check_input_layout: {e}")
        pytest.fail("check_input_layout method failed")

def test_format_dates():
    """
    Test the format_dates method.
    """
    logger.info("Starting test_format_dates")
    experiment = {
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    }
    dates_list = ["start_date", "end_date"]
    date_format = "%Y-%m-%d"
    try:
        result = RCTTools.format_dates(experiment, dates_list, date_format)
        logger.info(f"Result: {result}")
        assert result["start_date"] == "2023-01-01"
        assert result["end_date"] == "2023-12-31"
    except Exception as e:
        logger.error(f"Error in format_dates: {e}")
        pytest.fail("format_dates method failed")

def test_standarize_weights():
    """
    Test the standarize_weights method.
    """
    logger.info("Starting test_standarize_weights")
    experiment = {
        "EXPERIMENTAL_FACTORS": {
            "factor1": [
                ["group1", "group2"],
                [1, 2],
                2
            ]
        }
    }
    try:
        result = RCTTools.standarize_weights(experiment)
        logger.info(f"Result: {result}")
        assert "EXPERIMENTAL_FACTORS" in result
        assert result["EXPERIMENTAL_FACTORS"]["factor1"][1] == [0.3333333333333333, 0.6666666666666666]
    except Exception as e:
        logger.error(f"Error in standarize_weights: {e}")
        pytest.fail("standarize_weights method failed")

