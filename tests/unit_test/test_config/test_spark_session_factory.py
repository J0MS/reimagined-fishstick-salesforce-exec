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
from src.config.spark.factory import SparkSessionFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_spark_session_creation():
    """
    Test the creation of a Spark session.
    """
    logger.info("Starting test_spark_session_creation")
    try:
        spark = SparkSessionFactory.get_session()
        logger.info("Spark session created successfully")
        assert isinstance(spark, SparkSession)
        assert spark.conf.get("spark.app.name") == "RCT-API"
        assert spark.conf.get("spark.metrics.namespace") == "rct-api"
        assert spark.conf.get("spark.sql.execution.arrow.pyspark.enabled") == "true"
        assert "com.databricks:databricks-jdbc:2.6.36" in spark.conf.get("spark.jars.packages")
    except Exception as e:
        logger.error(f"Error creating Spark session: {e}")
        pytest.fail("Spark session creation failed")

def test_spark_session_configurations():
    """
    Test the configurations of the Spark session.
    """
    logger.info("Starting test_spark_session_configurations")
    spark = SparkSessionFactory.get_session()
    try:
        config = spark.sparkContext.getConf()
        logger.info("Spark session configurations retrieved successfully")
        assert config.get("spark.sql.catalog.spark_catalog") == "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        assert config.get("spark.sql.extensions") == "io.delta.sql.DeltaSparkSessionExtension"
    except Exception as e:
        logger.error(f"Error retrieving Spark session configurations: {e}")
        pytest.fail("Failed to retrieve Spark session configurations")

def test_spark_session_with_data():
    """
    Test Spark session by performing a simple data operation.
    """
    logger.info("Starting test_spark_session_with_data")
    spark = SparkSessionFactory.get_session()
    try:
        data = [("Alice", 1), ("Bob", 2), ("Cathy", 3)]
        df = spark.createDataFrame(data, ["name", "value"])
        result = df.groupBy("name").count().collect()
        logger.info(f"Data operation results: {result}")
        assert len(result) == 3
    except Exception as e:
        logger.error(f"Error performing data operation: {e}")
        pytest.fail("Data operation in Spark session failed")


