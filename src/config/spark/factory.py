"""
Create spark session

Copyright 2024 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


# Spark
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql import functions as F


class SparkSessionFactory():

  @staticmethod
  def get_session():
    conf = SparkConf().setAppName("RCTAPI").setMaster("local[*]")
    # Set the spark.metrics.namespace property
    #spark.sparkContext.setLogLevel("INFO")
    # spark.conf.set("spark.executor.memory", "8g")
    # spark.conf.set("spark.executor.cores", 4)
    spark = SparkSession.builder \
        .appName("RCT-API") \
        .config("spark.metrics.namespace", "rct-api") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.jars.packages",
                "com.databricks:databricks-jdbc:2.6.36,"
                + "org.apache.logging.log4j:log4j-api:2.23.1,"
                + "io.delta:delta-core_2.12:2.4.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    return spark



       # .config("spark.jars.packages", ",io.delta:delta-core_2.12:2.2.0")\
       # .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
       # .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \

       # .config("spark.driver.extraClassPath", "/opt/workspace/databricks-jdbc-2.6.34-sources.jar") \
       # .config("spark.driver.extraLibrary", "/opt/workspace/databricks-jdbc-2.6.34-sources.jar") \

#.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
#.config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
#.config("spark.jars.packages", "io.delta:delta-core_2.12:1.2.1") \
#.config("spark.jars.repositories", "https://maven-central.storage-download.googleapis.com/maven2/") \

#.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
#.config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
