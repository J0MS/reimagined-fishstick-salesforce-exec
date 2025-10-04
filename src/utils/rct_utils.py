"""
RCT API Utils

Copyright 2023 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


# import external libraries.
import logging

import pyspark.sql.functions as F
import pyspark.sql.types as types
from pyspark.sql import DataFrame

from ..config.config import LoggingFormatter


# Instantiate logger for RCT Utils
logger = logging.getLogger("RCT_UTILS")
logger.setLevel(logging.DEBUG)
# Instantiate console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
# Instantiate custom LogginFormater
console_handler.setFormatter(LoggingFormatter())
# Set configured handler
logger.addHandler(console_handler)


class RCTTools:
    """
    Module to define RCT tools
    """

    @staticmethod
    def check_input_layout(design_table: DataFrame) -> DataFrame:
        """
        Method to validate design_table dataframe
        Args:
            design_table: pyspark.sql.DataFrame
        Returns:
            pyspark.sql.DataFrame
        """
        required_columns = [
            "ex_factor_1",
            "ex_factor_2",
            "ex_factor_3",
            "ex_factor_4",
            "ex_factor_5",
            "block_factor_1",
            "block_factor_2",
            "block_factor_3",
            "block_factor_4",
            "block_factor_5",
        ]

        for column in required_columns:
            if column in design_table.columns:
                design_table = design_table.withColumn(
                    column,
                    F.when(F.col(column).isNotNull(), F.col(column))
                    .otherwise(None)
                    .cast(types.StringType()),
                )
            else:
                design_table = design_table.withColumn(
                    column, F.lit(None).cast(types.StringType())
                )

        return design_table

    @staticmethod
    def format_dates(experiment: dict, dates_list: list, date_format: str) -> dict:
        """
        Method to format dates
        Args:
            experiment: dict -> EOR for RCT
        Returns:
            dict
        """
        for date in dates_list:
            try:
                experiment.update({date: experiment.get(date).strftime(date_format)})
            except Exception as e:
                logger.warning(
                    "{} does not exist or is not a valid datetime.date() object".format(
                        date
                    )
                )
                continue
        return experiment


    @staticmethod
    def standarize_weights(experiment: dict) -> dict:
        """
        Method to standarize weights
        Args:
            experiment: dict -> EOR for RCT
        Returns:
            dict
        """
        level_name = [*experiment.get("EXPERIMENTAL_FACTORS").keys()][0]
        first_factor = list(experiment.get("EXPERIMENTAL_FACTORS"))
        experimental_factors = experiment.get("EXPERIMENTAL_FACTORS").get(first_factor[0])
        groups_weights = experimental_factors[1]
        logger.info("Current weights: {}".format(groups_weights))
        is_weight_type_int = all(isinstance(x, int) for x in groups_weights)

        if is_weight_type_int:
            logger.info("Weights type integer")
            sum_of_weights = sum(groups_weights)
            # Convert group weights to proportions of sum_of_weights
            proportions = [element / sum_of_weights for element in groups_weights]
            updated_experimental_factor = experiment.get("EXPERIMENTAL_FACTORS").get(first_factor[0])
            logger.info("Raw experimental factor")
            logger.info(updated_experimental_factor)
            updated_experimental_factor[1] = proportions
            logger.info("Processed experimental factor")
            experiment.update( {"EXPERIMENTAL_FACTORS": {level_name : updated_experimental_factor}})
            logger.info(experiment)
        else:
            logger.info("Weights type float")

        return experiment

