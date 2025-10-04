"""
Random Control Trial worker async engine.

Copyright 2023 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import pandas as pd
from celery import Celery, Task
from databricks import sql
from dataclasses import dataclass
# testops_tools package
from testops_tools.random_control_trial_2 import RCT
from testops_tools.outliers_treatment import OLT

from ...config.config import settings
from ...config.logger.factory import LoggingFactory

celery = Celery(__name__ )

celery.conf.broker_url = settings.RCT_API_CELERY_BROKER_URL
celery.conf.result_backend = settings.RCT_API_CELERY_RESULT_BACKEND

logger: logging.Logger = LoggingFactory().get_logger()

@celery.task(name="run_rct")
def run_rct(experiment_data: dict, design_table_dict: dict, experiment_id: str):
    logger.info("Into worker")
    design_table = pd.DataFrame.from_dict(design_table_dict)

    rct = RCT()
    rct_design_table = rct.random_control_trial(
        experiment_data,
        design_table
    )

    rct_design_table["experiment_id"] = experiment_id
    rct_table = rct_design_table.to_csv(index=False).encode("utf-8")

    return {"experiment_id": experiment_id,
            "rct_table": rct_table
            }


@celery.task(name="write_to_db")
def write_to_db(design_table_dict: dict, table_name: str) -> bool:
    """
    Method used to save RCT results

    Parameters
    ----------
    design_table_dict : dict
        RCT output table
    table_name : str
        Name of output table name

    Returns
    ----------
    bool
        True if insertion was succesful, False if not
    """
    logger.info("Starting RCT results insertion")
    design_table = pd.DataFrame.from_dict(design_table_dict)

    # Write spark DataFrame to Delta Lake using JDBC
    try:
       table_name = "testops.{}".format(table_name)

       logger.info("Initializing connection with database")
       # Connect to Databricks
       connection = sql.connect(
           server_hostname= settings.ADB_SERVER,
           http_path= settings.ADB_WAREHOUSE,
           access_token= settings.ADB_ACCESS_TOKEN
       )

       # Get a cursor
       cursor = connection.cursor()
       logger.info("Database connection estabished")
       delete_query = f"DROP TABLE IF EXISTS {table_name};"
       logger.info(f"{delete_query = }")
       cursor.execute(delete_query) 
       logger.info("table deleted")

       logger.info("Creating table: {}".format(table_name) )
       cursor.execute("CREATE TABLE IF NOT EXISTS {} (id DOUBLE, experiment_id INTEGER,ex_unit_id STRING,\
       ex_factor_1 STRING, ex_factor_2 STRING, ex_factor_3 STRING, ex_factor_4 STRING, ex_factor_5 STRING,\
       block_factor_1 STRING, block_factor_2 STRING, block_factor_3 STRING, block_factor_4 STRING, block_factor_5 STRING)".format(table_name) )
       logger.info("Table created")

       _id = [ i for i in range(len(design_table.index) )]
       experiment_id = design_table['experiment_id'].tolist()
       ex_unit_id = design_table['ex_unit_id'].tolist()
       ex_factor_1 = design_table['ex_factor_1'].tolist()
       ex_factor_2 = design_table['ex_factor_2'].tolist()
       ex_factor_3 = design_table['ex_factor_3'].tolist()
       ex_factor_4 = design_table['ex_factor_4'].tolist()
       ex_factor_5 = design_table['ex_factor_5'].tolist()
       block_factor_1 = design_table['block_factor_1'].tolist()
       block_factor_2 = design_table['block_factor_2'].tolist()
       block_factor_3 = design_table['block_factor_3'].tolist()
       block_factor_4 = design_table['block_factor_4'].tolist()
       block_factor_5 = design_table['block_factor_5'].tolist()


       values_list = [(a,b, str(c),str(d),str(e),str(f),str(g),str(h),str(i),str(j),str(k),str(l),str(m)) \
       for (a,b,c,d,e,f,g,h,i,j,k,l,m) in zip(_id, experiment_id, ex_unit_id,\
       ex_factor_1, ex_factor_2, ex_factor_3, ex_factor_4, ex_factor_5,\
       block_factor_1, block_factor_2, block_factor_3, block_factor_4, block_factor_5)]

       values_str = ",".join([f"({a},{b},\"{c}\",\"{d}\",\"{e}\",\"{f}\",\"{g}\",\"{h}\",\"{i}\",\"{j}\",\"{k}\",\"{l}\",\"{m}\" )"
       for (a,b,c,d,e,f,g,h,i,j,k,l,m ) in values_list ])

       logger.info("Starting data insertion in table: {}".format(table_name))

       cursor.execute(f"INSERT INTO {table_name} VALUES {values_str}")

       logger.info("Data insertion completed")
       return True

    except Exception as e:
        logger.error(f"Error inserting dataframe in deltalake: {e}")
        return False
    finally:
        # Closing open conections
        cursor.close()
        connection.close()


@dataclass
class RCTWorker(Task):
    """Class to execute RCT algorithm"""
    #logger: logging.Logger
    experiment_data: dict
    design_table: pd.DataFrame
    experiment_id: str
    #celery_engine: Celery = Celery()
    
    #@celery_engine.task(name="create_analysis_task")
    #@celery.task(name="create_analysis_task")
    #def compute(self, exp_obj: dict,  design_table: pd.DataFrame):
    #@celery.task(name="run_rct")
    def run(self):
        #self.logger.info("Into worker")
        rtc = RCT()
        rct_design_table = rtc.random_control_trial(
            self.experiment_data,
            self.design_table,
        )

        rct_design_table["experiment_id"] = self.experiment_id
        #rct_design_table["experiment_id"] = "00000000000"
        return rct_design_table

#RCTTask = celery.register_task(RCTWorker())

