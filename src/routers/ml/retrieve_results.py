"""
Retrieve RCT Results

Copyright 2024 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from dataclasses import dataclass
import logging
from .worker import celery
from ...config.logger.factory import LoggingFactory

logger: logging.Logger = LoggingFactory().get_logger()

@dataclass
class RetrieveResults:
    """
    Retrieve job results

    Attributes
    ----------
    logger : logging.Logger
        Logger objects
    """

    def retrieve(
        job_id: str
    ):

        """
        Retrieve RCT output table

        Parameters
        ----------
        job_id : str
            job_id to get the results


        Returns
        -------
        dict
            A dictionary with RCT result

        """
        logger.info("Retriving RCT results")
        try:
            task_result = celery.AsyncResult(job_id)
            result = {
                "task_id": job_id,
                "task_status": task_result.status,
                "task_result": task_result.result
                    }
            return result
        except Exception as e:
            logger.error('Error retrieving RCT results' )

            result = {
                "task_id": job_id,
                "task_status": task_result.status,
                "task_result": str(e)
                    }
            return result
