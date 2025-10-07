"""
API configuration via environment variables.

Copyright 2023 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ..ml.lead_scoring_model import LeadScoringOutput
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator, root_validator

class LeadScoringResponse(BaseModel):
    """
    HTTP Response definition for /compute
    """

    STATUS_CODE: int
    STATE: str
    EXECUTION_ID: str
    INFERENCE_REPORT: Optional[LeadScoringOutput] = Field(
        None,
        title="lead scoring report",
        description="Salesforce model leads scoring report",
    )

    
    

    class Config:
        arbitrary_types_allowed = True


class WriteToDBResponse(BaseModel):
    """
    HTTP Response definition for /write-to-db
    """

    statusCode: int
    state: str
    exp_id: int

    class Config:
        arbitrary_types_allowed = True
