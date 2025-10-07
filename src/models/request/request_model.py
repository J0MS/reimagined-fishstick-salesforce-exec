"""
Leas scoring HTTP request model definition

Copyright 2025 Salesforce Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
 
from ..ml.lead_scoring_model import LeadScoringInput
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator, root_validator


class LeadScoringRequest(BaseModel):
    """Experiment Object Model for RCT API.

    Attributes:
        LEAD_ID            Lead ID.
        MARKET             Market/Country for this lead.
        LEAD_PARAMETERS    Lead parameters, used for this inference.
    """

    LEAD_ID: Optional[int] = Field(
        None,
        title="lead_id",
        description="Lead ID to identify the potential lead",
    )

    MARKET: Optional[str]

    INFERENCE_PARAMETERS: Optional[LeadScoringInput] = Field(
        None,
        title="lead scoring parameters",
        description="A set of parameters used by salesforce model to calculate leads scoring",
    )




