"""
EOR Model definition, to be consumed by RCT API

Copyright 2023 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
 

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator, root_validator

class BLOCKING_FACTOR(BaseModel):
    id: str = Field(
        ...,
        title="id",
        description="Id",
    )
    selected: int = Field(
        ...,
        title="selected factor",
        description="Selected factor",
    )
    name: str = Field(
        ...,
        title="name",
        description="Name",
    )
    column: str = Field(
        ...,
        title="column name",
        description="Column name",
    )


class ExperimentObject(BaseModel):
    """Experiment Object Model for RCT API.

    Attributes:
        EXP_ID                  Experiment ID.
        BLOCKING_FACTORS        Blocking factors.
        EXPERIMENTAL_FACTORS    Experimental factors.
        SCOPE_SIZE              Scope size.
    """

    EXP_ID: Optional[int] = Field(
        None,
        title="experiment id",
        description="Unique ID to identify the experiment",
    )

    EXP_UNIT: Optional[str]
    EXP_MARKET: Optional[str]

    BLOCKING_FACTORS: Optional[List[BLOCKING_FACTOR]] = Field(
        [],
        title="blocking factors",
        description="Blocking factors",
    )

    EXPERIMENTAL_FACTORS: Optional[Dict] = Field(
        None,
        title="experimental factors",
        description="Experimental Factors",
    )

    SCOPE_SIZE: Optional[int] = Field(
        None, title="scope size", description="scope size"
    )

    PRIMARY_OUTCOME_NAME: Optional[str]


