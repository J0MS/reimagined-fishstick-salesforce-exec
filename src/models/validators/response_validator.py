"""
RCT API Response validator definitions.

Copyright 2023 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator, root_validator

class RCTOutputSchema(BaseModel):

    """
    Schema definition for RCT Output table
    """

    # id: Optional[int] = Field(..., ge=1, description="Record ID.")
    experiment_id: int = Field(
        ..., ge=1, description="Experiment ID, comming from EOR API."
    )
    ex_unit_id: Optional[str] = Field(
        ..., max_length=100, description="Experimental Unit ID."
    )
    ex_factor_1: Optional[str] = Field(
        ..., max_length=20, description="Experimental factor 1."
    )
    ex_factor_2: Optional[str] = Field(
        ..., max_length=20, description="Experimental factor 2."
    )
    ex_factor_3: Optional[str] = Field(
        ..., max_length=20, description="Experimental factor 3."
    )
    ex_factor_4: Optional[str] = Field(
        ..., max_length=20, description="Experimental factor 4."
    )
    ex_factor_5: Optional[str] = Field(
        ..., max_length=20, description="Experimental factor 5."
    )
    block_factor_1: Optional[str] = Field(
        ..., max_length=20, description="Blocking factor 1."
    )
    block_factor_2: Optional[str] = Field(
        ..., max_length=20, description="Blocking factor 2."
    )
    block_factor_3: Optional[str] = Field(
        ..., max_length=20, description="Blocking factor 3."
    )
    block_factor_4: Optional[str] = Field(
        ..., max_length=20, description="Blocking factor 4."
    )
    block_factor_5: Optional[str] = Field(
        ..., max_length=20, description="Blocking factor 5."
    )


class SchemaValidators:
    """
    Validators for consistency of RCT Output table
    """

    def unique_validator(self, field):
        def validator(cls, values):
            """
            Validator for check if experiment_id is unique
            """
            root_values = values.get("__root__")
            value_set = set()
            value_set.add(root_values[0].get(field))
            for value in root_values:
                if value[field] not in value_set:
                    raise ValueError(
                        "Duplicate {}, outlier:{}".format(field, value[field])
                    )
                else:
                    value_set.add(value[field])
            return values

        return root_validator(pre=True, allow_reuse=True)(validator)

    def check_ex_unit_id(self, field):
        def validator(cls, values):
            """
            Validator for ex_unit_id
            """
            root_values = values.get("__root__")
            # Check bussines requirements
            for value in root_values:
                if len(value[field]) < 1:
                    raise ValueError(
                        "Invalid {}, outlier:{}".format(field, value[field])
                    )
            return values

        return root_validator(pre=True, allow_reuse=True)(validator)

    def check_ex_factors(self, field):
        def validator(cls, values):
            """
            Validator for experimental factors
            """
            root_values = values.get("__root__")
            # Check bussines requirements
            for value in root_values:
                if len(value[field]) < 1:
                    raise ValueError(
                        "Invalid {}, outlier:{}".format(field, value[field])
                    )
            return values

        return root_validator(pre=True, allow_reuse=True)(validator)

    def check_block_factors(self, field):
        def validator(cls, values):
            """
            Validator for block factors
            """
            root_values = values.get("__root__")
            # Check bussines requirements
            for value in root_values:
                if len(value[field]) < 1:
                    raise ValueError(
                        "Invalid {}, outlier:{}".format(field, value[field])
                    )
            return values

        return root_validator(pre=True, allow_reuse=True)(validator)


class RCTResponseValidator(BaseModel):

    """
    Validator class using RCTOutputSchema
    """

    __root__: List[RCTOutputSchema]
    validate_unique_experiment_id = SchemaValidators().unique_validator("experiment_id")
    validate_ex_unit_id = SchemaValidators().check_ex_unit_id("ex_unit_id")
    validate_ex_factor_1 = SchemaValidators().check_ex_factors("ex_factor_1")
    validate_ex_factor_2 = SchemaValidators().check_ex_factors("ex_factor_2")
    validate_ex_factor_3 = SchemaValidators().check_ex_factors("ex_factor_3")
    validate_ex_factor_4 = SchemaValidators().check_ex_factors("ex_factor_4")
    validate_ex_factor_5 = SchemaValidators().check_ex_factors("ex_factor_5")
    validate_block_factor_1 = SchemaValidators().check_block_factors("block_factor_1")
    validate_block_factor_2 = SchemaValidators().check_block_factors("block_factor_2")
    validate_block_factor_3 = SchemaValidators().check_block_factors("block_factor_3")
    validate_block_factor_4 = SchemaValidators().check_block_factors("block_factor_4")
    validate_block_factor_5 = SchemaValidators().check_block_factors("block_factor_5")

