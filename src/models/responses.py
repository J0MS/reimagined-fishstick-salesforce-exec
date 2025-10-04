"""
API configuration via environment variables.

Copyright 2023 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pydantic import BaseModel

class RCTGatewayResponse(BaseModel):
    """
    HTTP Response definition for /rct-gateway
    """

    statusCode: int
    state: str
    exp_id: int
    rct_table: bytes

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
