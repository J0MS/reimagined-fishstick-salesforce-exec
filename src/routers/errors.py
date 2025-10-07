"""
Lead scoring API errors definition.

Copyright 2025 Salesforce Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from enum import Enum

class Exceptions(Enum):
    FAILED_DATABASE_CONNECTION = 'Failed database connection'
    FAILED_INSERTION = 'Failed insert in database'
    FAILED_ACCESS = 'Access attempt forbidden, missed or invalid token'
    FAILED_DECODE = 'Failed token access decode'
    FAILED_INFERENCE = 'Imposible return inference, bad response object'
    FAILED_TOKEN_VERIFICATION = 'Token access verfication failed '
    INVALID_CREDENTIALS = 'Missed or invalid tokens'
    INVALID_AUTH_SCHEME = 'Invalid auth scheme'
    BROKEN_PIPE = 'Broken pipe, invalid return value'
    INVALID_FILE = 'Invalid file or file not found exception'
