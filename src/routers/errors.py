"""Random Control Trial API Exceptions definition."""
from enum import Enum

class Exceptions(Enum):
    FAILED_EOR_API = 'Failed to invoke experiment object endpoint'
    FAILED_RCT_API = 'Failed to invoke RCT endpoint'
    FAILED_RCT_ENGINE = 'Failed to complete random control trial algorithm'
    FAILED_OLT_ENGINE = 'Failed to complete Outlier treatment algorithm'
    FAILED_DATABASE_CONNECTION = 'Failed database connection'
    FAILED_INSERTION = 'Failed insert in database'
    FAILED_ACCESS = 'Access attempt forbidden, missed or invalid token'
    FAILED_DECODE = 'Failed token access decode'
    FAILED_TOKEN_VERIFICATION = 'Token access verfication failed '
    INVALID_CREDENTIALS = 'Missed or invalid tokens'
    INVALID_AUTH_SCHEME = 'Invalid auth scheme'
    BROKEN_PIPE = 'Invalid input'
    INVALID_FILE = 'Invalid file or file not found exception'
