import jwt
import time
import logging
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .errors import Exceptions

class VerifyAccess(HTTPBearer):
    def __init__(self, jwt_secret: str, algorithm: str, logger: logging.Logger, auto_error: bool = True):
        self.jwt_secret = jwt_secret
        self.algorithm = algorithm
        self.logger = logger
        super(VerifyAccess, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super(VerifyAccess, self).__call__(request)
        if credentials:
            if not credentials.scheme == "Bearer":
                self.logger.error(Exceptions.INVALID_AUTH_SCHEME.value)
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=Exceptions.INVALID_AUTH_SCHEME.value)
            if not self.verify_token(credentials.credentials):
                self.logger.error(Exceptions.FAILED_ACCESS.value)
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=Exceptions.FAILED_ACCESS.value)
            return credentials.credentials
        else:
            self.logger.error(Exceptions.INVALID_CREDENTIALS.value)
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=Exceptions.INVALID_CREDENTIALS.value)

    def decode_token(self, token: str) -> dict:
        try:
            decoded_token = jwt.decode(token, self.jwt_secret, algorithms=[self.algorithm])
            return decoded_token if decoded_token["expires"] >= time.time() else None
        except Exception as e:
            self.logger.error(Exceptions.FAILED_DECODE.value)
            properties = {'custom_dimensions': {'exception': str(e)}}
            self.logger.exception('Captured an exception.', extra=properties)
            return {}


    def verify_token(self, token: str) -> bool:
        isTokenValid: bool = False
        try:
            payload = self.decode_token(token)
        except Exception as e:
            self.logger.error(Exceptions.FAILED_TOKEN_VERIFICATION.value)
            properties = {'custom_dimensions': {'exception': str(e)}}
            self.logger.exception('Captured an exception.', extra=properties)
            payload = None
        if payload:
            isTokenValid = True
            self.logger.info("Token access verfication succesful")
        return isTokenValid
