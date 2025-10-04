"""
Random control trial API.
API middlewares configuration

Copyright 2025 Salesforce Inc.
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from fastapi import FastAPI, HTTPException, Response, status, Query, Path, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Any, Callable, Coroutine

# Config modules
from .config import RegisteredTeams, LoggingFormatter, APIMetadata, APIPolicies

# Monitoring
import logging


class MiddlewareTools:
    """
    Class to define middleware tools

    Attributes
    ----------
    None

    Methods
    -------
    callback_add_role_name(envelope=None)
        Add cloud role tag to request
    dispatch_header_exception(item: str)
        Return the proper exception type, based on missed parameter on header request
    """
    @staticmethod
    def callback_add_role_name(envelope):
        """ Callback middleware for add role name
        Parameters
        ----------
        envelope : Any
            Incoming request
        """
        envelope.tags["ai.cloud.role"] = APIMetadata.api_cloud_role_name


    @staticmethod
    def dispatch_header_exception(item: str) -> JSONResponse:
        """ Handle headers exception
        Parameters
        ----------
        item : str
            Captured exception string.

        Return
        ----------
        JSONResponse : JSONResponse
            Response with status code an error message.
        """

        responses = {
          "absent_abi_user": "salesforce_user missing in headers",
          "absent_abi_team": "salesforce_team missing in headers",
          "invalid_abi_team": "salesforce_team not registered"
        }

        error_message = responses.get(item, "No handler available")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={'message': error_message})


class Middlewares:
    """
    Class to create middleware definitions

    Attributes
    ----------
    None

    Methods
    -------
    middleware_check_request(request, call_next, logger, parent_tracer)
        Check request to determine in any parameter are missing or request is malformed
    middleware_set_traceparent(request, call_next, logger, parent_tracer)
        Set traceparent for each request
    """
    @staticmethod
    async def middleware_check_request(request: Request,
                                    call_next: Callable,
                                    logger: LoggingFormatter,
                                    parent_tracer: Any =None):
        """ 
        Check request to determine in any parameter are missing or request is malformed

        Parameters
        ----------
        request : Request
            HTTP request object
        call_next : Callable
            Funtion to be excuted after middleware execution
        logger: LoggingFormatter
            Logger object
        parent_tracer: Any
            Monitoring object

        Return
        ----------
        response : starlette.responses.Response
            Response with processed request
        """
        # Instantiate console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        # Instantiate custom LoggingFormatter object
        console_handler.setFormatter(LoggingFormatter())

        try:
            # Capture headers components
            headers = request.headers
            registred_teams = [e.value for e in RegisteredTeams]
            # Define headers scope
            abi_user_absent = "abi_user" not in headers
            abi_team_absent = "abi_team" not in headers
            abi_team_invalid = headers["abi_team"] not in registred_teams if not abi_team_absent else True
            abi_user = "absent_abi_user" if abi_user_absent else headers["abi_user"]
            abi_team = "absent_abi_team" if abi_team_absent else headers["abi_team"]
            abi_team_validation = "invalid_abi_team" if abi_team_invalid else headers["abi_team"]

            # random-control-trial route does not need the middleware
            if request.url.path in APIPolicies().api_middlewares_exclusions:
                response = await call_next(request)
                return response

            #Set headers exceptions
            exceptions = [abi_user, abi_team, abi_team_validation]
            # Build current exceptions array
            captured_exceptions = []
            for key in exceptions:
                if 'absent' in key or 'invalid' in key:
                    captured_exceptions.append(key)
            # For captured exceptions, return the proper handler
            if any([abi_user_absent, abi_team_absent, abi_team_invalid]):
                for exception in captured_exceptions:
                    logger.exception("Exception in headers: {}".format(exception))
                    return MiddlewareTools.dispatch_header_exception(exception)

            response = await call_next(request)
            return response
        except Exception as exception:
            #tracer.end_span()
            properties = {'custom_dimensions': {'exception': str(exception)}}
            logger.exception('Inherited exception, caught in the middleware_opencensus:{}'.format(str(exception)),
                             extra=properties)
            raise exception


    @staticmethod
    async def middleware_set_traceparent(request: Request,
                                    call_next: Callable,
                                    logger: LoggingFormatter,
                                    parent_tracer: Any =None):
        """ Configure set operation_parent_id  middleware for Azure exporter

        Parameters
        ----------
        request : Request
            HTTP request object
        call_next : Callable
            Funtion to be excuted after middleware execution
        logger: LoggingFormatter
            Logger object
        parent_tracer: Any
            Monitoring object

        Return
        ----------
        response : starlette.responses.Response
            Response with processed request

        """
        # Instantiate console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        # Instantiate custom LoggingFormatter object
        console_handler.setFormatter(LoggingFormatter())

        try:
            response = await call_next(request)
            traceparent = "{}-{}-{}-{}".format(APIPolicies.trace_version,
                                               APIPolicies.trace_id,
                                               APIPolicies.parent_span_id,
                                               APIPolicies.trace_flags)
            response.headers["traceparent"] = traceparent
            response.headers["parent_span_id"] = APIPolicies.parent_span_id
            return response
        except Exception as exception:
            #tracer.end_span()
            properties = {'custom_dimensions': {'exception': str(exception)}}
            logger.exception('Inherited exception, caught in the middleware_set_traceparent:{}'.format(str(exception)),
                             extra=properties)
            raise exception


