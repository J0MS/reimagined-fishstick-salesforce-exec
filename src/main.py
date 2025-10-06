"""
Salesforce API.

API routes definition.

Copyright 2025 Anheuser Busch InBev Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# FastAPI
import sys
from functools import partial
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi import FastAPI

# Monitoring
import logging

# Salesforce API
from .config.config import settings, APIMetadata, APIPolicies
from .config.api_lifecycle import  APILifecycle
from .config.middlewares import Middlewares, MiddlewareTools
from .config.logger.factory import LoggingFactory
from .routers.auth import VerifyAccess
from .routers.health.health import HealthChecker
from .routers.ml.routes import ComputeRouter
from contextlib import asynccontextmanager

logger: logging.Logger = LoggingFactory().get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events: startup and shutdown"""
    # Startup
    logger.info("🚀 Starting Lead Scoring API...")
    
    try:
        APILifecycle().load_model_from_registry()
        logger.info("✅ Model loaded on startup")
    except Exception as e:
        logger.error(f"❌ Failed to load model on startup: {e}")
        #sys.exit(1)
    
    yield
    # Shutdown
    logger.info("👋 Shutting down Lead Scoring API...")



# Fastapi rct_api initialization
api = FastAPI(
    title=APIMetadata.api_title,
    description=APIMetadata.api_description,
    version=APIMetadata.api_version,
    terms_of_service=APIMetadata.terms_of_service,
    contact=APIMetadata().contact,
    license_info=APIMetadata().license_info,
    openapi_tags=APIMetadata().tags_metadata,
    openapi_url=APIMetadata.openapi_url,
    lifespan=lifespan
)

# Set API middlewares
#api.add_middleware(TrustedHostMiddleware, allowed_hosts=APIPolicies().api_allowed_host)
##api.middleware("http")(partial(Middlewares.middleware_set_traceparent, parent_tracer=None, logger=logger))
#api.middleware("http")(partial(Middlewares.middleware_check_request, parent_tracer=None, logger=logger))

# Routes configuration
health = HealthChecker(logger=logger,  state="SUCCESS", response_string="ok")
api.include_router(health.router)

compute_router = ComputeRouter( logger=logger)
api.include_router(compute_router.router)



