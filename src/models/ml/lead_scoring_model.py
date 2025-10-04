"""
Lead scorring API Model definition.

Copyright 2025 Salesforce Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime, date, timedelta
import operator
import pydantic
import json


class LeadScoringInput(BaseModel):
    """Input schema for lead scoring"""
    
    # Engagement metrics
    email_opens: int = Field(ge=0, description="Number of email opens")
    email_clicks: int = Field(ge=0, description="Number of email clicks")
    page_views: int = Field(ge=0, description="Number of page views")
    content_downloads: int = Field(ge=0, description="Number of content downloads")
    
    # Behavioral signals
    demo_requested: int = Field(ge=0, le=1, description="Whether demo was requested (0/1)")
    pricing_page_visits: int = Field(ge=0, description="Number of pricing page visits")
    case_study_views: int = Field(ge=0, description="Number of case study views")
    days_since_last_activity: float = Field(ge=0, description="Days since last activity")
    
    # Firmographic data
    company_size: float = Field(gt=0, description="Company size (employees)")
    annual_revenue: float = Field(gt=0, description="Annual revenue")
    
    # Demographic
    is_decision_maker: int = Field(ge=0, le=1, description="Is decision maker (0/1)")
    job_level_score: int = Field(ge=1, le=5, description="Job level score (1-5)")
    
    # Web behavior
    session_duration_avg: float = Field(ge=0, description="Average session duration (minutes)")
    pages_per_session: int = Field(ge=0, description="Pages per session")
    return_visitor: int = Field(ge=0, le=1, description="Is return visitor (0/1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email_opens": 10,
                "email_clicks": 5,
                "page_views": 25,
                "content_downloads": 3,
                "demo_requested": 1,
                "pricing_page_visits": 4,
                "case_study_views": 2,
                "days_since_last_activity": 2.5,
                "company_size": 500,
                "annual_revenue": 5000000,
                "is_decision_maker": 1,
                "job_level_score": 4,
                "session_duration_avg": 8.5,
                "pages_per_session": 5,
                "return_visitor": 1
            }
        }

class LeadScoringOutput(BaseModel):
    """Output schema for lead scoring prediction"""
    
    lead_score: int = Field(ge=1, le=5, description="Predicted lead score (1-5)")
    confidence: float = Field(ge=0, le=1, description="Prediction confidence")
    probabilities: dict = Field(description="Probability for each score (1-5)")
    
    model_info: dict = Field(description="Model metadata")
    prediction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "lead_score": 5,
                "confidence": 0.87,
                "probabilities": {
                    "score_1": 0.02,
                    "score_2": 0.03,
                    "score_3": 0.05,
                    "score_4": 0.03,
                    "score_5": 0.87
                },
                "model_info": {
                    "model_name": "lead-scoring-xgboost",
                    "version": "3",
                    "stage": "Production"
                },
                "prediction_timestamp": "2025-10-04T12:00:00Z"
            }
        }

