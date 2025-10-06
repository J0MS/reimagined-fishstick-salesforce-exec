"""
Lead Scoring runner definition.

Copyright 2025 Salesforce Inc.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    cohen_kappa_score,
    mean_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import joblib
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import json
from datetime import datetime



# ============================================================================
# EXAMPLE USAGE WITH MLFLOW
# ============================================================================

def generate_sample_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic lead data for demonstration"""
    np.random.seed(42)
    
    data = {
        'email_opens': np.random.poisson(5, n_samples),
        'email_clicks': np.random.poisson(2, n_samples),
        'page_views': np.random.poisson(10, n_samples),
        'content_downloads': np.random.poisson(1, n_samples),
        'demo_requested': np.random.binomial(1, 0.2, n_samples),
        'pricing_page_visits': np.random.poisson(1, n_samples),
        'case_study_views': np.random.poisson(1, n_samples),
        'days_since_last_activity': np.random.exponential(10, n_samples),
        'company_size': np.random.lognormal(4, 1.5, n_samples),
        'annual_revenue': np.random.lognormal(15, 2, n_samples),
        'is_decision_maker': np.random.binomial(1, 0.3, n_samples),
        'job_level_score': np.random.randint(1, 6, n_samples),
        'session_duration_avg': np.random.exponential(5, n_samples),
        'pages_per_session': np.random.poisson(3, n_samples),
        'return_visitor': np.random.binomial(1, 0.4, n_samples),
    }
    
    X = pd.DataFrame(data)
    
    score_base = (
        X['email_clicks'] * 0.3 +
        X['demo_requested'] * 3 +
        X['is_decision_maker'] * 2 +
        X['pricing_page_visits'] * 0.5 +
        np.log1p(X['company_size']) * 0.2 +
        X['content_downloads'] * 0.4
    )
    
    score_normalized = (score_base - score_base.min()) / (score_base.max() - score_base.min())
    y = np.clip(np.round(score_normalized * 4 + 1), 1, 5).astype(int)
    
    return X, pd.Series(y)


if __name__ == "__main__":
    print("=" * 80)
    print("LEAD SCORING MODEL WITH MLFLOW TRACKING")
    print("=" * 80)
    
    # Generate sample data
    print("\n📊 Generating sample lead data...")
    X, y = generate_sample_data(n_samples=1000)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Score distribution:\n{y.value_counts().sort_index()}")
    
    # Train model with MLflow tracking
    print("\n🚀 Training Lead Scoring Model with MLflow...")
    model = LeadScoringModel(experiment_name="lead-scoring")
    
    metrics = model.train(
        X, y, 
        approach="ordinal", 
        use_cv=True,
        run_name="ordinal_v1",
        tags={
            "team": "data-science",
            "environment": "development"
        }
    )
    
    # Register model
    print("\n📦 Registering model in MLflow Model Registry...")
    version = model.register_model(
        model_name="lead-scoring-xgboost",
        stage="Staging",
        description="XGBoost ordinal classifier for 1-5 lead scoring"
    )
    
    # Make predictions
    print("\n🔮 Making predictions on sample leads...")
    sample_leads = X.head(5)
    predictions = model.predict_with_confidence(sample_leads)
    print("\nPrediction Results:")
    print(predictions)
    
    # Compare runs
    print("\n📊 Comparing model runs...")
    LeadScoringModel.compare_runs(experiment_name="lead-scoring", top_n=3)
    
    # Example: Load model from registry
    print("\n📥 Loading model from registry...")
    # loaded_model = LeadScoringModel.load_model_from_registry(
    #     model_name="lead-scoring-xgboost",
    #     stage="Staging"
    # )
    
    print("\n" + "="*80)
    print("✅ COMPLETE! View MLflow UI with: mlflow ui")
    print("   Then navigate to: http://localhost:5000")
    print("="*80)
