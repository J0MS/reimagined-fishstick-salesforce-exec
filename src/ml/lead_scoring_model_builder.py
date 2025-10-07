"""
Lead Scoring Model definition.

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
#from ..config.config import settings

# Set MLflow tracking URI 
#mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)  # For local file-based tracking

class LeadScoringModel:
    """
    XGBoost-based Lead Scoring Model (1-5 scale)
    
    Approach: Ordinal Classification with complete experiment tracking
    - Tracks all experiments, parameters, metrics, and artifacts in MLflow
    - Supports model versioning and registry
    - Enables model comparison and deployment
    """
    
    def __init__(self, experiment_name: str = "lead-scoring"):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.processed_feature_names = None  # ← NUEVO: nombres después del procesamiento
        self.experiment_name = experiment_name
        self.run_id = None
        
        # Set or create MLflow experiment
        mlflow.set_experiment(experiment_name)
        
    def create_model(self, approach: str = "ordinal") -> xgb.XGBClassifier:
        """
        Create XGBoost model with optimal hyperparameters for lead scoring.
        
        Args:
            approach: 'ordinal' (recommended) or 'multiclass'
        
        Returns:
            Configured XGBoost classifier
        """
        model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=5,
                max_depth=4,
                min_child_weight=5,
                n_estimators=200,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                gamma=0.1,
                tree_method='hist',
                random_state=42,
                n_jobs=-1,
                eval_metric=['mlogloss', 'merror'],
                early_stopping_rounds=20
            )
        
        return model
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for lead scoring"""
        df = df.copy()
        
        # Engagement score
        if all(col in df.columns for col in ['email_opens', 'email_clicks', 'page_views']):
            df['engagement_score'] = (
                df['email_opens'] * 0.3 + 
                df['email_clicks'] * 0.5 + 
                df['page_views'] * 0.2
            )
        
        # Recency features
        if 'days_since_last_activity' in df.columns:
            df['activity_recency'] = 1 / (df['days_since_last_activity'] + 1)
        
        # Intent signals
        if 'demo_requested' in df.columns and 'pricing_page_visits' in df.columns:
            df['high_intent'] = (
                (df['demo_requested'] == 1) | 
                (df['pricing_page_visits'] > 2)
            ).astype(int)
        
        # Company fit score
        if all(col in df.columns for col in ['company_size', 'annual_revenue']):
            df['company_fit_score'] = (
                np.log1p(df['company_size']) * 0.5 +
                np.log1p(df['annual_revenue']) * 0.5
            )
        
        return df
    
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        approach: str = "ordinal",
        use_cv: bool = True,
        run_name: Optional[str] = None,
        tags: Optional[Dict] = None
    ) -> Dict:
        """
        Train the lead scoring model with MLflow tracking.
        
        Args:
            X: Feature dataframe
            y: Target variable (scores 1-5)
            approach: 'ordinal' or 'multiclass'
            use_cv: Whether to use cross-validation
            run_name: Custom name for this MLflow run
            tags: Additional tags for the run
        
        Returns:
            Dictionary with training metrics
        """
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name) as run:
            self.run_id = run.info.run_id
            
            # Log tags
            mlflow.set_tag("model_type", "XGBoost")
            mlflow.set_tag("approach", approach)
            mlflow.set_tag("task", "lead_scoring")
            mlflow.set_tag("score_range", "1-5")
            
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            # Store original feature names
            self.feature_names = X.columns.tolist()
            
            # Log dataset info
            mlflow.log_param("n_samples", len(X))
            mlflow.log_param("n_features", len(X.columns))
            mlflow.log_param("feature_names", json.dumps(self.feature_names))
            
            # Log class distribution
            class_dist = y.value_counts().sort_index().to_dict()
            mlflow.log_dict(class_dist, "class_distribution.json")
            
            # Prepare features
            X_processed = self.prepare_features(X)
            
            # Store processed feature names (CRÍTICO)
            self.processed_feature_names = X_processed.columns.tolist()
            
            # Scale numerical features
            X_scaled = self.scaler.fit_transform(X_processed)
            
            # Encode labels (1-5 to 0-4 for XGBoost)
            y_encoded = y - 1
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_encoded, 
                test_size=0.2, 
                stratify=y_encoded,
                random_state=42
            )
            
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            
            # ========== CROSS-VALIDATION (ANTES DEL ENTRENAMIENTO FINAL) ==========
            cv_mean = None
            cv_std = None
            
            if use_cv:
                print("Running cross-validation...")
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                # Crear modelo limpio para CV (sin early stopping)
                cv_model = self.create_model(approach)
                
                # Remover parámetros que causan conflictos en CV
                cv_params = cv_model.get_params()
                if 'callbacks' in cv_params:
                    cv_model.set_params(callbacks=None)
                if 'early_stopping_rounds' in cv_params:
                    cv_model.set_params(early_stopping_rounds=None)
                
                try:
                    cv_scores = cross_val_score(
                        cv_model, 
                        X_scaled, 
                        y_encoded, 
                        cv=cv, 
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=0
                    )
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    mlflow.log_metric("cv_accuracy_mean", cv_mean)
                    mlflow.log_metric("cv_accuracy_std", cv_std)
                    
                    for i, score in enumerate(cv_scores):
                        mlflow.log_metric(f"cv_fold_{i+1}_accuracy", score)
                    
                    print(f"✓ Cross-validation completed: {cv_mean:.4f} (+/- {cv_std:.4f})")
                    
                except Exception as e:
                    print(f"⚠️ Warning: Cross-validation failed: {str(e)}")
                    print("Continuing without CV metrics...")
            
            # ========== ENTRENAMIENTO FINAL CON EARLY STOPPING ==========
            print("Training final model...")
            
            # Create model
            self.model = self.create_model(approach)
            
            # Log all model hyperparameters
            model_params = self.model.get_params()
            for param, value in model_params.items():
                mlflow.log_param(f"model_{param}", value)
            
            # Train with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
            
            # 🔍 DIAGNÓSTICO DE FEATURES
            print(f"📊 Feature count check:")
            print(f"  - Original features: {len(self.feature_names)}")
            print(f"  - After processing: {len(self.processed_feature_names)}")
            print(f"  - Model importances: {len(self.model.feature_importances_)}")
            
            # Log best iteration
            if hasattr(self.model, 'best_iteration'):
                mlflow.log_metric("best_iteration", self.model.best_iteration)
            
            # ========== PREDICCIONES Y MÉTRICAS ==========
            # Predictions
            y_pred = self.model.predict(X_val)
            y_pred_proba = self.model.predict_proba(X_val)
            y_pred_scores = y_pred + 1
            y_val_scores = y_val + 1
            
            # Calculate and log metrics
            accuracy = accuracy_score(y_val, y_pred)
            mae = mean_absolute_error(y_val_scores, y_pred_scores)
            kappa = cohen_kappa_score(y_val, y_pred, weights='quadratic')
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("cohen_kappa_quadratic", kappa)
            
            # Per-class metrics
            class_report = classification_report(
                y_val_scores, y_pred_scores,
                target_names=['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5'],
                output_dict=True
            )
            
            # Log per-class F1 scores
            for score in range(1, 6):
                f1 = class_report[f'Score {score}']['f1-score']
                precision = class_report[f'Score {score}']['precision']
                recall = class_report[f'Score {score}']['recall']
                
                mlflow.log_metric(f"f1_score_{score}", f1)
                mlflow.log_metric(f"precision_score_{score}", precision)
                mlflow.log_metric(f"recall_score_{score}", recall)
            
            # Log classification report
            mlflow.log_dict(class_report, "classification_report.json")
            
            # ========== VISUALIZACIONES ==========
            # Create confusion matrix plot
            cm = confusion_matrix(y_val_scores, y_pred_scores)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=range(1, 6), yticklabels=range(1, 6))
            plt.title('Confusion Matrix - Lead Scoring')
            plt.ylabel('True Score')
            plt.xlabel('Predicted Score')
            plt.tight_layout()
            
            # Save and log confusion matrix
            cm_path = "confusion_matrix.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()
            
            # Feature importance plot
            try:
                importance_df = self.get_feature_importance(top_n=15)
                plt.figure(figsize=(10, 6))
                plt.barh(range(len(importance_df)), importance_df['importance'])
                plt.yticks(range(len(importance_df)), importance_df['feature'])
                plt.xlabel('Importance')
                plt.title('Top 15 Most Important Features')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                importance_path = "feature_importance.png"
                plt.savefig(importance_path)
                mlflow.log_artifact(importance_path)
                plt.close()
                
                # Log feature importance as dict
                importance_dict = importance_df.set_index('feature')['importance'].to_dict()
                mlflow.log_dict(importance_dict, "feature_importance.json")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not generate feature importance plot: {str(e)}")
            
            # ========== GUARDAR MODELO ==========
            # Create model signature
            signature = infer_signature(X_train, y_pred_scores[:len(X_train)])
            
            # Log the model with signature
            mlflow.xgboost.log_model(
                self.model,
                "model",
                signature=signature,
                input_example=X_train[:5],
                registered_model_name="lead-scoring-xgboost"
            )
            
            # Log the scaler
            mlflow.sklearn.log_model(
                self.scaler,
                "scaler"
            )
            
            # Save and log preprocessing artifacts
            preprocessing_info = {
                'feature_names': self.feature_names,
                'processed_feature_names': self.processed_feature_names,  # ← NUEVO
                'scaler_mean': self.scaler.mean_.tolist(),
                'scaler_scale': self.scaler.scale_.tolist(),
                'approach': approach,
                'trained_at': datetime.now().isoformat()
            }
            mlflow.log_dict(preprocessing_info, "preprocessing_info.json")
            
            # ========== COMPILAR RESULTADOS ==========
            # Compile metrics dictionary
            metrics = {
                'accuracy': accuracy,
                'mae': mae,
                'cohen_kappa': kappa,
                'classification_report': classification_report(
                    y_val_scores, y_pred_scores,
                    target_names=['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5']
                ),
                'run_id': self.run_id
            }
            
            if use_cv and cv_mean is not None:
                metrics['cv_accuracy_mean'] = cv_mean
                metrics['cv_accuracy_std'] = cv_std
            
            # ========== IMPRIMIR RESULTADOS ==========
            print("=" * 60)
            print("LEAD SCORING MODEL TRAINING RESULTS")
            print("=" * 60)
            print(f"MLflow Run ID: {self.run_id}")
            print(f"Approach: {approach.upper()}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"MAE (Mean Absolute Error): {metrics['mae']:.4f}")
            print(f"Cohen's Kappa (Quadratic): {metrics['cohen_kappa']:.4f}")
            if use_cv and cv_mean is not None:
                print(f"CV Accuracy: {metrics['cv_accuracy_mean']:.4f} (+/- {metrics['cv_accuracy_std']:.4f})")
            print("\nClassification Report:")
            print(metrics['classification_report'])
            print("=" * 60)
            print(f"\nView results: mlflow ui")
            print(f"Experiment: {self.experiment_name}")
            print("=" * 60)
            
            return metrics
    
    def predict_score(self, X: pd.DataFrame, return_probabilities: bool = False):
        """Predict lead scores for new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_processed = self.prepare_features(X)
        X_scaled = self.scaler.transform(X_processed)
        
        predictions = self.model.predict(X_scaled)
        scores = predictions + 1
        
        if return_probabilities:
            probabilities = self.model.predict_proba(X_scaled)
            return scores, probabilities
        
        return scores
    
    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict scores with confidence levels and probability distribution"""
        scores, probabilities = self.predict_score(X, return_probabilities=True)
        
        results = pd.DataFrame({
            'predicted_score': scores,
            'confidence': probabilities.max(axis=1)
        })
        
        for i in range(5):
            results[f'prob_score_{i+1}'] = probabilities[:, i]
        
        return results
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Obtener importancias del modelo
        importances = self.model.feature_importances_
        
        # Usar processed_feature_names si está disponible, si no usar feature_names
        if self.processed_feature_names is not None:
            feature_names = self.processed_feature_names
        else:
            feature_names = self.feature_names
        
        # Verificar consistencia
        if len(feature_names) != len(importances):
            print(f"⚠️ Warning: Feature mismatch. Using generic names.")
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Crear DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Retornar top_n
        return importance_df.head(top_n)
    
    def register_model(
        self, 
        model_name: str = "lead-scoring-xgboost",
        stage: str = "Staging",
        description: Optional[str] = None
    ):
        """
        Register or update model in MLflow Model Registry.
        
        Args:
            model_name: Name for the registered model
            stage: Model stage (None, Staging, Production, Archived)
            description: Description of this model version
        """
        if self.run_id is None:
            raise ValueError("No run_id found. Train the model first.")
        
        client = MlflowClient()
        
        # Get the latest version of the model
        model_uri = f"runs:/{self.run_id}/model"
        
        try:
            # Register model
            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            version = result.version
            
            # Update version description
            if description:
                client.update_model_version(
                    name=model_name,
                    version=version,
                    description=description
                )
            
            # Transition to specified stage
            if stage:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage=stage
                )
            
            print(f"✅ Model registered: {model_name} (version {version})")
            print(f"   Stage: {stage}")
            print(f"   Run ID: {self.run_id}")
            
            return version
            
        except Exception as e:
            print(f"❌ Error registering model: {str(e)}")
            raise
    
    @staticmethod
    def load_model_from_registry(
        model_name: str = "lead-scoring-xgboost",
        stage: str = "Production"
    ) -> 'LeadScoringModel':
        """
        Load model from MLflow Model Registry.
        
        Args:
            model_name: Registered model name
            stage: Model stage to load (Production, Staging, etc.)
        
        Returns:
            LeadScoringModel instance with loaded model
        """
        #mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)  # For local file-based tracking
        model_uri = f"models:/{model_name}/{stage}"
        
        # Load the model
        loaded_model = mlflow.xgboost.load_model(model_uri)
        
        # Create instance and set model
        instance = LeadScoringModel()
        instance.model = loaded_model
        
        # Load preprocessing info
        client = MlflowClient()
        model_version = client.get_latest_versions(model_name, stages=[stage])[0]
        run_id = model_version.run_id
        
        # Download preprocessing artifacts
        artifacts_path = client.download_artifacts(run_id, "preprocessing_info.json")
        with open(artifacts_path, 'r') as f:
            preprocessing_info = json.load(f)
        
        instance.feature_names = preprocessing_info['feature_names']
        instance.processed_feature_names = preprocessing_info.get('processed_feature_names')
        
        # Reconstruct scaler
        instance.scaler = StandardScaler()
        instance.scaler.mean_ = np.array(preprocessing_info['scaler_mean'])
        instance.scaler.scale_ = np.array(preprocessing_info['scaler_scale'])
        
        print(f"✅ Model loaded from registry: {model_name} ({stage})")
        print(f"   Version: {model_version.version}")
        print(f"   Run ID: {run_id}")
        
        return instance
    
    @staticmethod
    def compare_runs(experiment_name: str = "lead-scoring", top_n: int = 5):
        """
        Compare top N runs from an experiment.
        
        Args:
            experiment_name: Name of the experiment
            top_n: Number of top runs to display
        """
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found.")
            return
        
        # Get all runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.accuracy DESC"],
            max_results=top_n
        )
        
        print(f"\n{'='*80}")
        print(f"TOP {top_n} RUNS - {experiment_name}")
        print(f"{'='*80}\n")
        
        for i, run in enumerate(runs, 1):
            print(f"Rank {i}:")
            print(f"  Run ID: {run.info.run_id}")
            print(f"  Accuracy: {run.data.metrics.get('accuracy', 'N/A'):.4f}")
            print(f"  MAE: {run.data.metrics.get('mae', 'N/A'):.4f}")
            print(f"  Cohen's Kappa: {run.data.metrics.get('cohen_kappa_quadratic', 'N/A'):.4f}")
            print(f"  Approach: {run.data.tags.get('approach', 'N/A')}")
            print(f"  Started: {run.info.start_time}")
            print()


