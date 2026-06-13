"""
Model Router for Legalassist-AI.
Handles dynamic routing of LLM calls, automatic fallback, and metrics logging.
"""

import time
import logging
import datetime as dt
from typing import List, Dict, Any, Tuple, Optional
import openai
from openai import OpenAI

from config import Config
from db.session import SessionLocal
from db.models.analytics import ModelRoutingRule, ModelPerformance

logger = logging.getLogger(__name__)

class ModelRouter:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelRouter, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.primary_client = None
        self.secondary_client = None
        self._initialized = True
        self._init_clients()

    def _init_clients(self):
        """Initialize OpenAI API clients for primary and secondary providers."""
        try:
            if Config.OPENROUTER_API_KEY:
                self.primary_client = OpenAI(
                    api_key=Config.OPENROUTER_API_KEY,
                    base_url=Config.OPENROUTER_BASE_URL
                )
            else:
                logger.warning("Primary OPENROUTER_API_KEY is not configured.")
        except Exception as e:
            logger.error(f"Failed to initialize primary client: {e}")

        try:
            # Fallback to primary keys/urls if secondary keys are not set
            fallback_key = Config.SECONDARY_API_KEY or Config.OPENROUTER_API_KEY
            fallback_url = Config.SECONDARY_BASE_URL or Config.OPENROUTER_BASE_URL
            
            if fallback_key:
                self.secondary_client = OpenAI(
                    api_key=fallback_key,
                    base_url=fallback_url
                )
            else:
                logger.warning("Secondary client API Key is not configured.")
        except Exception as e:
            logger.error(f"Failed to initialize secondary client: {e}")

    def route_task(
        self,
        task: str,
        case_type: Optional[str] = None,
        jurisdiction: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Determine the preferred model and provider based on ModelRoutingRule.
        Returns a tuple of (model_name, provider_type) where provider_type is 'primary' or 'secondary'.
        """
        db = SessionLocal()
        try:
            # Query rules matching the task and optional metadata
            query = db.query(ModelRoutingRule).filter(
                ModelRoutingRule.task == task,
                ModelRoutingRule.approved == True
            )
            
            if case_type:
                query = query.filter(
                    (ModelRoutingRule.case_type == case_type) | (ModelRoutingRule.case_type == None)
                )
            if jurisdiction:
                query = query.filter(
                    (ModelRoutingRule.jurisdiction == jurisdiction) | (ModelRoutingRule.jurisdiction == None)
                )

            # Order by specificity (rules with exact case_type / jurisdiction match first)
            rules = query.all()
            if rules:
                # Simple sorting key: prioritize rules that have specific fields set over None
                def specificity_key(rule):
                    score = 0
                    if rule.case_type is not None:
                        score += 2
                    if rule.jurisdiction is not None:
                        score += 1
                    return score

                best_rule = sorted(rules, key=specificity_key, reverse=True)[0]
                logger.info(f"Routed task '{task}' to model '{best_rule.preferred_model}' via rule '{best_rule.name}'")
                return best_rule.preferred_model, "primary"

        except Exception as e:
            logger.error(f"Error checking model routing rules: {e}")
        finally:
            db.close()

        # Fallback to default configuration
        return Config.DEFAULT_MODEL, "primary"

    def log_performance(
        self,
        model_name: str,
        task: str,
        latency_ms: int,
        tokens_used: int = 0,
        case_type: Optional[str] = None,
        jurisdiction: Optional[str] = None
    ):
        """Log LLM performance metrics to the database."""
        db = SessionLocal()
        try:
            # Check if performance record already exists
            perf = db.query(ModelPerformance).filter(
                ModelPerformance.model_name == model_name,
                ModelPerformance.task == task,
                ModelPerformance.case_type == case_type,
                ModelPerformance.jurisdiction == jurisdiction
            ).first()

            now = dt.datetime.now(dt.timezone.utc)
            if perf:
                perf.samples += 1
                if perf.average_latency_ms is not None:
                    perf.average_latency_ms = int(
                        ((perf.average_latency_ms * (perf.samples - 1)) + latency_ms) / perf.samples
                    )
                else:
                    perf.average_latency_ms = latency_ms
                perf.last_updated = now
            else:
                perf = ModelPerformance(
                    model_name=model_name,
                    task=task,
                    case_type=case_type,
                    jurisdiction=jurisdiction,
                    samples=1,
                    average_latency_ms=latency_ms,
                    last_updated=now
                )
                db.add(perf)

            db.commit()
        except Exception as e:
            logger.error(f"Failed to log model performance: {e}")
            db.rollback()
        finally:
            db.close()

    def execute_call(
        self,
        task: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        timeout: float,
        case_type: Optional[str] = None,
        jurisdiction: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Execute the LLM call using the primary model and route.
        If it fails, automatically fall back to the secondary model/provider.
        """
        model, provider = self.route_task(task, case_type, jurisdiction)
        
        # Ensure clients are initialized
        if not self.primary_client or not self.secondary_client:
            self._init_clients()

        # Determine primary client to use
        client = self.primary_client
        if not client:
            logger.warning("Primary client not initialized. Falling back directly to secondary client.")
            client = self.secondary_client
            model = Config.SECONDARY_MODEL
            provider = "secondary"

        if not client:
            return None, "No LLM API clients configured. Please verify your keys in Settings."

        start_time = time.time()
        try:
            logger.info(f"Attempting LLM call using {provider} provider with model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            latency_ms = int((time.time() - start_time) * 1000)
            content = response.choices[0].message.content.strip()
            
            # Log successful performance
            tokens_used = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens or 0
            self.log_performance(model, task, latency_ms, tokens_used, case_type, jurisdiction)
            
            return content, None

        except Exception as primary_error:
            logger.warning(f"Primary LLM call failed: {primary_error}. Attempting fallback to secondary provider.")
            
            # Fallback configuration
            fallback_client = self.secondary_client
            fallback_model = Config.SECONDARY_MODEL

            if not fallback_client:
                return None, f"Primary call failed: {primary_error}. Fallback client not configured."

            fallback_start_time = time.time()
            try:
                logger.info(f"Attempting fallback LLM call with model: {fallback_model}")
                response = fallback_client.chat.completions.create(
                    model=fallback_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
                latency_ms = int((time.time() - fallback_start_time) * 1000)
                content = response.choices[0].message.content.strip()
                
                # Log successful fallback performance
                tokens_used = 0
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = response.usage.total_tokens or 0
                self.log_performance(fallback_model, task, latency_ms, tokens_used, case_type, jurisdiction)
                
                return content, None

            except Exception as secondary_error:
                error_msg = (
                    f"Both primary and fallback LLM calls failed. "
                    f"Primary error: {primary_error}. Secondary error: {secondary_error}"
                )
                logger.error(error_msg)
                return None, error_msg

# Global model router instance
model_router = ModelRouter()
