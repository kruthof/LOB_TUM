#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Look-Ahead Bias Lab – Streamlit App (Prototype)
Implements key modules from the project description:
  - 3.2 Bias-sensitive Prompt-Engine
  - 3.3 Ensemble-Inference Module
  - 3.4 Channel Identification (TF-IDF + optional LLM assist)
  - 3.5 Anti–Look-Ahead-Bias Tests (Masked-Text, Time-Shift, Sensitivity)
  - 3.6 Visualization Dashboard with export (PNG/PDF)
Design goals:
  * Privacy-respecting: can run fully offline (local models), no training on user data.
  * Reproducible: deterministic seeds and full raw-output persistence.
  * Extensible: provider shim for adding more LLMs.

Author: You (Garvin)
License: MIT (for illustrative prototype)
"""

from __future__ import annotations
import os
import io
import json
import time
import uuid
import math
import glob
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
import pandas as pd
from dateutil import parser as dtparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy import stats as scipy_stats
from itertools import combinations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Optional providers (guarded imports)
try:
    import openai  # openai>=1.0.0 style
except Exception:
    openai = None

try:
    import litellm  # Unified interface for multiple LLM providers
    litellm.suppress_debug_info = True  # Reduce verbose logging
except Exception:
    litellm = None

# ------------------------------
# Constants & lightweight utils
# ------------------------------
APP_VERSION = "0.9.0"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_DEFAULT_SEED = 7

# Basic financial channels to seed TF-IDF/LLM classification.
DEFAULT_CHANNELS = [
    "inflation", "employment", "growth", "monetary_policy",
    "fiscal_policy", "housing", "trade", "sentiment", "credit",
    "earnings", "productivity", "energy", "technology"
]

# Future-looking keywords for enhanced masking
FUTURE_KEYWORDS = [
    "outlook", "forecast", "guidance", "projection", "expect",
    "will", "going to", "next year", "next quarter", "upcoming",
    "future", "anticipated", "planned", "target", "goal",
    "forward", "ahead", "predict", "estimate", "likely",
    "intend", "aim", "vision", "strategy", "roadmap"
]

# ------------------------------
# Data structures
# ------------------------------

@dataclass
class ProviderConfig:
    """Configuration for a single model provider run."""
    name: str                  # Human label for display (e.g., "gpt-4o-mini", "claude-3-5-sonnet")
    provider: str              # "openai", "litellm", "mock"
    model: str                 # Model identifier for the provider
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    seed: int = RANDOM_DEFAULT_SEED
    extra: Dict[str, Any] = None  # Provider-specific kwargs

@dataclass
class PromptSpec:
    """Bias-sensitive prompt guardrails for text-driven inference."""
    task_instruction: str
    date_context: str
    channel_context: str
    content: str
    system_instructions: Optional[str] = None

    def render(self) -> Dict[str, Any]:
        """
        Render a provider-agnostic chat prompt.
        """
        sys_msg = self.system_instructions or (
            "You are a careful assistant for academic finance research. "
            "Never use information after the provided date context."
        )
        user_msg = (
            f"# Task\n{self.task_instruction}\n\n"
            f"# Date Context (do NOT use info after this date)\n{self.date_context}\n\n"
            f"# Channel Context (economic channels to consider)\n{self.channel_context}\n\n"
            f"# Document\n{self.content}\n"
        )
        return {
            "system": sys_msg,
            "user": user_msg,
        }

@dataclass
class InferenceRecord:
    """Raw record per (doc, provider) containing prompt & output."""
    run_id: str
    doc_id: str
    provider_name: str
    provider: str
    model: str
    prompt: Dict[str, Any]
    output_text: str
    meta: Dict[str, Any]

# ------------------------------
# Provider shims (pluggable)
# ------------------------------

class ModelProvider:
    """
    Base provider shim. Implement `infer()` in subclasses to return text.
    """
    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg

    def infer(self, prompt: Dict[str, Any]) -> str:
        raise NotImplementedError

class OpenAIProvider(ModelProvider):
    """
    OpenAI API client for chat.completions (new SDK style).
    Requires OPENAI_API_KEY in env. Handles seed (if supported by model),
    temperature, top_p, and max_tokens.
    """
    def __init__(self, cfg: ProviderConfig):
        super().__init__(cfg)
        if openai is None:
            raise RuntimeError("openai package not installed. pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY")
        try:
            api_key = api_key or st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found (env or Streamlit secrets).")
        openai.api_key = api_key

        # Use client per new SDK
        self.client = openai.OpenAI()

    def infer(self, prompt: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        # Retry logic with exponential backoff for rate limiting
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=float(self.cfg.temperature),
                    top_p=float(self.cfg.top_p),
                    max_tokens=int(self.cfg.max_tokens),
                    messages=messages,
                    seed=int(self.cfg.seed),
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a rate limit error
                if "rate" in error_str or "429" in error_str or "too many" in error_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"Rate limited. Retrying in {delay:.1f}s... (attempt {attempt+1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        return f"[RATE_LIMIT_ERROR] {e}"
                else:
                    return f"[PROVIDER_ERROR] {type(e).__name__}: {e}"
        
        return "[PROVIDER_ERROR] Max retries exceeded"

class LiteLLMProvider(ModelProvider):
    """
    Unified provider using LiteLLM for multiple LLM backends.
    Supports: Claude (Anthropic), Gemini (Google), Cohere, and 100+ others.
    API keys expected in environment or Streamlit secrets:
    - ANTHROPIC_API_KEY for Claude
    - GOOGLE_API_KEY for Gemini
    - COHERE_API_KEY for Cohere
    - etc.
    """
    def __init__(self, cfg: ProviderConfig):
        super().__init__(cfg)
        if litellm is None:
            raise RuntimeError("litellm package not installed. pip install litellm")
        
        # Setup API keys from environment or Streamlit secrets
        self._setup_api_keys()
        
    def _setup_api_keys(self):
        """Load API keys from environment or Streamlit secrets"""
        # Map of common providers to their env var names
        key_names = {
            "anthropic": "ANTHROPIC_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "google": "GOOGLE_API_KEY",
            "cohere": "COHERE_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        
        # Detect provider from model name
        model_lower = self.cfg.model.lower()
        for provider, key_name in key_names.items():
            if provider in model_lower:
                # Try environment first
                api_key = os.environ.get(key_name)
                if not api_key:
                    # Try Streamlit secrets
                    try:
                        api_key = st.secrets.get(key_name)
                        if api_key:
                            os.environ[key_name] = api_key
                    except Exception:
                        pass
                break
    
    def infer(self, prompt: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        
        # Retry logic with exponential backoff
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Build completion parameters
                # Note: Not all providers support all parameters
                completion_params = {
                    "model": self.cfg.model,
                    "messages": messages,
                    "temperature": float(self.cfg.temperature),
                    "top_p": float(self.cfg.top_p),
                    "max_tokens": int(self.cfg.max_tokens),
                }
                
                # Only add seed for providers that support it (OpenAI, some others)
                # Anthropic (Claude) and Google (Gemini) don't support seed parameter
                model_lower = self.cfg.model.lower()
                if self.cfg.seed and not any(x in model_lower for x in ["claude", "gemini", "cohere"]):
                    completion_params["seed"] = int(self.cfg.seed)
                
                # LiteLLM unified completion call
                response = litellm.completion(**completion_params)
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a rate limit error
                if "rate" in error_str or "429" in error_str or "too many" in error_str or "quota" in error_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"Rate limited. Retrying in {delay:.1f}s... (attempt {attempt+1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        return f"[RATE_LIMIT_ERROR] Exceeded max retries: {e}"
                else:
                    # Not a rate limit error, return immediately
                    return f"[PROVIDER_ERROR] {type(e).__name__}: {e}"
        
        return "[PROVIDER_ERROR] Max retries exceeded"

class MockProvider(ModelProvider):
    """
    Offline stub for local testing; produces pseudo-deterministic outputs.
    This is useful for unit tests and privacy-only settings.
    """
    def infer(self, prompt: Dict[str, Any]) -> str:
        seed = self.cfg.seed
        np.random.seed(seed)
        # trivial "analysis" by counting tokens; produces stable pseudo-content
        user = prompt.get("user", "")
        token_count = len(user.split())
        score = np.random.rand() * (token_count % 13 + 1)
        # Pretend we return channel probabilities in text, plus a "signal"
        return (
            "Channels: inflation=0.22 employment=0.11 credit=0.05 sentiment=0.19 "
            f"growth=0.09 earnings=0.07 energy=0.06 monetary_policy=0.21\n"
            f"Signal: {score:.3f}\n"
            "Rationale: pseudo-deterministic mock output for testing."
        )

# Add more providers by subclassing ModelProvider or using LiteLLMProvider.

# ------------------------------
# Connection testing
# ------------------------------

def test_provider_connection(provider: ModelProvider) -> Tuple[bool, str]:
    """
    Test if a provider connection is working with a minimal request.
    Returns (success: bool, message: str)
    """
    try:
        test_prompt = {
            "system": "You are a helpful assistant.",
            "user": "Reply with just the word 'OK' and nothing else."
        }
        result = provider.infer(test_prompt)
        
        if "[PROVIDER_ERROR]" in result or "[RATE_LIMIT_ERROR]" in result:
            return False, f"Connection failed: {result}"
        
        return True, f"Connection successful! Response: {result[:100]}"
    except Exception as e:
        return False, f"Connection test failed with exception: {type(e).__name__}: {e}"

# ------------------------------
# Prompt engine (3.2)
# ------------------------------

def build_prompt(
    task_instruction: str,
    doc_text: str,
    as_of_date: str,
    channels: List[str],
    system_instructions: Optional[str] = None,
) -> PromptSpec:
    """
    Create a bias-sensitive prompt that explicitly forbids use of post-date info,
    and foregrounds economic channels (guardrails).
    """
    # Guardrail: we pin the as_of_date and embed channels explicitly.
    ch_ctx = ", ".join(channels) if channels else "N/A"
    ps = PromptSpec(
        task_instruction=task_instruction.strip(),
        date_context=as_of_date.strip(),
        channel_context=ch_ctx,
        content=doc_text.strip(),
        system_instructions=system_instructions,
    )
    return ps

# ------------------------------
# LLM-assisted channel extraction
# ------------------------------

def extract_channels_from_corpus(
    docs: List[str],
    num_channels: int,
    provider: ModelProvider,
    sample_size: int = 50
) -> List[str]:
    """
    Use an LLM to extract the most relevant economic/financial channels from a corpus.
    Samples documents to avoid token limits.
    """
    # Sample documents if corpus is large
    sample_docs = docs[:sample_size] if len(docs) > sample_size else docs
    corpus_sample = "\n\n---\n\n".join(sample_docs[:10])  # Use first 10 for context
    
    prompt_spec = PromptSpec(
        task_instruction=(
            f"Analyze the following economic/financial text corpus and identify the {num_channels} most important "
            f"economic or financial topics/channels present. Return ONLY a JSON array of channel names.\n\n"
            f"Examples of good channels: 'inflation', 'employment', 'monetary_policy', 'earnings', 'housing', 'trade'\n\n"
            f"Return format: [\"channel1\", \"channel2\", ...]\n\n"
            f"Corpus sample:\n{corpus_sample[:3000]}"
        ),
        date_context="N/A",
        channel_context="",
        content="",
        system_instructions="Return ONLY a valid JSON array of strings. No explanation."
    )
    
    raw = provider.infer(prompt_spec.render())
    
    # Try to parse JSON
    try:
        start = raw.find("[")
        end = raw.rfind("]")
        if start >= 0 and end >= start:
            channels = json.loads(raw[start:end+1])
            return [str(ch).strip().lower().replace(" ", "_") for ch in channels[:num_channels]]
    except Exception:
        pass
    
    # Fallback to default if parsing fails
    return DEFAULT_CHANNELS[:num_channels]

def generate_best_practice_prompt(
    user_prompt: str,
    channels: List[str],
    granular_mode: bool = True
) -> Tuple[str, str]:
    """
    Generate a best-practice prompt that helps avoid look-ahead bias.
    Returns (task_instruction, system_instructions)
    """
    if granular_mode and channels:
        # Channel-specific analysis to avoid broad predictions
        task_instruction = (
            f"Analyze the document's discussion of the following economic channels:\n"
            f"{', '.join(channels)}\n\n"
            f"For each relevant channel mentioned, provide:\n"
            f"1. What specific facts or data points are stated (not predictions)\n"
            f"2. A score [0-1] indicating the current state/sentiment for that channel\n\n"
            f"Focus ONLY on information explicitly stated in the text. "
            f"Do not extrapolate to future implications.\n\n"
            f"Format your response as:\n"
            f"Channel: [name]\n"
            f"Facts: [specific facts from text]\n"
            f"Score: [0-1]\n"
            f"Signal: [average score across all channels]"
        )
    else:
        # Keep user's prompt but add guardrails
        task_instruction = (
            f"{user_prompt}\n\n"
            f"IMPORTANT: Analyze ONLY the information explicitly present in the text. "
            f"Do not make forward-looking predictions or use information that would only be known after the document date."
        )
    
    system_instructions = (
        "You are an expert financial analyst conducting historical research. "
        "You must NEVER use information that would only be available after the document's timestamp. "
        "Treat each document as if you are living in that moment in time. "
        "If the text contains forward-looking statements (forecasts, guidance, predictions), "
        "note them as 'stated predictions' but do not validate them with hindsight knowledge. "
        "Focus on facts, data, and sentiment as expressed AT THAT TIME."
    )
    
    return task_instruction, system_instructions

# ------------------------------
# Corpus Summarization (NEW for v2)
# ------------------------------

def summarize_corpus(
    docs: List[str],
    provider: ModelProvider,
    max_length: int = 200,
    request_delay: float = 0.5
) -> List[str]:
    """
    Summarize documents focusing on stated facts, not predictions.
    Returns list of summaries (same length as docs).
    """
    summaries = []
    
    for i, doc in enumerate(docs):
        prompt = {
            "system": (
                "You are a precise summarizer for academic research. "
                "Focus ONLY on facts, data, and events explicitly stated in the text. "
                "Do NOT include dates."
            ),
            "user": (
                f"Summarize the following document in maximum {max_length} words. "
                f"Include only factual information, data points, and stated events. "
                f"Exclude all forward-looking language, including dates and entities. e.g. refer to company A acquired company B instead of mercedes acquired bmw.\n\n"
                f"Document:\n{doc[:3000]}"  # Limit input length
            )
        }
        
        try:
            summary = provider.infer(prompt)
            # Remove error markers if present
            if "[PROVIDER_ERROR]" in summary or "[RATE_LIMIT_ERROR]" in summary:
                summary = doc[:max_length * 5]  # Fallback to truncated original
            summaries.append(summary)
        except Exception as e:
            print(f"Summarization failed for doc {i}: {e}")
            summaries.append(doc[:max_length * 5])  # Fallback
        
        # Delay to avoid rate limits
        if provider.cfg.provider != "mock" and request_delay > 0:
            time.sleep(request_delay)
    
    return summaries

# ------------------------------
# Statistical Comparison Functions (NEW for v2)
# ------------------------------

def compare_versions_statistically(
    df_results: pd.DataFrame,
    version_col: str = "corpus_version",
    score_col: str = "sentiment",
    doc_col: str = "doc_id"
) -> Dict[str, Any]:
    """
    Statistical tests comparing sentiment across corpus versions.
    
    Args:
        df_results: DataFrame with columns [doc_id, corpus_version, sentiment, ...]
        version_col: Column name for version (original/masked/summary)
        score_col: Column name for sentiment scores
        doc_col: Column name for document IDs
    
    Returns:
        Dict with p-values, effect sizes, means, etc.
    """
    results = {}
    
    # Get data by version
    versions = df_results[version_col].unique()
    
    if len(versions) < 2:
        return {"error": "Need at least 2 versions to compare"}
    
    # Calculate means and std devs
    for version in versions:
        version_data = df_results[df_results[version_col] == version][score_col].dropna()
        results[f"{version}_mean"] = float(version_data.mean())
        results[f"{version}_std"] = float(version_data.std())
        results[f"{version}_n"] = len(version_data)
    
    # Pairwise comparisons
    from itertools import combinations
    
    for v1, v2 in combinations(versions, 2):
        data1 = df_results[df_results[version_col] == v1][score_col].dropna()
        data2 = df_results[df_results[version_col] == v2][score_col].dropna()
        
        if len(data1) > 1 and len(data2) > 1:
            # T-test
            t_stat, p_value = scipy_stats.ttest_ind(data1, data2)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(data1)-1)*data1.std()**2 + (len(data2)-1)*data2.std()**2) / (len(data1)+len(data2)-2))
            cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval for mean difference
            diff_mean = data1.mean() - data2.mean()
            diff_se = pooled_std * np.sqrt(1/len(data1) + 1/len(data2))
            ci_lower = diff_mean - 1.96 * diff_se
            ci_upper = diff_mean + 1.96 * diff_se
            
            results[f"{v1}_vs_{v2}"] = {
                "p_value": float(p_value),
                "t_statistic": float(t_stat),
                "cohens_d": float(cohens_d),
                "mean_difference": float(diff_mean),
                "ci_95_lower": float(ci_lower),
                "ci_95_upper": float(ci_upper),
                "significant": p_value < 0.05
            }
    
    # ANOVA if 3+ groups
    if len(versions) >= 3:
        groups = [df_results[df_results[version_col] == v][score_col].dropna() for v in versions]
        f_stat, p_anova = scipy_stats.f_oneway(*groups)
        results["anova"] = {
            "f_statistic": float(f_stat),
            "p_value": float(p_anova),
            "significant": p_anova < 0.05
        }
    
    return results

def interpret_bias_detection(stats: Dict[str, Any]) -> str:
    """
    Interpret statistical results for bias detection.
    Returns human-readable interpretation.
    """
    interpretation = []
    
    # Check original vs masked
    if "original_vs_masked" in stats:
        comp = stats["original_vs_masked"]
        if comp["significant"] and comp["mean_difference"] > 0:
            interpretation.append(
                f"⚠️ **LOOK-AHEAD BIAS DETECTED**: Original text produces "
                f"significantly higher sentiment ({comp['mean_difference']:.3f} points, "
                f"p={comp['p_value']:.4f}) than masked text. "
                f"Effect size: {comp['cohens_d']:.2f}"
            )
        elif comp["significant"]:
            interpretation.append(
                f"ℹ️ Significant difference found (p={comp['p_value']:.4f}), "
                f"but masked version is higher. May indicate over-masking."
            )
        else:
            interpretation.append(
                f"✓ **No significant difference**: Original vs Masked "
                f"(p={comp['p_value']:.4f}, d={comp['cohens_d']:.2f})"
            )
    
    # Check original vs summary
    if "original_vs_summary" in stats:
        comp = stats["original_vs_summary"]
        if comp["significant"] and comp["mean_difference"] > 0:
            interpretation.append(
                f"⚠️ **BIAS in SUMMARIES**: Original produces significantly higher "
                f"sentiment than summaries ({comp['mean_difference']:.3f} points, "
                f"p={comp['p_value']:.4f})"
            )
        elif not comp["significant"]:
            interpretation.append(
                f"✓ Original vs Summary: No significant difference "
                f"(p={comp['p_value']:.4f})"
            )
    
    # Overall ANOVA
    if "anova" in stats and stats["anova"]["significant"]:
        interpretation.append(
            f"\n📊 **Overall ANOVA**: Significant differences across versions "
            f"(F={stats['anova']['f_statistic']:.2f}, p={stats['anova']['p_value']:.4f})"
        )
    
    return "\n\n".join(interpretation) if interpretation else "No comparison available"

# ------------------------------
# LLM-Based Masking (NEW for v2)
# ------------------------------

def llm_based_masking(
    doc_text: str,
    cutoff_date: str,
    provider: ModelProvider,
    categories_to_mask: List[str] = None
) -> str:
    """
    Use LLM to intelligently identify and mask selected categories.
    The LLM understands context better than keyword matching.
    
    Args:
        categories_to_mask: List of category strings to mask (e.g., ["Time (dates, years, quarters)", "Organizations"])
    """
    # Build masking instructions based on selected categories
    if not categories_to_mask:
        categories_to_mask = ["Time (dates, years, quarters)"]  # Default
    
    mask_instructions = []
    for category in categories_to_mask:
        if "Time" in category:
            mask_instructions.append(f"- Dates, years, quarters, and time periods after {cutoff_date}")
        if "Organizations" in category:
            mask_instructions.append("- Organization names, company names, institution names")
        if "Numbers" in category:
            mask_instructions.append("- Specific numbers, percentages, metrics, financial figures")
        if "Locations" in category:
            mask_instructions.append("- Geographic locations, cities, countries, regions")
        if "Person Names" in category:
            mask_instructions.append("- Names of individuals, executives, public figures")
        if "Gender" in category:
            mask_instructions.append("- Gender pronouns and gender-specific terms (he/she/him/her)")
        if "Product" in category:
            mask_instructions.append("- Product names, brand names, trademarked terms")
    
    instructions_text = "\n".join(mask_instructions)
    
    prompt = {
        "system": (
            "You are a precise text editor for academic research. "
            "Your task is to selectively mask specified elements while preserving other content intact."
        ),
        "user": (
            f"Document Date: {cutoff_date}\n\n"
            f"Task: Rewrite this document by replacing ONLY the following elements with [MASK]:\n\n"
            f"{instructions_text}\n\n"
            f"Keep all other information intact. Do not mask elements not listed above.\n"
            f"Replace only the specified parts, keep the rest unchanged.\n\n"
            f"Document:\n{doc_text}\n\n"
            f"Return ONLY the masked version, no explanations."
        )
    }
    
    try:
        masked = provider.infer(prompt)
        # Remove error markers if present
        if "[PROVIDER_ERROR]" in masked or "[RATE_LIMIT_ERROR]" in masked:
            return doc_text  # Fallback to original if masking fails
        return masked
    except Exception as e:
        print(f"LLM masking failed: {e}")
        return doc_text  # Fallback

def keyword_based_masking(
    doc_text: str, 
    cutoff_date: str, 
    keywords: List[str] = FUTURE_KEYWORDS,
    categories_to_mask: List[str] = None
) -> str:
    """
    Simple keyword-based masking (fallback option).
    Less intelligent than LLM-based but faster and cheaper.
    
    Args:
        categories_to_mask: List of category strings to mask
    """
    if not categories_to_mask:
        categories_to_mask = ["Time (dates, years, quarters)"]  # Default
    
    try:
        cutoff_year = dtparser.parse(cutoff_date).year
    except Exception:
        cutoff_year = 10_000
    
    masked = doc_text
    
    # Apply category-specific masking
    if any("Time" in cat for cat in categories_to_mask):
        # 1. Mask year tokens >= cutoff
        tokens = masked.split()
        new_tokens = []
        for w in tokens:
            w_clean = "".join(ch for ch in w if ch.isdigit())
            if w_clean.isdigit():
                try:
                    val = int(w_clean)
                    if val >= cutoff_year:
                        new_tokens.append("[MASK]")
                        continue
                except Exception:
                    pass
            new_tokens.append(w)
        masked = " ".join(new_tokens)
        
        # 2. Mask future-looking keywords
        for keyword in keywords:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            masked = pattern.sub("[MASK]", masked)
    
    # Note: Other categories (Organizations, Numbers, etc.) are best handled by LLM
    # Keyword-based masking focuses on Time category
    
    return masked

def enhanced_masked_text_guard(doc_text: str, cutoff_date: str, keywords: List[str] = FUTURE_KEYWORDS) -> str:
    """
    Alias for keyword_based_masking for backwards compatibility.
    """
    return keyword_based_masking(doc_text, cutoff_date, keywords)

# ------------------------------
# Channel identification (3.4)
# ------------------------------

def tfidf_channel_scores(
    docs: List[str],
    channels: List[str],
    ngram_range: Tuple[int, int] = (1, 2),
) -> pd.DataFrame:
    """
    Build TF-IDF across corpus and score channels by average TF-IDF of tokens
    matching channel keywords (exact token match for simplicity).
    Returns a dataframe [doc_idx, channel, score].
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=ngram_range,
        min_df=1
    )
    X = vectorizer.fit_transform(docs)
    vocab = vectorizer.vocabulary_
    results = []

    for di in range(X.shape[0]):
        for ch in channels:
            # exact match on token or bigram
            tok = ch.lower().replace(" ", "_")
            # try several forms
            candidates = set([ch.lower(), ch.lower().replace(" ", "_"), ch.lower().replace("_", " ")])
            idxs = [vocab.get(c) for c in candidates if vocab.get(c) is not None]
            if not idxs:
                score = 0.0
            else:
                score = float(X[di, idxs].sum())
            results.append({"doc_idx": di, "channel": ch, "score": score})

    df = pd.DataFrame(results)
    # Normalize per-document to [0,1]
    df["score_norm"] = df.groupby("doc_idx")["score"].transform(
        lambda s: MinMaxScaler().fit_transform(s.values.reshape(-1,1)).ravel() if s.max() > 0 else s)
    return df

# Optional: an LLM-assisted channel classifier (lightweight)
def llm_channel_tags(text: str, channels: List[str], provider: Optional[ModelProvider]) -> Dict[str, float]:
    """
    Ask the LLM to rate the relevance of provided channels on [0,1].
    If provider is None, return empty dict.
    """
    if provider is None:
        return {}
    spec = PromptSpec(
        task_instruction=(
            "Rate each given economic channel for relevance to the document on [0,1]. "
            "Return a JSON object {channel: score} with only the given channels as keys."
        ),
        date_context="As-of date: use only the document's internal timestamp if mentioned.",
        channel_context=", ".join(channels),
        content=text[:4000],  # safety truncation
        system_instructions="Return ONLY valid JSON. Do not explain.",
    )
    raw = provider.infer(spec.render())
    # attempt to parse JSON robustly
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end >= start:
            return json.loads(raw[start:end+1])
    except Exception:
        pass
    return {}

# ------------------------------
# Ensemble inference (3.3)
# ------------------------------

def run_ensemble(
    df: pd.DataFrame,
    providers: List[ProviderConfig],
    task_instruction: str,
    as_of_col: str = "timestamp",
    text_col: str = "text",
    channels: List[str] = DEFAULT_CHANNELS,
    system_instructions: Optional[str] = None,
    limit_docs: Optional[int] = None,
    request_delay: float = 0.5,
) -> List[InferenceRecord]:
    """
    Orchestrate N runs over M models; persist raw outputs.
    Returns a list of InferenceRecord.
    """
    # Provider instances
    instances: List[ModelProvider] = []
    for cfg in providers:
        if cfg.provider == "openai":
            instances.append(OpenAIProvider(cfg))
        elif cfg.provider == "litellm":
            instances.append(LiteLLMProvider(cfg))
        elif cfg.provider == "mock":
            instances.append(MockProvider(cfg))
        else:
            raise ValueError(f"Unsupported provider: {cfg.provider}")

    # Slice docs if requested
    df_ = df.copy()
    if limit_docs is not None:
        df_ = df_.head(int(limit_docs))

    records: List[InferenceRecord] = []
    run_id = time.strftime("run_%Y%m%d_%H%M%S")

    for i, row in df_.iterrows():
        doc_id = str(row.get("doc_id", i))
        text = str(row[text_col])
        as_of_date = str(row[as_of_col])
        ps = build_prompt(task_instruction, text, as_of_date, channels, system_instructions)

        for inst in instances:
            rendered = ps.render()
            out = inst.infer(rendered)
            rec = InferenceRecord(
                run_id=run_id,
                doc_id=doc_id,
                provider_name=inst.cfg.name,
                provider=inst.cfg.provider,
                model=inst.cfg.model,
                prompt=rendered,
                output_text=out,
                meta={"temperature": inst.cfg.temperature, "top_p": inst.cfg.top_p,
                      "seed": inst.cfg.seed, "max_tokens": inst.cfg.max_tokens}
            )
            records.append(rec)
            
            # Add delay between API calls to avoid rate limiting (skip for mock providers)
            if inst.cfg.provider != "mock" and request_delay > 0:
                time.sleep(request_delay)

    # Persist to disk (JSONL + Parquet)
    jsonl_path = os.path.join(OUTPUT_DIR, f"{run_id}_raw.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    pq_path = os.path.join(OUTPUT_DIR, f"{run_id}_raw.parquet")
    pd.DataFrame([asdict(r) for r in records]).to_parquet(pq_path, index=False)

    return records

# ------------------------------
# Batched Ensemble (Multiple docs per API call)
# ------------------------------

def run_ensemble_batched(
    df: pd.DataFrame,
    providers: List[ProviderConfig],
    task_instruction: str,
    as_of_col: str = "timestamp",
    text_col: str = "text",
    channels: List[str] = DEFAULT_CHANNELS,
    system_instructions: Optional[str] = None,
    limit_docs: Optional[int] = None,
    request_delay: float = 0.5,
    batch_size: int = 5,
) -> List[InferenceRecord]:
    """
    Batched version: send multiple documents in one API call for efficiency.
    Expects structured JSON output with one result per document.
    
    Args:
        batch_size: Number of documents to process per API call
    """
    # Provider instances
    instances: List[ModelProvider] = []
    for cfg in providers:
        if cfg.provider == "openai":
            instances.append(OpenAIProvider(cfg))
        elif cfg.provider == "litellm":
            instances.append(LiteLLMProvider(cfg))
        elif cfg.provider == "mock":
            instances.append(MockProvider(cfg))
        else:
            raise ValueError(f"Unsupported provider: {cfg.provider}")

    # Slice docs if requested
    df_ = df.copy()
    if limit_docs is not None:
        df_ = df_.head(int(limit_docs))

    records: List[InferenceRecord] = []
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    
    # Process in batches
    for batch_start in range(0, len(df_), batch_size):
        batch_df = df_.iloc[batch_start:batch_start + batch_size]
        
        # Build batch prompt with all documents
        docs_info = []
        for idx, row in batch_df.iterrows():
            doc_id = str(row.get("doc_id", idx))
            text = str(row[text_col])
            as_of_date = str(row[as_of_col])
            docs_info.append({
                'doc_id': doc_id,
                'text': text,
                'as_of_date': as_of_date
            })
        
        # Create batched prompt
        batch_prompt_user = f"""Analyze the following {len(docs_info)} documents and provide sentiment analysis for EACH ONE.

# Task
{task_instruction}

# Documents to Analyze
"""
        for i, doc in enumerate(docs_info):
            batch_prompt_user += f"\n---\nDocument ID: {doc['doc_id']}\n"
            batch_prompt_user += f"As-of Date: {doc['as_of_date']}\n"
            batch_prompt_user += f"Text: {doc['text'][:2000]}\n"  # Limit text length per doc
        
        batch_prompt_user += f"""
---

# Output Format
Return a JSON array with EXACTLY {len(docs_info)} objects, one per document in the SAME ORDER.
Each object must have:
{{
  "doc_id": "<the document ID>",
  "signal": <float 0-1>,
  "reasoning": "<brief explanation>",
  "stance": "<negative|neutral|positive>"
}}

Return ONLY valid JSON array, no other text. Example:
[
  {{"doc_id": "{docs_info[0]['doc_id']}", "signal": 0.65, "reasoning": "...", "stance": "positive"}},
  ...
]
"""
        
        batch_prompt = {
            'system': system_instructions or "You are a careful assistant for finance research.",
            'user': batch_prompt_user
        }
        
        # Run batch through each provider
        for inst in instances:
            try:
                out = inst.infer(batch_prompt)
                
                # Parse JSON response
                try:
                    # Find JSON array in output
                    start_idx = out.find('[')
                    end_idx = out.rfind(']')
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = out[start_idx:end_idx+1]
                        results = json.loads(json_str)
                        
                        # Create individual records for each document
                        for doc_info, result in zip(docs_info, results):
                            rec = InferenceRecord(
                                run_id=run_id,
                                doc_id=result.get('doc_id', doc_info['doc_id']),
                                provider_name=inst.cfg.name,
                                provider=inst.cfg.provider,
                                model=inst.cfg.model,
                                prompt=batch_prompt,
                                output_text=f"Signal: {result.get('signal', 0.5)}\n{result.get('reasoning', '')}\nStance: {result.get('stance', 'neutral')}",
                                meta={
                                    "temperature": inst.cfg.temperature,
                                    "top_p": inst.cfg.top_p,
                                    "seed": inst.cfg.seed,
                                    "max_tokens": inst.cfg.max_tokens,
                                    "batched": True,
                                    "batch_size": len(docs_info)
                                }
                            )
                            records.append(rec)
                    else:
                        # JSON parsing failed - create error records
                        for doc_info in docs_info:
                            rec = InferenceRecord(
                                run_id=run_id,
                                doc_id=doc_info['doc_id'],
                                provider_name=inst.cfg.name,
                                provider=inst.cfg.provider,
                                model=inst.cfg.model,
                                prompt=batch_prompt,
                                output_text=f"[PARSE_ERROR] Could not parse JSON from batch response: {out[:200]}",
                                meta={"batched": True, "batch_size": len(docs_info), "error": "json_parse_failed"}
                            )
                            records.append(rec)
                
                except json.JSONDecodeError as e:
                    # JSON parsing failed - create error records
                    for doc_info in docs_info:
                        rec = InferenceRecord(
                            run_id=run_id,
                            doc_id=doc_info['doc_id'],
                            provider_name=inst.cfg.name,
                            provider=inst.cfg.provider,
                            model=inst.cfg.model,
                            prompt=batch_prompt,
                            output_text=f"[PARSE_ERROR] JSON decode failed: {e}",
                            meta={"batched": True, "batch_size": len(docs_info), "error": "json_decode_error"}
                        )
                        records.append(rec)
                
            except Exception as e:
                # API call failed - create error records
                for doc_info in docs_info:
                    rec = InferenceRecord(
                        run_id=run_id,
                        doc_id=doc_info['doc_id'],
                        provider_name=inst.cfg.name,
                        provider=inst.cfg.provider,
                        model=inst.cfg.model,
                        prompt=batch_prompt,
                        output_text=f"[API_ERROR] {e}",
                        meta={"batched": True, "batch_size": len(docs_info), "error": "api_failed"}
                    )
                    records.append(rec)
            
            # Delay between batches
            if inst.cfg.provider != "mock" and request_delay > 0:
                time.sleep(request_delay)
    
    # Persist to disk
    jsonl_path = os.path.join(OUTPUT_DIR, f"{run_id}_batched.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    pq_path = os.path.join(OUTPUT_DIR, f"{run_id}_batched.parquet")
    pd.DataFrame([asdict(r) for r in records]).to_parquet(pq_path, index=False)

    return records

# ------------------------------
# Two-Stage Ensemble (Channel Subscores → Final Prediction)
# ------------------------------

def parse_channel_scores_from_output(output_text: str, channels: List[str]) -> Dict[str, float]:
    """
    Parse channel scores from LLM output.
    Handles multiple formats:
    - Multi-line: "Channel: inflation\nFacts: ...\nScore: 0.4"
    - Inline: "Channel: inflation | Score: 0.4"
    - Simple: "inflation: 0.4"
    - Assignment: "inflation = 0.4"
    Returns dict of {channel: score}
    """
    import re
    scores = {}
    
    # Strategy 1: Multi-line format (Channel: X ... Score: Y)
    # Split into channel blocks
    blocks = re.split(r'\n(?=Channel:|\n)', output_text, flags=re.IGNORECASE)
    
    for block in blocks:
        # Look for "Channel: <name>" and "Score: <value>" in the same block
        channel_match = re.search(r'Channel:\s*(\w+)', block, re.IGNORECASE)
        score_match = re.search(r'Score:\s*(\d+\.?\d*)', block, re.IGNORECASE)
        
        if channel_match and score_match:
            channel_name = channel_match.group(1).lower()
            score_value = float(score_match.group(1))
            
            # Normalize if needed
            if score_value > 1.0:
                score_value = score_value / 100.0
            
            # Check if this matches one of our channels
            for channel in channels:
                if channel.lower() == channel_name or channel.lower().replace('_', '') == channel_name.replace('_', ''):
                    scores[channel] = score_value
                    break
    
    # Strategy 2: Inline patterns (fallback for any channels not yet found)
    lines = output_text.lower().split('\n')
    
    for line in lines:
        for channel in channels:
            if channel in scores:
                continue  # Already found this channel
            
            channel_lower = channel.lower()
            # Try multiple inline patterns
            patterns = [
                (f"{channel_lower}:", r'(\d+\.?\d*)'),
                (f"{channel_lower}\s*=", r'(\d+\.?\d*)'),
                (f"{channel_lower}\s+", r'(\d+\.?\d*)'),
            ]
            
            for pattern, score_regex in patterns:
                if pattern in line or re.search(pattern, line):
                    # Extract number after the pattern
                    try:
                        # Find position after pattern
                        match = re.search(pattern + r'\s*' + score_regex, line)
                        if match:
                            score = float(match.group(1))
                            # Normalize if needed (assume 0-1 range)
                            if score > 1.0:
                                score = score / 100.0  # Convert percentage
                            scores[channel] = score
                            break
                    except Exception:
                        continue
    
    return scores

def aggregate_stage1_scores(
    stage1_records: List[InferenceRecord],
    channels: List[str]
) -> pd.DataFrame:
    """
    Parse all Stage 1 outputs to extract channel subscores.
    Returns DataFrame with columns: doc_id, provider_name, channel, score
    """
    results = []
    
    for rec in stage1_records:
        scores = parse_channel_scores_from_output(rec.output_text, channels)
        for channel, score in scores.items():
            results.append({
                "doc_id": rec.doc_id,
                "provider_name": rec.provider_name,
                "channel": channel,
                "score": score
            })
    
    return pd.DataFrame(results)

def run_stage2_ensemble(
    stage1_df: pd.DataFrame,
    providers: List[ProviderConfig],
    stage2_task: str,
    channels: List[str],
    request_delay: float = 0.5,
    aggregation: str = "mean"  # "mean", "median", or "per_provider"
) -> List[InferenceRecord]:
    """
    Stage 2: Use only channel subscores (not original text) to make final predictions.
    
    Args:
        stage1_df: DataFrame with columns [doc_id, provider_name, channel, score]
        providers: Stage 2 model configs
        stage2_task: Task instruction for Stage 2 (e.g., "predict economy state")
        channels: List of channel names
        aggregation: How to aggregate scores from multiple Stage 1 providers
        request_delay: Delay between API calls
    
    Returns:
        List of Stage 2 inference records
    """
    # Provider instances for Stage 2
    instances: List[ModelProvider] = []
    for cfg in providers:
        if cfg.provider == "openai":
            instances.append(OpenAIProvider(cfg))
        elif cfg.provider == "litellm":
            instances.append(LiteLLMProvider(cfg))
        elif cfg.provider == "mock":
            instances.append(MockProvider(cfg))
        else:
            raise ValueError(f"Unsupported provider: {cfg.provider}")
    
    records: List[InferenceRecord] = []
    run_id = time.strftime("run_%Y%m%d_%H%M%S_stage2")
    
    # Group by document
    for doc_id in stage1_df["doc_id"].unique():
        doc_data = stage1_df[stage1_df["doc_id"] == doc_id]
        
        # Aggregate scores across Stage 1 providers
        if aggregation == "mean":
            # Average scores across all Stage 1 providers
            agg_scores = doc_data.groupby("channel")["score"].mean().to_dict()
            score_summaries = [f"Document ID: {doc_id}\n\nChannel Subscores (averaged across Stage 1 models):"]
            for ch in channels:
                score = agg_scores.get(ch, 0.0)
                score_summaries.append(f"{ch}: {score:.3f}")
            input_text = "\n".join(score_summaries)
            
        elif aggregation == "median":
            # Median scores across all Stage 1 providers
            agg_scores = doc_data.groupby("channel")["score"].median().to_dict()
            score_summaries = [f"Document ID: {doc_id}\n\nChannel Subscores (median across Stage 1 models):"]
            for ch in channels:
                score = agg_scores.get(ch, 0.0)
                score_summaries.append(f"{ch}: {score:.3f}")
            input_text = "\n".join(score_summaries)
            
        else:  # per_provider
            # Keep separate scores for each Stage 1 provider
            score_summaries = [f"Document ID: {doc_id}\n\nChannel Subscores by Stage 1 provider:"]
            for provider in doc_data["provider_name"].unique():
                provider_data = doc_data[doc_data["provider_name"] == provider]
                score_summaries.append(f"\n{provider}:")
                for _, row in provider_data.iterrows():
                    score_summaries.append(f"  {row['channel']}: {row['score']:.3f}")
            input_text = "\n".join(score_summaries)
        
        # Build Stage 2 prompt (subscores only, no original text!)
        stage2_prompt = {
            "system": (
                "You are an expert economic forecaster. "
                "You will receive ONLY numeric channel scores (0-1 scale) extracted from economic documents. "
                "Make your prediction based solely on these aggregate scores. "
                "Do NOT make assumptions about the underlying text content."
            ),
            "user": (
                f"{input_text}\n\n"
                f"# Task\n{stage2_task}\n\n"
                f"Provide your prediction and reasoning. "
                f"Include a line 'Signal: <number>' with a final numeric prediction [0-1]."
            )
        }
        
        # Run through Stage 2 providers
        for inst in instances:
            out = inst.infer(stage2_prompt)
            rec = InferenceRecord(
                run_id=run_id,
                doc_id=doc_id,
                provider_name=inst.cfg.name,
                provider=inst.cfg.provider,
                model=inst.cfg.model,
                prompt=stage2_prompt,
                output_text=out,
                meta={
                    "stage": 2,
                    "aggregation": aggregation,
                    "temperature": inst.cfg.temperature,
                    "top_p": inst.cfg.top_p,
                    "seed": inst.cfg.seed,
                    "max_tokens": inst.cfg.max_tokens
                }
            )
            records.append(rec)
            
            # Delay between API calls
            if inst.cfg.provider != "mock" and request_delay > 0:
                time.sleep(request_delay)
    
    # Persist Stage 2 results
    jsonl_path = os.path.join(OUTPUT_DIR, f"{run_id}_stage2.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    
    pq_path = os.path.join(OUTPUT_DIR, f"{run_id}_stage2.parquet")
    pd.DataFrame([asdict(r) for r in records]).to_parquet(pq_path, index=False)

    return records

# ------------------------------
# Anti–Look-Ahead Tests (3.5)
# ------------------------------

def masked_text_guard(doc_text: str, cutoff_date: str) -> str:
    """
    A simple masked-text check: redact explicit future-looking markers.
    For a prototype, we remove year-like tokens >= cutoff year and explicit phrases.
    """
    try:
        cutoff_year = dtparser.parse(cutoff_date).year
    except Exception:
        cutoff_year = 10_000  # no-op if parse fails

    tokens = doc_text.split()
    masked = []
    for w in tokens:
        w_clean = "".join(ch for ch in w if ch.isdigit())
        if w_clean.isdigit():
            try:
                val = int(w_clean)
                if val >= cutoff_year:
                    masked.append("[MASK_YEAR]")
                    continue
            except Exception:
                pass
        if any(k in w.lower() for k in ["tomorrow", "next quarter", "next year", "guidance", "forecast", "outlook"]):
            masked.append("[MASK_FUTURE]")
        else:
            masked.append(w)
    return " ".join(masked)

def time_shift_subset(df: pd.DataFrame, time_col: str, cutoff: str) -> pd.DataFrame:
    """
    Filter rows strictly <= cutoff date for training/evaluation to emulate an 'as-of' world.
    """
    cutoff_ts = dtparser.parse(cutoff)
    return df[pd.to_datetime(df[time_col]) <= cutoff_ts].copy()

def sensitivity_grid(
    base_cfgs: List[ProviderConfig],
    temperatures: List[float],
    seeds: List[int],
) -> List[ProviderConfig]:
    """
    Expand a base set of provider configs into a Cartesian product across
    temperature and seed to probe output sensitivity.
    """
    out = []
    for cfg in base_cfgs:
        for t in temperatures:
            for s in seeds:
                new = ProviderConfig(
                    name=f"{cfg.name}-t{t}-s{s}",
                    provider=cfg.provider,
                    model=cfg.model,
                    temperature=t,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_tokens,
                    seed=s,
                    extra=cfg.extra or {}
                )
                out.append(new)
    return out

# ------------------------------
# Parsing model outputs for metrics
# ------------------------------

def extract_signal_from_output(output_text: str) -> float:
    """
    Parse 'Signal: x.xxx' or 'Score: x.xxx' from LLM output and normalize to [0-1] scale.
    Handles both 0-1 scale and 0-10 scale automatically.
    """
    for line in output_text.splitlines():
        line_lower = line.lower().strip()
        # Check for both "signal:" and "score:"
        if line_lower.startswith("signal:") or line_lower.startswith("score:"):
            try:
                value = float(line.split(":", 1)[1].strip())
                # Normalize to [0-1] if value is outside range
                if value > 1.0:
                    if value <= 10.0:
                        # 0-10 scale → normalize to 0-1
                        value = value / 10.0
                    elif value <= 100.0:
                        # 0-100 percentage scale → normalize to 0-1
                        value = value / 100.0
                    else:
                        # Invalid range, return as NaN
                        return float("nan")
                return value
            except Exception:
                pass
    return float("nan")

def compute_sensitivity_stats(rec_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple dispersion metrics (std, iqr) of model signals per doc.
    """
    df = rec_df.copy()
    df["signal"] = df["output_text"].apply(extract_signal_from_output)
    stats = df.groupby(["doc_id"])["signal"].agg(["count", "mean", "std"]).reset_index()
    return stats

# ------------------------------
# Excel Export Function
# ------------------------------

def create_excel_export(df_raw=None) -> io.BytesIO:
    """
    Create comprehensive Excel export with all results in separate sheets.
    Returns BytesIO object for download.
    
    Args:
        df_raw: Original dataframe with doc_id, timestamp, text columns
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Experimental Bias Detection Results (Section 5)
        if 'experiment_results' in st.session_state:
            df_exp = st.session_state['experiment_results']
            df_exp.to_excel(writer, sheet_name='Bias_Detection', index=False)
        
        # Sheet 2: Sensitivity Analysis Results
        if 'sensitivity_results' in st.session_state:
            df_sens = st.session_state['sensitivity_results']
            # Include parsed config info
            df_sens.to_excel(writer, sheet_name='Sensitivity_Analysis', index=False)
        
        if 'sensitivity_summary' in st.session_state:
            df_sens_sum = st.session_state['sensitivity_summary']
            df_sens_sum.to_excel(writer, sheet_name='Sensitivity_Summary', index=False)
        
        # Sheet 3: Two-Stage Stage 1 (Channel Scores)
        if 'two_stage_stage1_df' in st.session_state:
            df_s1 = st.session_state['two_stage_stage1_df']
            df_s1.to_excel(writer, sheet_name='Stage1_Channel_Scores', index=False)
        
        # Sheet 4: Two-Stage Stage 2 (Overall Predictions)
        if 'two_stage_stage2_df' in st.session_state:
            df_s2 = st.session_state['two_stage_stage2_df']
            df_s2.to_excel(writer, sheet_name='Stage2_Overall_Scores', index=False)
        
        # Sheet 5: Masked Data (Original + Masked texts)
        if 'text_masked' in st.session_state:
            if df_raw is not None and 'text' in df_raw.columns:
                # Full data with original and masked
                masked_data = pd.DataFrame({
                    'doc_id': df_raw['doc_id'],
                    'timestamp': df_raw['timestamp'],
                    'original_text': df_raw['text'],
                    'masked_text': st.session_state['text_masked']
                })
                masked_data.to_excel(writer, sheet_name='Masked_Data', index=False)
            else:
                # Just masked texts
                masked_df = pd.DataFrame({
                    'masked_text': st.session_state['text_masked']
                })
                masked_df.to_excel(writer, sheet_name='Masked_Data', index=False)
        
        # Sheet 6: Summary Data (Original + Summary texts)
        if 'text_summary' in st.session_state:
            if df_raw is not None and 'text' in df_raw.columns:
                # Full data with original, summary, and masked summary
                summary_data = pd.DataFrame({
                    'doc_id': df_raw['doc_id'],
                    'timestamp': df_raw['timestamp'],
                    'original_text': df_raw['text'],
                    'summary_text': st.session_state['text_summary'],
                    'summary_masked': st.session_state.get('text_summary_masked', [''] * len(st.session_state['text_summary']))
                })
                summary_data.to_excel(writer, sheet_name='Summary_Data', index=False)
            else:
                # Just summaries
                summary_df = pd.DataFrame({
                    'summary_text': st.session_state['text_summary']
                })
                summary_df.to_excel(writer, sheet_name='Summary_Data', index=False)
        
        # Sheet 7: Configuration Summary
        config_data = []
        config_data.append({'Parameter': 'App Version', 'Value': APP_VERSION})
        config_data.append({'Parameter': 'Export Date', 'Value': time.strftime('%Y-%m-%d %H:%M:%S')})
        
        if 'providers' in st.session_state:
            for i, p in enumerate(st.session_state.get('providers', [])):
                config_data.append({'Parameter': f'Provider {i+1}', 'Value': f"{p.name} ({p.model})"})
        
        pd.DataFrame(config_data).to_excel(writer, sheet_name='Configuration', index=False)
    
    output.seek(0)
    return output

def create_pdf_report(
    ai_summary_text: str,
    has_bias_detection: bool,
    has_sensitivity: bool,
    has_stage2: bool
) -> io.BytesIO:
    """
    Create comprehensive PDF report with ALL tables and visualizations from Summary Dashboard.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=20
    )
    
    subheading_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=12
    )
    
    # Title page with logo
    # Try to add TUM logo if available
    logo_path = "img/logo_tum.jpeg"
    if os.path.exists(logo_path):
        try:
            logo = RLImage(logo_path, width=1.5*inch, height=0.75*inch)
            logo.hAlign = 'LEFT'
            story.append(logo)
            story.append(Spacer(1, 0.2*inch))
        except Exception as e:
            # If logo fails to load, continue without it
            pass
    
    story.append(Paragraph("LOOK-AHEAD BIAS ANALYSIS REPORT", title_style))
    story.append(Paragraph("[AI-GENERATED REPORT]", styles['Normal']))
    story.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # AI Summary (extract just executive summary part)
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    # Parse AI text to extract executive summary
    lines = ai_summary_text.split('\n')
    in_exec_summary = False
    exec_lines = []
    for line in lines:
        if 'EXECUTIVE SUMMARY' in line:
            in_exec_summary = True
            continue
        if in_exec_summary and line.strip().startswith('1.'):
            break
        if in_exec_summary and line.strip():
            exec_lines.append(line.strip())
    
    exec_text = ' '.join(exec_lines) if exec_lines else "See detailed sections below."
    for para in exec_text.split('\n\n'):
        if para.strip():
            story.append(Paragraph(para.strip(), styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
    
    story.append(PageBreak())
    
    # Section 1: Bias Detection
    if has_bias_detection:
        story.append(Paragraph("1. BIAS DETECTION RESULTS", heading_style))
        df_exp = st.session_state['experiment_results']
        exp_summary = df_exp.groupby('corpus_version')['sentiment'].agg(['mean', 'std', 'count']).reset_index()
        
        # Convert dataframe to table
        table_data = [['Corpus Version', 'Mean', 'Std', 'Count']]
        for _, row in exp_summary.iterrows():
            table_data.append([
                str(row['corpus_version']),
                f"{row['mean']:.3f}",
                f"{row['std']:.3f}",
                str(int(row['count']))
            ])
        
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.2*inch))
        
        # Statistical interpretation
        if len(df_exp['corpus_version'].unique()) > 1:
            stats = compare_versions_statistically(df_exp)
            interpretation = interpret_bias_detection(stats)
            story.append(Paragraph("Interpretation:", styles['Heading3']))
            for line in interpretation.split('\n'):
                if line.strip():
                    story.append(Paragraph(line.strip(), styles['Normal']))
        
        story.append(PageBreak())
    
    # Section 2: Sensitivity Analysis
    if has_sensitivity:
        story.append(Paragraph("2. SENSITIVITY ANALYSIS RESULTS", heading_style))
        df_sens_sum = st.session_state['sensitivity_summary']
        
        story.append(Paragraph("Hyperparameter Testing (Temperature × Seed combinations):", subheading_style))
        
        # Table: First 15 rows
        sens_cols = ['corpus_version', 'base_provider', 'temperature', 'seed', 'mean', 'std'] if 'corpus_version' in df_sens_sum.columns else ['base_provider', 'temperature', 'seed', 'mean', 'std']
        table_data = [[col.replace('_', ' ').title() for col in sens_cols]]
        
        for _, row in df_sens_sum.head(15).iterrows():
            table_data.append([
                str(row[col]) if col in ['corpus_version', 'base_provider'] else f"{row[col]:.3f}" if isinstance(row[col], (int, float)) else str(row[col])
                for col in sens_cols
            ])
        
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.2*inch))
        
        if len(df_sens_sum) > 15:
            story.append(Paragraph(f"(Showing first 15 of {len(df_sens_sum)} configurations)", styles['Italic']))
        
        story.append(PageBreak())
    
    # Section 3: Two-Stage Results
    if has_stage2:
        story.append(Paragraph("3. CHANNEL-BASED TWO-STAGE ANALYSIS", heading_style))
        
        # Stage 1: Channel scores
        if 'two_stage_stage1_df' in st.session_state:
            df_s1 = st.session_state['two_stage_stage1_df']
            story.append(Paragraph("Stage 1: Average Channel Scores", subheading_style))
            
            channel_avg = df_s1.groupby('channel')['score'].agg(['mean', 'std', 'count']).reset_index()
            table_data = [['Channel', 'Mean', 'Std', 'Count']]
            for _, row in channel_avg.sort_values('mean', ascending=False).iterrows():
                table_data.append([
                    str(row['channel']),
                    f"{row['mean']:.3f}",
                    f"{row['std']:.3f}",
                    str(int(row['count']))
                ])
            
            t = Table(table_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(t)
            story.append(Spacer(1, 0.2*inch))
        
        # Stage 2: Overall predictions
        df_s2 = st.session_state['two_stage_stage2_df']
        story.append(Paragraph("Stage 2: Overall Sentiment Predictions", subheading_style))
        
        s2_summary = df_s2.groupby(['doc_id', 'provider'])['overall_score'].agg(['mean', 'std']).reset_index()
        
        # Convert to table
        table_data = [['Doc ID', 'Provider', 'Mean', 'Std']]
        for _, row in s2_summary.head(20).iterrows():
            table_data.append([
                str(row['doc_id']),
                str(row['provider']),
                f"{row['mean']:.3f}",
                f"{row['std']:.3f}"
            ])
        
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(PageBreak())
    
    # Section 4: Cross-Method Comparison (if multiple methods available)
    comparison_data = []
    if has_bias_detection:
        df_exp = st.session_state['experiment_results']
        for version in df_exp['corpus_version'].unique():
            ver_data = df_exp[df_exp['corpus_version'] == version]
            comparison_data.append({
                'Method': f"Direct ({version.title()})",
                'Mean': ver_data['sentiment'].mean(),
                'Std': ver_data['sentiment'].std(),
                'N': len(ver_data)
            })
    
    if has_stage2:
        df_s2 = st.session_state['two_stage_stage2_df']
        comparison_data.append({
            'Method': 'Channel-Based',
            'Mean': df_s2['overall_score'].mean(),
            'Std': df_s2['overall_score'].std(),
            'N': len(df_s2)
        })
    
    if len(comparison_data) > 1:
        story.append(Paragraph("4. CROSS-METHOD COMPARISON", heading_style))
        
        # Methods table
        story.append(Paragraph("Summary by Method:", subheading_style))
        table_data = [['Method', 'Mean', 'Std', 'N']]
        for item in comparison_data:
            table_data.append([
                item['Method'],
                f"{item['Mean']:.3f}",
                f"{item['Std']:.3f}",
                str(item['N'])
            ])
        
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.2*inch))
        
        # Statistical tests (t-tests)
        if len(comparison_data) >= 2:
            story.append(Paragraph("Statistical Tests (Pairwise T-Tests):", subheading_style))
            
            # Perform tests (reuse logic from dashboard)
            method_data = {}
            if has_bias_detection:
                df_exp = st.session_state['experiment_results']
                for version in df_exp['corpus_version'].unique():
                    method_data[f"Direct ({version.title()})"] = df_exp[df_exp['corpus_version'] == version]['sentiment'].dropna().values
            if has_stage2:
                method_data['Channel-Based'] = df_s2['overall_score'].dropna().values
            
            test_results = []
            method_names = list(method_data.keys())
            for i in range(len(method_names)):
                for j in range(i + 1, len(method_names)):
                    data1 = method_data[method_names[i]]
                    data2 = method_data[method_names[j]]
                    if len(data1) > 1 and len(data2) > 1:
                        t_stat, p_value = scipy_stats.ttest_ind(data1, data2)
                        pooled_std = np.sqrt(
                            ((len(data1)-1)*np.std(data1, ddof=1)**2 + (len(data2)-1)*np.std(data2, ddof=1)**2) / 
                            (len(data1)+len(data2)-2)
                        )
                        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
                        test_results.append([
                            f"{method_names[i]} vs {method_names[j]}",
                            f"{np.mean(data1) - np.mean(data2):.3f}",
                            f"{p_value:.4f}",
                            "Yes" if p_value < 0.05 else "No",
                            f"{cohens_d:.2f}"
                        ])
            
            if test_results:
                table_data = [['Comparison', 'Mean Diff', 'p-value', 'Significant', "Cohen's d"]]
                table_data.extend(test_results)
                
                t = Table(table_data, colWidths=[2.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 8)
                ]))
                story.append(t)
        
        story.append(PageBreak())
    
    # Disclaimer
    story.append(Paragraph("DISCLAIMER", heading_style))
    story.append(Paragraph(
        "This report was generated by an AI language model and should be reviewed by "
        "human researchers before use in publications or decision-making. All statistical "
        "tests and interpretations should be independently verified.",
        styles['Normal']
    ))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# ------------------------------
# Streamlit UI (3.6)
# ------------------------------

st.set_page_config(page_title="Look-Ahead Bias Lab", layout="wide")

# Add TUM logo to sidebar if available
logo_sidebar_path = "img/logo_tum.jpeg"
if os.path.exists(logo_sidebar_path):
    st.sidebar.image(logo_sidebar_path, use_column_width=True)
    st.sidebar.markdown("---")

st.sidebar.title("Look-Ahead Bias Lab")
st.sidebar.caption(f"v{APP_VERSION} ")

st.title("Look-Ahead Bias Lab")


st.markdown(
    "**This tool tests whether your text-based classifications remain stable when "
    "information that should be non-causal or unavailable is removed or compressed.**\n\n"
    "- **Goal:** Evaluate invariance under information ablation. If predictions change "
    "when dates, entities, or gendered terms are hidden—or when the text is compressed into a "
    "masked 50-word summary—your pipeline may rely on biased or temporally leaky signals.\n"
    "- **Inputs:** Upload a corpus (e.g., news headlines, filings, transcripts).\n"
    "- **Masking controls:** Choose to redact dates/periods, named entities, and gendered terms.\n"
    "- **Derived views (per document):** (1) Original, (2) Masked, (3) Masked summary (~50 words; same masking).\n"
    "- **Task execution:** Run your user-defined classification identically across all three views.\n"
    "- **Parameter sweep:** Vary key LLM parameters (e.g., temperature, seed) to probe robustness.\n"
    "- **Comparison tests:** Pairwise t-tests between result sets and reporting of effect sizes/practical differences.\n"
    "- **Interpretation:** Stable outputs suggest robustness; significant divergences indicate potential bias/leakage "
    "and point to masked variants as safer baselines.\n"
    "- **Reproducibility:** Prompts, masks, parameters, predictions, and test results are logged and exportable."
)



# --- Data upload / loading ---
st.header("Step 1: Upload your Dataset")
st.write("Upload a CSV or Parquet with columns: `doc_id`, `timestamp` (ISO), `text`.")

uploaded = st.file_uploader("Upload file", type=["csv", "parquet"])
if uploaded:
    if uploaded.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_parquet(uploaded)
else:
    # Minimal demo data
    df_raw = pd.DataFrame({
        "doc_id": ["A", "B", "C"],
        "timestamp": ["2020-09-30", "2020-10-31", "2020-12-15"],
        "text": [
            "CPI prints cool; inflation may ease. Management outlook for 2021 is upbeat.",
            "Fed signals steady policy. Employment trend is mixed heading into Q4.",
            "Earnings beat guidance; strong outlook next year 2021 and beyond."
        ]
    })

# Clean types
df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d")
df_raw["doc_id"] = df_raw["doc_id"].astype(str)

# Row limit option
st.write(f"**Total rows in dataset:** {len(df_raw)}")

col_limit1, col_limit2 = st.columns([3, 1])
with col_limit1:
    use_row_limit = st.checkbox(
        "Limit rows for analysis (useful for testing with large datasets)",
        value=False,
        key="use_row_limit"
    )
with col_limit2:
    if use_row_limit:
        row_limit = st.number_input(
            "Number of rows to use",
            min_value=1,
            max_value=len(df_raw),
            value=min(10, len(df_raw)),
            step=1,
            key="row_limit_value"
        )

# Apply row limit if enabled
if use_row_limit:
    # Check if row limit changed
    previous_limit = st.session_state.get('previous_row_limit', None)
    current_limit = row_limit
    
    df_raw = df_raw.head(row_limit)
    st.info(f"ℹ️ Using first {len(df_raw)} rows for all subsequent analyses")
    
    # Only clear cache if the limit actually changed
    if previous_limit != current_limit:
        st.warning(f"Row limit changed from {previous_limit} to {current_limit}. Clearing cached analyses.")
        for key in ['text_masked', 'masked_created', 'text_summary', 'summary_created', 
                    'text_summary_masked', 'experiment_results', 'sensitivity_results',
                    'two_stage_stage1_df', 'two_stage_stage2_df']:
            st.session_state.pop(key, None)
        st.session_state['previous_row_limit'] = current_limit
    elif previous_limit is None:
        # First time enabling limit
        st.session_state['previous_row_limit'] = current_limit
else:
    # Row limit disabled
    previous_was_limited = st.session_state.get('previous_row_limit', None)
    if previous_was_limited is not None:
        # Just disabled the limit - clear cache
        st.warning("Row limit disabled. Clearing cached analyses to use full dataset.")
        for key in ['text_masked', 'masked_created', 'text_summary', 'summary_created', 
                    'text_summary_masked', 'experiment_results', 'sensitivity_results',
                    'two_stage_stage1_df', 'two_stage_stage2_df']:
            st.session_state.pop(key, None)
        st.session_state.pop('previous_row_limit', None)

st.dataframe(df_raw.head(50))

# --- Simple Sentiment Prompt (Section 2) ---


# A1: User's sentiment question
st.header("Step 2: Define Your Question")
user_prompt = st.text_area(
    "What do you want to know about the documents?",
    value="What is the overall sentiment about the economy? Is it positive or negative?",
    help="Keep it simple - we'll add bias protection automatically",
    height=100
)

# A2: Select what to mask
st.header("Step 3:Select What to Mask")
st.write("Choose which elements to mask in your corpus. Time/dates masked by default to prevent look-ahead bias.")

masking_categories = st.multiselect(
    "Select categories to mask",
    options=[
        "Time (dates, years, quarters)",
        "Organizations (companies, institutions)", 
        "Numbers (metrics, percentages, values)",
        "Locations (cities, countries, regions)",
        "Person Names",
        "Gender References",
        "Product/Brand Names"
    ],
    default=["Time (dates, years, quarters)"],
    help="LLM will mask only selected categories. More masking = more bias protection.",
    key="masking_categories_select"
)

# Store in session state
st.session_state['masking_categories'] = masking_categories

if masking_categories:
    st.info(f"✓ Will mask: {', '.join(masking_categories)}")
else:
    st.warning("⚠️ No categories selected - text will not be masked")

# No channels in basic sentiment analysis
channels_list = []





# --- Providers (3.3) ---
st.header("Step 4: Select which LLM providers to use.")

# Provider selection with API key management
st.write("Connect to LLM providers. Click buttons to configure API keys.")
st.info(
    "API keys entered here are stored temporarily on the Streamline server only "
    "for the duration of your app session. For full control over your credentials, "
    "you can download the app from GitHub and run it locally. Please upload keys "
    "here only if you are comfortable with this risk."
)

# Check which providers are already configured
openai_configured = bool(os.environ.get("OPENAI_API_KEY") or (hasattr(st, 'secrets') and st.secrets.get("OPENAI_API_KEY")))
claude_configured = bool(os.environ.get("ANTHROPIC_API_KEY") or (hasattr(st, 'secrets') and st.secrets.get("ANTHROPIC_API_KEY")))
gemini_configured = bool(os.environ.get("GOOGLE_API_KEY") or (hasattr(st, 'secrets') and st.secrets.get("GOOGLE_API_KEY")))

col_p1, col_p2, col_p3 = st.columns(3)

# OpenAI
with col_p1:
    st.write("**OpenAI**")
    if openai_configured:
        st.success("✓ Connected")
        use_openai = st.checkbox("Use OpenAI", value=True, key="use_openai_check")
        if use_openai:
            openai_model = st.selectbox("Model", 
                                       ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                                       index=0, key="openai_model_select")
    else:
        use_openai = False
        if st.button("🔌 Connect to OpenAI", key="connect_openai"):
            st.session_state['show_openai_key_input'] = True
    
    # Show API key input form
    if st.session_state.get('show_openai_key_input', False):
        with st.form("openai_key_form"):
            api_key = st.text_input("Enter OpenAI API Key", type="password")
            submitted = st.form_submit_button("💾 Save & Test")
            
            if submitted and api_key:
                # Save to secrets.toml
                secrets_path = ".streamlit/secrets.toml"
                os.makedirs(".streamlit", exist_ok=True)
                
                # Read existing secrets
                existing_secrets = {}
                if os.path.exists(secrets_path):
                    with open(secrets_path, 'r') as f:
                        for line in f:
                            if '=' in line and not line.strip().startswith('#'):
                                key, val = line.split('=', 1)
                                existing_secrets[key.strip()] = val.strip()
                
                # Add/update OpenAI key
                existing_secrets['OPENAI_API_KEY'] = f'"{api_key}"'
                
                # Write back
                with open(secrets_path, 'w') as f:
                    for key, val in existing_secrets.items():
                        f.write(f'{key} = {val}\n')
                
                # Test connection
                os.environ['OPENAI_API_KEY'] = api_key
                try:
                    test_cfg = ProviderConfig(name="test", provider="openai", model="gpt-4o-mini", temperature=0)
                    test_prov = OpenAIProvider(test_cfg)
                    success, message = test_provider_connection(test_prov)
                    
                    if success:
                        st.success(f"✅ Connected successfully! {message}")
                        st.session_state['show_openai_key_input'] = False
                    else:
                        st.error(f"❌ Connection failed: {message}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# Claude
with col_p2:
    st.write("**Claude (Anthropic)**")
    if claude_configured:
        st.success("✓ Connected")
        use_claude = st.checkbox("Use Claude", value=True, key="use_claude_check")
        if use_claude:
            claude_model = st.selectbox("Model",
                                       ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
                                       index=1, key="claude_model_select")
    else:
        use_claude = False
        if st.button("🔌 Connect to Claude", key="connect_claude"):
            st.session_state['show_claude_key_input'] = True
    
    if st.session_state.get('show_claude_key_input', False):
        with st.form("claude_key_form"):
            api_key = st.text_input("Enter Anthropic API Key", type="password")
            submitted = st.form_submit_button("💾 Save & Test")
            
            if submitted and api_key:
                secrets_path = ".streamlit/secrets.toml"
                os.makedirs(".streamlit", exist_ok=True)
                
                existing_secrets = {}
                if os.path.exists(secrets_path):
                    with open(secrets_path, 'r') as f:
                        for line in f:
                            if '=' in line and not line.strip().startswith('#'):
                                key, val = line.split('=', 1)
                                existing_secrets[key.strip()] = val.strip()
                
                existing_secrets['ANTHROPIC_API_KEY'] = f'"{api_key}"'
                
                with open(secrets_path, 'w') as f:
                    for key, val in existing_secrets.items():
                        f.write(f'{key} = {val}\n')
                
                os.environ['ANTHROPIC_API_KEY'] = api_key
                try:
                    test_cfg = ProviderConfig(name="test", provider="litellm", model="claude-3-5-haiku-20241022", temperature=0)
                    test_prov = LiteLLMProvider(test_cfg)
                    success, message = test_provider_connection(test_prov)
                    
                    if success:
                        st.success(f"✅ Connected successfully! {message}")
                        st.session_state['show_claude_key_input'] = False
                        st.rerun()
                    else:
                        st.error(f"❌ Connection failed: {message}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# Gemini
with col_p3:
    st.write("**Gemini (Google)**")
    if gemini_configured:
        st.success("✓ Connected")
        use_gemini = st.checkbox("Use Gemini", value=True, key="use_gemini_check")
        if use_gemini:
            gemini_model = st.selectbox("Model",
                                       ["gemini/gemini-2.0-flash-exp", "gemini/gemini-1.5-pro"],
                                       index=0, key="gemini_model_select")
    else:
        use_gemini = False
        if st.button("🔌 Connect to Gemini", key="connect_gemini"):
            st.session_state['show_gemini_key_input'] = True
    
    if st.session_state.get('show_gemini_key_input', False):
        with st.form("gemini_key_form"):
            api_key = st.text_input("Enter Google API Key", type="password")
            submitted = st.form_submit_button("💾 Save & Test")
            
            if submitted and api_key:
                secrets_path = ".streamlit/secrets.toml"
                os.makedirs(".streamlit", exist_ok=True)
                
                existing_secrets = {}
                if os.path.exists(secrets_path):
                    with open(secrets_path, 'r') as f:
                        for line in f:
                            if '=' in line and not line.strip().startswith('#'):
                                key, val = line.split('=', 1)
                                existing_secrets[key.strip()] = val.strip()
                
                existing_secrets['GOOGLE_API_KEY'] = f'"{api_key}"'
                
                with open(secrets_path, 'w') as f:
                    for key, val in existing_secrets.items():
                        f.write(f'{key} = {val}\n')
                
                os.environ['GOOGLE_API_KEY'] = api_key
                try:
                    test_cfg = ProviderConfig(name="test", provider="litellm", model="gemini/gemini-2.0-flash-exp", temperature=0)
                    test_prov = LiteLLMProvider(test_cfg)
                    success, message = test_provider_connection(test_prov)
                    
                    if success:
                        st.success(f"✅ Connected successfully! {message}")
                        st.session_state['show_gemini_key_input'] = False
                        st.rerun()
                    else:
                        st.error(f"❌ Connection failed: {message}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# --------------------------------
# 3c) Performance Settings (code-configured)
# --------------------------------

# ---- CONFIG (edit here, not in the UI) ----
LIMIT_DOCS_DEFAULT = 0          # 0 = no limit, otherwise max number of docs
REQUEST_DELAY_DEFAULT = 1.0     # seconds between API calls
USE_BATCHING_DEFAULT = True     # True = enable batching, False = one-doc-per-call
BATCH_SIZE_DEFAULT = 5          # docs per API call when batching is enabled

# ---- Apply config ----
limit_docs = int(LIMIT_DOCS_DEFAULT) if LIMIT_DOCS_DEFAULT is not None else 0
request_delay = float(REQUEST_DELAY_DEFAULT)
use_batching = bool(USE_BATCHING_DEFAULT)

if use_batching:
    batch_size = int(max(1, BATCH_SIZE_DEFAULT))
else:
    batch_size = 1  # no batching



# ---- Batching efficiency info (same logic as before, but no controls) ----
num_docs_estimate = len(df_raw) if 'df_raw' in locals() else 10
if limit_docs and limit_docs > 0:
    num_docs_estimate = min(num_docs_estimate, limit_docs)

if use_batching and batch_size > 0:
    num_batches = math.ceil(num_docs_estimate / batch_size)
    savings = num_docs_estimate - num_batches
    reduction_pct = (savings / num_docs_estimate * 100) if num_docs_estimate > 0 else 0

    



# Build provider list
providers: List[ProviderConfig] = []

if use_openai:
    providers.append(ProviderConfig(
        name=f"OpenAI-{openai_model}",
        provider="openai",
        model=openai_model,
        temperature=0.2,
        top_p=1.0,
        max_tokens=256,
        seed=RANDOM_DEFAULT_SEED,
    ))

if use_claude:
    providers.append(ProviderConfig(
        name=f"Claude-{claude_model.split('-')[2] if len(claude_model.split('-')) > 2 else 'model'}",
        provider="litellm",
        model=claude_model,
        temperature=0.2,
        top_p=1.0,
        max_tokens=256,
        seed=RANDOM_DEFAULT_SEED,
    ))

if use_gemini:
    providers.append(ProviderConfig(
        name=f"Gemini-{gemini_model.split('/')[-1]}",
        provider="litellm",
        model=gemini_model,
        temperature=0.2,
        top_p=1.0,
        max_tokens=256,
        seed=RANDOM_DEFAULT_SEED,
    ))

# Show selected providers
if providers:
    st.info(f"📋 Selected providers: {', '.join([p.name for p in providers])}")
else:
    st.warning("⚠️ No providers selected! At least one provider is required to run.")


## ------------------------------
# SECTION 4: Corpus Transformations (NEW, STREAMLINED)
# ------------------------------


# ---- CONFIG (no UI, only code) ----
MASKING_METHOD_DEFAULT = "LLM-based"  # "LLM-based" or "Keyword"
MASK_PROVIDER_INDEX_DEFAULT = 0       # which provider in `providers` for masking

SUMMARY_LENGTH_DEFAULT = 50          # words
SUMMARY_PROVIDER_INDEX_DEFAULT = 0    # which provider in `providers` for summaries

NUM_RUNS_DEFAULT = 2                  # bias experiment: runs per version
VERSIONS_TO_TEST_DEFAULT = ["original", "masked", "summary"]  # which corpus versions to compare

# Sensitivity analysis config
ENABLE_SENSITIVITY_DEFAULT = True
TEMPS_DEFAULT = [0.2, 0.8]
SEEDS_DEFAULT = [7, 11]
SENSITIVITY_RUNS_DEFAULT = 2
SENSITIVITY_VERSIONS_DEFAULT = ["original", "masked"]  # which versions to include in sensitivity grid

# ---- Restore masked/summary from session if needed ----
# Restore masked data from session state if available (only if length matches)
if 'text_masked' in st.session_state and 'text_masked' not in df_raw.columns:
    if len(st.session_state['text_masked']) == len(df_raw):
        df_raw["text_masked"] = st.session_state['text_masked']
    else:
        st.session_state.pop('text_masked', None)
        st.session_state.pop('masked_created', None)

# Restore summary data from session state if available (only if length matches)
if 'text_summary' in st.session_state and 'text_summary' not in df_raw.columns:
    if len(st.session_state['text_summary']) == len(df_raw):
        df_raw["text_summary"] = st.session_state['text_summary']
    else:
        st.session_state.pop('text_summary', None)
        st.session_state.pop('summary_created', None)

if 'text_summary_masked' in st.session_state and 'text_summary_masked' not in df_raw.columns:
    if len(st.session_state['text_summary_masked']) == len(df_raw):
        df_raw["text_summary_masked"] = st.session_state['text_summary_masked']
    else:
        st.session_state.pop('text_summary_masked', None)


# ------------------------------
# Helper: masking + summaries
# ------------------------------
def run_masking_and_summaries(df_raw, providers, request_delay):
    if not providers:
        st.error("No providers configured in Section 3. Cannot run masking/summaries.")
        return df_raw

    # ------ Masking ------
    masking_method = MASKING_METHOD_DEFAULT
    mask_provider_idx = min(MASK_PROVIDER_INDEX_DEFAULT, len(providers) - 1)
    mask_cfg = providers[mask_provider_idx]

    st.info(f"Using **{masking_method}** masking with provider: **{mask_cfg.name}**")

    if masking_method == "LLM-based":
        with st.spinner(f"Using LLM to mask {len(df_raw)} documents..."):
            try:
                if mask_cfg.provider == "openai":
                    mask_prov = OpenAIProvider(mask_cfg)
                elif mask_cfg.provider == "litellm":
                    mask_prov = LiteLLMProvider(mask_cfg)
                else:
                    mask_prov = MockProvider(mask_cfg)

                masked_texts = []
                progress = st.progress(0)
                # Get selected masking categories from session state
                selected_categories = st.session_state.get('masking_categories', ["Time (dates, years, quarters)"])
                
                for i, row in df_raw.iterrows():
                    masked = llm_based_masking(
                        row["text"], 
                        row["timestamp"], 
                        mask_prov,
                        categories_to_mask=selected_categories
                    )
                    masked_texts.append(masked)
                    progress.progress((i + 1) / len(df_raw))

                    if mask_cfg.provider != "mock" and request_delay > 0:
                        time.sleep(request_delay)

                df_raw["text_masked"] = masked_texts
                progress.empty()

                st.session_state['masked_created'] = True
                st.session_state['text_masked'] = masked_texts
                st.session_state['masking_method'] = "LLM-based (Recommended)"

                st.success("✓ LLM-based masking completed!")

            except Exception as e:
                st.error(f"LLM masking failed: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        with st.spinner("Applying keyword-based masking..."):
            # Get selected masking categories
            selected_categories = st.session_state.get('masking_categories', ["Time (dates, years, quarters)"])
            
            masked_texts = df_raw.apply(
                lambda r: keyword_based_masking(
                    r["text"], 
                    r["timestamp"],
                    categories_to_mask=selected_categories
                ),
                axis=1
            ).tolist()
            df_raw["text_masked"] = masked_texts
            st.session_state['masked_created'] = True
            st.session_state['text_masked'] = masked_texts
            st.session_state['masking_method'] = "Keyword-based (Simple)"
        st.success("✓ Keyword-based masking completed!")

    # Show example
    if 'text_masked' in df_raw.columns and len(df_raw) > 0:
        st.write("**Example (first document, masking):**")
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.write("*Original:*")
            st.code(df_raw.iloc[0]["text"][:300])
        with col_ex2:
            st.write("*Masked:*")
            st.code(df_raw.iloc[0]["text_masked"][:300])

    # ------ Summaries ------
    st.info("Generating fact-focused summaries...")
    summary_provider_idx = min(SUMMARY_PROVIDER_INDEX_DEFAULT, len(providers) - 1)
    summary_cfg = providers[summary_provider_idx]

    try:
        if summary_cfg.provider == "openai":
            summary_prov = OpenAIProvider(summary_cfg)
        elif summary_cfg.provider == "litellm":
            summary_prov = LiteLLMProvider(summary_cfg)
        else:
            summary_prov = MockProvider(summary_cfg)

        with st.spinner(f"Generating summaries for {len(df_raw)} documents..."):
            summaries = summarize_corpus(
                df_raw["text"].tolist(),
                summary_prov,
                max_length=SUMMARY_LENGTH_DEFAULT,
                request_delay=request_delay if 'request_delay' in locals() else 1.0
            )
            df_raw["text_summary"] = summaries

        st.write("Applying masking to summaries...")

        masking_method_used = st.session_state.get('masking_method', 'Keyword-based (Simple)')

        # Get selected masking categories (same as main text)
        selected_categories = st.session_state.get('masking_categories', ["Time (dates, years, quarters)"])
        
        if masking_method_used.startswith("LLM-based"):
            masked_summaries = []
            progress_mask = st.progress(0)
            for i, row in df_raw.iterrows():
                masked = llm_based_masking(
                    row["text_summary"], 
                    row["timestamp"], 
                    summary_prov,
                    categories_to_mask=selected_categories
                )
                masked_summaries.append(masked)
                progress_mask.progress((i + 1) / len(df_raw))
                if summary_cfg.provider != "mock" and request_delay > 0:
                    time.sleep(request_delay)
            df_raw["text_summary_masked"] = masked_summaries
            progress_mask.empty()
        else:
            df_raw["text_summary_masked"] = df_raw.apply(
                lambda r: keyword_based_masking(
                    r["text_summary"], 
                    r["timestamp"],
                    categories_to_mask=selected_categories
                ),
                axis=1
            ).tolist()

        st.session_state['summary_created'] = True
        st.session_state['text_summary'] = summaries
        st.session_state['text_summary_masked'] = df_raw["text_summary_masked"].tolist()

        st.success("✓ Summaries created and masked!")

        if len(df_raw) > 0:
            st.write("**Example (first document, summaries):**")
            st.write("*Original:*")
            st.code(df_raw.iloc[0]["text"][:200])
            st.write("*Summary:*")
            st.code(df_raw.iloc[0]["text_summary"])

    except Exception as e:
        st.error(f"Summarization failed: {e}")
        import traceback
        st.code(traceback.format_exc())

    return df_raw


# ------------------------------
# SECTION 5: Bias Detection (STREAMLINED)
# ------------------------------


def get_versions_available(df_raw):
    versions = ["original"]
    if 'text_masked' in df_raw.columns:
        versions.append("masked")
    if 'text_summary_masked' in df_raw.columns:
        versions.append("summary")
    return versions

def run_bias_detection_experiment(df_raw, providers, request_delay):
    versions_available = get_versions_available(df_raw)
    if len(versions_available) == 1:
        st.warning("⚠️ Create masked and/or summarized versions in Section 4 first")
        return

    st.success(f"✓ {len(versions_available)} versions available: {', '.join(versions_available)}")

    # Use configured defaults, but intersect with what is actually available
    versions_to_test = [v for v in VERSIONS_TO_TEST_DEFAULT if v in versions_available]
    if len(versions_to_test) < 2:
        # ensure at least 2
        versions_to_test = versions_available

    num_runs = NUM_RUNS_DEFAULT

    if not providers:
        st.error("Configure providers in Section 3 first")
        return

    # Estimate calls
    lim_docs = int(limit_docs) if 'limit_docs' in locals() and limit_docs and int(limit_docs) > 0 else len(df_raw)
    if use_batching and batch_size > 1:
        num_batches_per_run = math.ceil(lim_docs / batch_size)
        total_calls = num_batches_per_run * len(providers) * len(versions_to_test) * num_runs
       
    else:
        total_calls = lim_docs * len(providers) * len(versions_to_test) * num_runs
        st.info(
            f"📊 **Experiment size:** {total_calls} total API calls "
            f"({lim_docs} docs × {len(providers)} providers × {len(versions_to_test)} versions × {num_runs} runs)"
        )


    with st.spinner(f"Running experiment ({total_calls} calls)..."):
        try:
            results = []
            progress_bar = st.progress(0)
            total_iterations = len(versions_to_test) * num_runs * len(providers)
            current_iteration = 0

            for version in versions_to_test:
                if version == "original":
                    text_col = "text"
                elif version == "masked":
                    text_col = "text_masked"
                else:
                    text_col = "text_summary_masked"

                for run in range(num_runs):
                    for provider_cfg in providers:
                        if use_batching and batch_size > 1:
                            version_records = run_ensemble_batched(
                                df=df_raw,
                                providers=[provider_cfg],
                                task_instruction=task_instruction,
                                text_col=text_col,
                                channels=[],
                                system_instructions=system_instructions,
                                limit_docs=int(limit_docs) if 'limit_docs' in locals() and limit_docs and int(limit_docs) > 0 else None,
                                request_delay=request_delay if 'request_delay' in locals() else 1.0,
                                batch_size=batch_size
                            )
                        else:
                            version_records = run_ensemble(
                                df=df_raw,
                                providers=[provider_cfg],
                                task_instruction=task_instruction,
                                text_col=text_col,
                                channels=[],
                                system_instructions=system_instructions,
                                limit_docs=int(limit_docs) if 'limit_docs' in locals() and limit_docs and int(limit_docs) > 0 else None,
                                request_delay=request_delay if 'request_delay' in locals() else 1.0
                            )

                        for rec in version_records:
                            sentiment = extract_signal_from_output(rec.output_text)
                            results.append({
                                "doc_id": rec.doc_id,
                                "corpus_version": version,
                                "run": run,
                                "provider": rec.provider_name,
                                "sentiment": sentiment
                            })

                        current_iteration += 1
                        progress_bar.progress(current_iteration / total_iterations)

            df_results = pd.DataFrame(results)

            max_sentiment = df_results['sentiment'].max()
            if max_sentiment > 1.0:
                st.warning(
                    f"⚠️ Note: Some LLM outputs had scores > 1.0 (max: {max_sentiment:.2f}). "
                    f"These have been automatically normalized to 0-1 scale. "
                    f"Please update your prompt to be more specific about using 0-1 scale."
                )

            df_results = df_results[df_results['sentiment'].notna()].copy()
            st.session_state['experiment_results'] = df_results

            progress_bar.empty()
            st.success(f"✓ Experiment completed: {len(results)} measurements")

            # D3: Statistical Analysis
            stats = compare_versions_statistically(df_results)

            st.markdown("### 🎯 Bias Detection Results")
            interpretation = interpret_bias_detection(stats)
            st.markdown(interpretation)

            with st.expander("📈 Detailed Statistics"):
                st.json(stats)

            

        except Exception as e:
            st.error(f"Experiment failed: {e}")
            import traceback
            st.code(traceback.format_exc())


# Show previous results if available
if 'experiment_results' in st.session_state:
    with st.expander("📋 View Previous Experiment Results"):
        st.dataframe(st.session_state['experiment_results'])

# ------------------------------
# Sensitivity analysis (STREAMLINED)
# ------------------------------


def run_sensitivity_grid(df_raw, providers, request_delay):
    if not ENABLE_SENSITIVITY_DEFAULT:
        st.info("Sensitivity analysis disabled in code configuration.")
        return

    openai_providers = [p for p in providers if 'openai' in p.provider.lower() or 'gpt' in p.model.lower()]
    if not openai_providers:
        st.warning("⚠️ No OpenAI providers selected. Sensitivity analysis requires at least one OpenAI model.")
        return

    st.success(f"✓ Using {len(openai_providers)} OpenAI provider(s) for sensitivity analysis: "
               f"{', '.join([p.name for p in openai_providers])}")

    temps_list = TEMPS_DEFAULT
    seeds_list = SEEDS_DEFAULT
    sensitivity_runs = SENSITIVITY_RUNS_DEFAULT

    sensitivity_versions_available = get_versions_available(df_raw)
    sensitivity_versions = [v for v in SENSITIVITY_VERSIONS_DEFAULT if v in sensitivity_versions_available]
    if not sensitivity_versions:
        sensitivity_versions = sensitivity_versions_available[:1]

    prov_grid = sensitivity_grid(openai_providers, temps_list, seeds_list)

    num_docs = len(df_raw)
    if limit_docs and int(limit_docs) > 0:
        num_docs = min(num_docs, int(limit_docs))
    num_providers = len(prov_grid)
    num_versions = len(sensitivity_versions)

    if use_batching and batch_size > 1:
        num_batches_per_run = math.ceil(num_docs / batch_size)
        total_api_calls = num_batches_per_run * num_providers * num_versions * sensitivity_runs
    else:
        total_api_calls = num_docs * num_providers * num_versions * sensitivity_runs

    if total_api_calls > 0:
        estimated_time = total_api_calls * request_delay
    


    with st.spinner(f"Running sensitivity grid across {len(prov_grid)} configs × {len(sensitivity_versions)} versions × {sensitivity_runs} runs..."):
        try:
            lim = int(limit_docs) if 'limit_docs' in locals() and limit_docs and int(limit_docs) > 0 else None

            all_records = []
            progress_bar = st.progress(0)
            total_iterations = len(sensitivity_versions) * sensitivity_runs
            current_iter = 0

            for version in sensitivity_versions:
                if version == "original":
                    text_col = "text"
                elif version == "masked":
                    text_col = "text_masked"
                else:
                    text_col = "text_summary_masked"

                for run_num in range(sensitivity_runs):
                    if use_batching and batch_size > 1:
                        version_records = run_ensemble_batched(
                            df=df_raw,
                providers=prov_grid,
                task_instruction=task_instruction,
                as_of_col="timestamp",
                            text_col=text_col,
                            channels=[],
                system_instructions=system_instructions,
                limit_docs=lim,
                            request_delay=request_delay if 'request_delay' in locals() else 1.0,
                            batch_size=batch_size
                        )
                    else:
                        version_records = run_ensemble(
                            df=df_raw,
                            providers=prov_grid,
                            task_instruction=task_instruction,
                            as_of_col="timestamp",
                            text_col=text_col,
                            channels=[],
                            system_instructions=system_instructions,
                            limit_docs=lim,
                            request_delay=request_delay if 'request_delay' in locals() else 1.0
                        )

                    for rec in version_records:
                        rec_dict = asdict(rec)
                        rec_dict['corpus_version'] = version
                        rec_dict['run_number'] = run_num
                        all_records.append(rec_dict)

                    current_iter += 1
                    progress_bar.progress(current_iter / total_iterations)

            progress_bar.empty()
            st.success(f"✓ Completed {len(all_records)} model calls in sensitivity grid")

            df_sens = pd.DataFrame(all_records)
            df_sens["signal"] = df_sens["output_text"].apply(extract_signal_from_output)

            def parse_temp_seed(name):
                temp_match = re.search(r't([\d.]+)', name)
                seed_match = re.search(r's(\d+)', name)
                temp = float(temp_match.group(1)) if temp_match else 0.0
                seed = int(seed_match.group(1)) if seed_match else 7
                base_name = name.split('-t')[0] if '-t' in name else name
                return pd.Series({'base_provider': base_name, 'temperature': temp, 'seed': seed})

            config_info = df_sens['provider_name'].apply(parse_temp_seed)
            df_sens = pd.concat([df_sens, config_info], axis=1)

            sensitivity_summary = df_sens.groupby(
                ['corpus_version', 'base_provider', 'temperature', 'seed']
            )['signal'].agg([
                ('mean', 'mean'),
                ('std', 'std'),
                ('count', 'count')
            ]).reset_index()

            sensitivity_summary_openai = sensitivity_summary[
                sensitivity_summary['base_provider'].str.contains('OpenAI|gpt', case=False, na=False)
            ]

          
            st.session_state['sensitivity_results'] = df_sens
            st.session_state['sensitivity_summary'] = sensitivity_summary

        except Exception as e:
            st.error(f"Sensitivity grid failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# ------------------------------
# COST ESTIMATE
# ------------------------------
st.write("---")
st.subheader("💰 Cost Estimate")
st.info("Estimated API costs for running the complete analysis. The values shown here are rough, educational estimates based on standard per-token pricing. Real billing may differ depending on model changes, region, billing tier, and token throughput—users should always verify current pricing directly on the provider websites.")

if 'providers' in locals() and providers and 'df_raw' in locals():
    # ------------------------------
    # Token estimate (rough)
    # ------------------------------
    num_docs = len(df_raw)
    avg_doc_length = df_raw['text'].str.len().mean() if 'text' in df_raw.columns else 500
    tokens_per_doc = int(avg_doc_length / 4)  # Rough estimate: 4 chars per token
    prompt_overhead = 200  # System prompt + instructions
    total_tokens_per_doc = tokens_per_doc + prompt_overhead

    # This is *only* an experiment-level simplification:
    # original + masked, 2 runs each
    num_versions = 2  # original + masked
    num_runs = 2
    num_seed = 2
    num_temperatures = 2
    total_calls = num_docs * len(providers) * num_versions * num_runs * num_seed * num_temperatures * -1    

    # With batching
    if use_batching and batch_size > 1:
        num_batches = math.ceil(num_docs / batch_size)
        tokens_per_batch = (batch_size * tokens_per_doc) + prompt_overhead

        summarise_calls = num_batches 
        masking_calls = num_batches 

        total_calls_batched = num_batches * len(providers) * num_versions * num_runs* num_seed * num_temperatures + summarise_calls + masking_calls

        # Batched calls have multiple docs per call
        summarize_tokens = tokens_per_batch * num_batches
        masking_tokens = tokens_per_batch * num_batches
        total_tokens = num_batches * tokens_per_batch * len(providers) * num_versions * num_runs* num_seed * num_temperatures + summarize_tokens + masking_tokens
    else:
        total_calls_batched = total_calls
        total_tokens = total_calls * total_tokens_per_doc

    # ------------------------------
    # Split into input vs output tokens (simple assumption)
    # ------------------------------
    # For this dashboard: lots of analysis, short-ish outputs.
    # Adjust this if you know your pipeline’s profile better.
    OUTPUT_FRACTION = 0.50  # 25% output, 75% input as a rough default

    total_input_tokens = total_tokens * (1 - OUTPUT_FRACTION)
    total_output_tokens = total_tokens * OUTPUT_FRACTION

    # ------------------------------
    # Cost per 1M tokens (input / output)
    # ------------------------------
    cost_per_1m_input = {}
    cost_per_1m_output = {}

    for prov in providers:
        model = prov.model.lower()

        # --- OpenAI ---
        if "gpt-4o-mini" in model:
            # GPT-4o mini: $0.15 in / $0.60 out per 1M tokens
            cost_per_1m_input[prov.name] = 0.15
            cost_per_1m_output[prov.name] = 0.60

        elif "gpt-4o" in model:
            # GPT-4o: $2.50 in / $10 out per 1M tokens
            cost_per_1m_input[prov.name] = 2.50
            cost_per_1m_output[prov.name] = 10.00

        # You can add GPT-4.1 / GPT-5 mappings here once your stack uses them.

        # --- Anthropic ---
        elif "claude-3-5-sonnet" in model or "claude-3.5-sonnet" in model:
            # Claude 3.5 Sonnet: $3 in / $15 out
            cost_per_1m_input[prov.name] = 3.00
            cost_per_1m_output[prov.name] = 15.00

        elif "claude-3-5-haiku" in model or "claude-3.5-haiku" in model:
            # Claude 3.5 Haiku: $0.80 in / $4 out
            cost_per_1m_input[prov.name] = 0.80
            cost_per_1m_output[prov.name] = 4.00

        elif "claude" in model:
            # Generic Claude default – approximate Sonnet pricing
            cost_per_1m_input[prov.name] = 3.00
            cost_per_1m_output[prov.name] = 15.00

        # --- Google Gemini ---
        elif "gemini-2.0-flash" in model:
            # Gemini 2.0 Flash: $0.10 in / $0.40 out
            cost_per_1m_input[prov.name] = 0.10
            cost_per_1m_output[prov.name] = 0.40

        elif "gemini-1.5-pro" in model:
            # Gemini 1.5 Pro: ≈ $1.25 in / $5 out
            cost_per_1m_input[prov.name] = 1.25
            cost_per_1m_output[prov.name] = 5.00

        elif "gemini" in model:
            # Generic Gemini default – approximate 1.5 Pro pricing
            cost_per_1m_input[prov.name] = 1.25
            cost_per_1m_output[prov.name] = 5.00

        # --- Fallback for any other model ---
        else:
            # Conservative generic defaults
            cost_per_1m_input[prov.name] = 1.00
            cost_per_1m_output[prov.name] = 2.00

    # ------------------------------
    # Total cost across providers
    # ------------------------------
    total_cost = 0.0
    breakdown_data = []

    for prov in providers:
        provider_input_tokens = total_input_tokens / len(providers)
        provider_output_tokens = total_output_tokens / len(providers)

        rate_in = cost_per_1m_input[prov.name]
        rate_out = cost_per_1m_output[prov.name]

        provider_cost_in = (provider_input_tokens / 1_000_000) * rate_in
        provider_cost_out = (provider_output_tokens / 1_000_000) * rate_out
        provider_cost = provider_cost_in + provider_cost_out

        total_cost += provider_cost

        breakdown_data.append({
            'Provider': prov.name,
            'Model': prov.model,
            'Input tokens (est.)': f"{int(provider_input_tokens):,}",
            'Output tokens (est.)': f"{int(provider_output_tokens):,}",
            'Rate in ($/1M)': f"${rate_in:.2f}",
            'Rate out ($/1M)': f"${rate_out:.2f}",
            'Cost (est.)': f"${provider_cost:.2f}",
        })

    # ------------------------------
    # Display estimate
    # ------------------------------
    col_cost1, col_cost2 = st.columns(2)

    with col_cost1:
        st.metric("Total API Calls", f"{total_calls_batched:,}")
        st.metric("Estimated Tokens (in + out)", f"{int(total_tokens):,}")

    with col_cost2:
        st.metric("Estimated Cost", f"${total_cost:.2f}")

    # Detailed breakdown
    with st.expander("📊 Detailed Cost Breakdown"):
        st.table(pd.DataFrame(breakdown_data))

        st.caption(
            f"Breakdown: {num_docs} docs × {len(providers)} providers × "
            f"{num_versions} versions × {num_runs} runs"
        )
        if use_batching and batch_size > 1:
            st.caption(f"Batching: {num_batches} batches of up to {batch_size} docs each")

else:
    st.info("Configure providers in Section 3 to see cost estimate")

# ------------------------------
# ONE SHARED BUTTON: RUN EVERYTHING
# ------------------------------
st.write("---")
st.header("Step 5: Run Analysis")
if st.button("🚀 Run Analysis"):
    if 'providers' not in locals() or not providers:
        st.error("No providers configured. Please configure providers before running analysis.")
    else:
        # Simple sentiment prompt with temporal guardrails (NO channels)
        task_instruction = (
        f"{user_prompt}\n\n"
        f"AS-OF CUTOFF:\n"
        f"- As-of date: {{as_of_date}}\n"
        f"- If the document/publication date is known and is AFTER {{as_of_date}}, output exactly:\n"
        f"  ABORT: document post-dates as_of_date\n"
        f"  and stop.\n\n"
        f"STRICT ANTI-LOOK-AHEAD RULES:\n"
        f"- Use ONLY information explicitly present in the provided text.\n"
        f"- Do NOT use knowledge of outcomes or events that occurred after the text's timestamp.\n"
        f"- Ignore later-added editor notes, revisions, or update banners dated after {{as_of_date}}.\n"
        f"- Do NOT infer realized performance from forward-looking statements (treat guidance as expectations, not outcomes).\n"
        f"- No external data, no web, no model memory: if it's not in the text, you cannot use it.\n\n"
        f"WHAT TO EXTRACT:\n"
        f"- Identify the document/publication date if present; otherwise return 'unknown'.\n"
        f"- Summarize the prevailing sentiment strictly as expressed AT THE TIME of writing.\n"
        f"- Weight forward-looking language conservatively; do not convert it into ex-post success/failure.\n"
        f"- Cite evidence by quoting short snippets from the text; include line/section references if available.\n\n"
        f"SCORING:\n"
        f"- Provide a numeric sentiment score on a 0–1 scale (decimals allowed).\n"
        f"  * 0.0 = very negative, 0.5 = neutral, 1.0 = very positive.\n"
        f"- Calibrate narrowly: avoid extreme values unless the language is unequivocal.\n"
        f"- Provide a separate confidence score (0–1) reflecting clarity/amount of evidence in the text (NOT your correctness about the future).\n\n"
        f"OUTPUT FORMAT (exactly this structure):\n"
        f"- First line MUST be exactly: Signal: 0.XX  (where 0.XX is your sentiment score between 0 and 1)\n"
        f"- Then a compact JSON object on the next lines with these fields only:\n"
        f"  {{\n"
        f"    \"as_of_date\": \"{{as_of_date}}\",\n"
        f"    \"document_date\": \"<YYYY-MM-DD or 'unknown'>\",\n"
        f"    \"Score\": <float 0-1>,\n"
        f"    \"confidence\": <float 0-1>,\n"
        f"    \"stance\": \"negative|slightly_negative|neutral|slightly_positive|positive\",\n"
        f"    \"evidence\": [\"<short quote 1>\", \"<short quote 2>\", \"<short quote 3>\"] ,\n"
        f"    \"forward_looking_flags\": <integer count>,\n"
        f"    \"reasoning\": \"<<=120 words; concise, text-grounded; no future knowledge>\"\n"
        f"  }}\n\n"
        f"GUARDRAILS & EDGE CASES:\n"
        f"- If sentiment is mixed/ambiguous, prefer scores near 0.5 and explain briefly.\n"
        f"- If the text references specific future dates (e.g., 'next quarter'), mark them as forward-looking but do not score based on realized results.\n"
        f"- If critical fields are missing (date, subject), proceed with best-effort analysis but note the limitation in 'reasoning'.\n"
        f"- If the text contains rumors or conditional language ('if', 'may', 'could'), treat as low-weight evidence.\n"
        f"- Never include information not directly supported by the provided text.\n"
    )

        
        system_instructions = (
            "You are an expert analyst conducting historical document analysis. "
            "You must NEVER use information that would only be available after the document's timestamp. "
            "Treat each document as if you are living in that moment in time. "
            "Focus on facts, data, and sentiment as expressed AT THAT TIME."
        )
        
        # Store in session state
        st.session_state['generated_task'] = task_instruction
        st.session_state['generated_system'] = system_instructions
        

    # A3: Review & Edit Final Prompt

        # Use generated prompt if available
        default_task = st.session_state.get('generated_task', 
            f"{user_prompt}\n\nProvide a sentiment score [0-1] and explain. Include 'Signal: <number>'"
        )
        default_system = st.session_state.get('generated_system',
            "You are an analyst. Focus only on information present in the document at the time of writing."
        )

        task_instruction = default_task

        system_instructions = default_system
        df_raw = run_masking_and_summaries(df_raw, providers, request_delay if 'request_delay' in locals() else 1.0)
        run_bias_detection_experiment(df_raw, providers, request_delay if 'request_delay' in locals() else 1.0)
        run_sensitivity_grid(df_raw, providers, request_delay if 'request_delay' in locals() else 1.0)

# ------------------------------
# Stage 2: Subscore-Based Prediction (Optional)
# ------------------------------

# ------------------------------
# SECTION 8: Summary Dashboard
# ------------------------------
st.header("Summary Dashboard")
st.write("Consolidated view of all analyses completed in this session.")

# Check what analyses have been completed
has_bias_detection = 'experiment_results' in st.session_state
has_sensitivity = 'sensitivity_summary' in st.session_state
has_stage1 = 'two_stage_stage1_df' in st.session_state
has_stage2 = 'two_stage_stage2_df' in st.session_state

if not any([has_bias_detection, has_sensitivity, has_stage1, has_stage2]):
    st.info("No analyses completed yet.")
else:
    # Summary of Bias Detection (Section 5)
    if has_bias_detection:
        st.subheader("📊 Bias Detection Results (Section 5)")
        df_exp = st.session_state['experiment_results']
        
        # Overall statistics by version
        exp_summary = df_exp.groupby('corpus_version')['sentiment'].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('count', 'count')
        ]).reset_index()
        
        st.write("**Mean Sentiment by Corpus Version:**")
        st.dataframe(exp_summary)
        
        # Statistical comparison if available
        if len(df_exp['corpus_version'].unique()) > 1:
            stats = compare_versions_statistically(df_exp)
            interpretation = interpret_bias_detection(stats)
            
            st.write("**Bias Detection Result:**")
            st.markdown(interpretation)
    
    # Summary of Sensitivity Analysis
    if has_sensitivity:
        st.subheader("📊 Sensitivity Analysis Results")
        df_sens_sum = st.session_state['sensitivity_summary']
        
        # Show summary by version if multiple versions tested
        if 'corpus_version' in df_sens_sum.columns:
            st.write("**Sensitivity by Version:**")
            for version in df_sens_sum['corpus_version'].unique():
                ver_data = df_sens_sum[df_sens_sum['corpus_version'] == version]
                st.write(f"*{version.upper()}:*")
                st.dataframe(ver_data[['base_provider', 'temperature', 'seed', 'mean', 'std']])
        else:
            st.write("**Sensitivity Summary:**")
            st.dataframe(df_sens_sum[['base_provider', 'temperature', 'seed', 'mean', 'std']])
    
    # Summary of Two-Stage Analysis
    if has_stage1 or has_stage2:
        st.subheader("📊 Two-Stage Channel-Based Results (Section 7)")
        
        # Stage 1 summary
        if has_stage1:
            df_s1 = st.session_state['two_stage_stage1_df']
            st.write("**Stage 1: Channel Scores Summary**")
            
            # Average score by channel
            channel_avg = df_s1.groupby('channel')['score'].agg([
                ('mean', 'mean'),
                ('std', 'std'),
                ('count', 'count')
            ]).reset_index()
            st.dataframe(channel_avg.sort_values('mean', ascending=False))
        
        # Stage 2 summary
        if has_stage2:
            df_s2 = st.session_state['two_stage_stage2_df']
            st.write("**Stage 2: Overall Sentiment Predictions**")
            
            # By document and provider
            s2_summary = df_s2.groupby(['doc_id', 'provider'])['overall_score'].agg([
                ('mean', 'mean'),
                ('std', 'std'),
                ('count', 'count')
            ]).reset_index()
            st.dataframe(s2_summary)
    
    # Comparison across all methods (if multiple available)
    st.subheader("🔍 Cross-Method Comparison")
    
    comparison_data = []
    
    # Direct sentiment (Section 5 - original version)
    if has_bias_detection:
        df_exp = st.session_state['experiment_results']
        if 'original' in df_exp['corpus_version'].unique():
            orig_data = df_exp[df_exp['corpus_version'] == 'original']
            comparison_data.append({
                'Method': 'Direct Sentiment (Original)',
                'Mean': orig_data['sentiment'].mean(),
                'Std': orig_data['sentiment'].std(),
                'N': len(orig_data)
            })
        
        if 'masked' in df_exp['corpus_version'].unique():
            masked_data = df_exp[df_exp['corpus_version'] == 'masked']
            comparison_data.append({
                'Method': 'Direct Sentiment (Masked)',
                'Mean': masked_data['sentiment'].mean(),
                'Std': masked_data['sentiment'].std(),
                'N': len(masked_data)
            })
        
        if 'summary' in df_exp['corpus_version'].unique():
            summary_data = df_exp[df_exp['corpus_version'] == 'summary']
            comparison_data.append({
                'Method': 'Direct Sentiment (Summary)',
                'Mean': summary_data['sentiment'].mean(),
                'Std': summary_data['sentiment'].std(),
                'N': len(summary_data)
            })
    
    # Channel-based prediction (Section 7)
    if has_stage2:
        df_s2 = st.session_state['two_stage_stage2_df']
        comparison_data.append({
            'Method': 'Channel-Based (Two-Stage)',
            'Mean': df_s2['overall_score'].mean(),
            'Std': df_s2['overall_score'].std(),
            'N': len(df_s2)
        })
    
    if len(comparison_data) > 1:
        st.write("**All Methods Compared:**")
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison)
        
        # Statistical testing: Pairwise t-tests between methods
        st.write("**Statistical Testing (Pairwise T-Tests):**")
        st.write("Testing if methods produce significantly different sentiment scores.")
        
        # Get raw data for each method
        method_data = {}
        
        if has_bias_detection:
            df_exp = st.session_state['experiment_results']
            for version in df_exp['corpus_version'].unique():
                method_name = f"Direct Sentiment ({version.title()})"
                method_data[method_name] = df_exp[df_exp['corpus_version'] == version]['sentiment'].dropna().values
        
        if has_stage2:
            df_s2 = st.session_state['two_stage_stage2_df']
            method_data['Channel-Based (Two-Stage)'] = df_s2['overall_score'].dropna().values
        
        # Perform pairwise t-tests
        test_results = []
        method_names = list(method_data.keys())
        
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                method1 = method_names[i]
                method2 = method_names[j]
                data1 = method_data[method1]
                data2 = method_data[method2]
                
                if len(data1) > 1 and len(data2) > 1:
                    # Independent samples t-test
                    t_stat, p_value = scipy_stats.ttest_ind(data1, data2)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        ((len(data1)-1)*np.std(data1, ddof=1)**2 + (len(data2)-1)*np.std(data2, ddof=1)**2) / 
                        (len(data1)+len(data2)-2)
                    )
                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
                    
                    # Difference
                    mean_diff = np.mean(data1) - np.mean(data2)
                    
                    test_results.append({
                        'Comparison': f"{method1}\nvs\n{method2}",
                        'Mean Diff': f"{mean_diff:.3f}",
                        'p-value': f"{p_value:.4f}",
                        'Significant': "Yes ✓" if p_value < 0.05 else "No",
                        "Cohen's d": f"{cohens_d:.2f}",
                        'Effect Size': (
                            "Large" if abs(cohens_d) >= 0.8 else
                            "Medium" if abs(cohens_d) >= 0.5 else
                            "Small" if abs(cohens_d) >= 0.2 else
                            "Negligible"
                        )
                    })
        
        if test_results:
            df_tests = pd.DataFrame(test_results)
            st.dataframe(df_tests)
            
            # Interpretation
            sig_comparisons = [r for r in test_results if "Yes" in r['Significant']]
            if sig_comparisons:
                st.warning(f"⚠️ Found {len(sig_comparisons)} significant difference(s) between methods (p < 0.05)")
            else:
                st.success("✓ No significant differences found between methods")
        
        # Visualization
        fig_comp = go.Figure()
        for _, row in df_comparison.iterrows():
            fig_comp.add_trace(go.Bar(
                name=row['Method'],
                x=[row['Method']],
                y=[row['Mean']],
                error_y=dict(type='data', array=[row['Std']]),
                text=f"n={row['N']}",
                textposition='outside'
            ))
        
        fig_comp.update_layout(
            title="Mean Sentiment Across All Methods",
            xaxis_title="Method",
            yaxis_title="Mean Sentiment [0-1]",
            yaxis=dict(range=[-0.05, 1.05]),
            showlegend=False
        )
        st.plotly_chart(fig_comp)
    
    # AI-Generated Report
    st.subheader("📝 AI-Generated Summary Report")
    st.write("Generate a comprehensive analysis report using an LLM based on all your results.")
    
    # Check if OpenAI is available for report generation
    report_providers = [p for p in providers if 'openai' in p.provider.lower() or 'gpt' in p.model.lower()] if 'providers' in locals() else []
    
    if not report_providers:
        st.warning("⚠️ OpenAI provider required for report generation. Please enable OpenAI in Section 3.")
    else:
        report_provider_idx = st.selectbox(
            "Select LLM for report generation",
            range(len(report_providers)),
            format_func=lambda i: report_providers[i].name if i < len(report_providers) else "",
            key="report_provider_select"
        )
        
        with st.spinner("Generating comprehensive analysis report..."):
            try:
                # Collect all summary data
                report_data = {
                    'has_bias_detection': has_bias_detection,
                    'has_sensitivity': has_sensitivity,
                    'has_stage2': has_stage2,
                    'comparison_table': df_comparison.to_string() if len(comparison_data) > 1 else "N/A",
                    'statistical_tests': df_tests.to_string() if 'df_tests' in locals() and len(test_results) > 0 else "N/A"
                }
                
                if has_bias_detection:
                    df_exp = st.session_state['experiment_results']
                    report_data['bias_summary'] = exp_summary.to_string()
                    if len(df_exp['corpus_version'].unique()) > 1:
                        stats = compare_versions_statistically(df_exp)
                        report_data['bias_interpretation'] = interpret_bias_detection(stats)
                
                if has_stage2:
                    df_s2 = st.session_state['two_stage_stage2_df']
                    report_data['stage2_summary'] = s2_summary.to_string()
                
                # Build report prompt
                report_prompt = f"""
You are an expert research analyst. Write a concise, prose-only report about the look-ahead/bias tests below.

STYLE & SCOPE
- Prose paragraphs only (no tables, no bullet lists).
- Clear, neutral, and cautious wording (e.g., “no indication observed” rather than “no bias”).
- Focus on essentials: what we did, key results, what we can/cannot infer about bias, and next steps.

======================
LOOK-AHEAD / BIAS ANALYSIS REPORT
[AI-GENERATED REPORT]
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
======================

EXECUTIVE SUMMARY
Provide a brief 2–3 paragraph overview covering: the dataset and transformations evaluated (original vs. masked vs. masked-summary), the main empirical signal from the tests, whether differences were observed, and the high-level implication for potential bias or leakage.

METHODS (WHAT WE DID)
Describe, in 1–2 paragraphs, the evaluation setup: uploaded corpus type; masking choices (the following masking choices were used: {', '.join(masking_categories)}); creation of masked and ~50-word masked-summary variants; the user-defined classification task; and any parameter sweeps (e.g., temperature, seed) used to assess robustness.

RESULTS (WHAT WE FOUND)
{ "Summarize pairwise comparisons across original, masked, and masked-summary views and state whether differences were statistically significant (t-tests) and practically meaningful." if has_bias_detection else "Bias detection comparisons were not performed in this run." }
{report_data.get('bias_interpretation', '')}
In 1–2 paragraphs, synthesize the direction and magnitude of differences, stability under parameter variation, and any notable document- or channel-level patterns if observed.

INTERPRETATION ABOUT BIAS (WHAT WE CAN SAY)
State, in careful terms, what the evidence supports about potential bias or temporal leakage. Use qualified language (e.g., “the tests provide evidence consistent with…”, “we did not observe indications of…”). Explain why masked or masked-summary variants may provide safer baselines when divergences appear.

BOUNDARIES OF INFERENCE (WHAT WE CANNOT SAY)
Clarify limits of the analysis (e.g., model- and prompt-specificity, sample size, corpus domain, absence of causal identification, sensitivity to hyperparameters). Note that non-significant results do not prove absence of bias and significant results do not identify the exact causal source.

ROBUSTNESS & SENSITIVITY
{ "Briefly describe how outputs varied (or not) across temperature/seed and whether conclusions held across settings." if has_sensitivity else "Sensitivity analysis across hyperparameters was not performed in this run." }

NEXT STEPS FOR THE RESEARCHER
Offer concrete, prioritized actions (e.g., proceed with masked variant; refine masking rules; standardize prompting; expand sample; pre-register thresholds; validate on an out-of-time split; consider complementary diagnostics).

CONCLUSION
Provide a concise closing paragraph that restates the central finding and its implication for using the current pipeline in downstream analysis.

======================
DISCLAIMER: This report was generated by an AI model ({report_providers[report_provider_idx].model}) and should be reviewed by human researchers before use in publications or decision-making.
======================

Generate the report following this structure and style.
"""

                
                # Create provider instance
                report_cfg = report_providers[report_provider_idx]
                if report_cfg.provider == "openai":
                    report_prov = OpenAIProvider(report_cfg)
                elif report_cfg.provider == "litellm":
                    report_prov = LiteLLMProvider(report_cfg)
                else:
                    report_prov = MockProvider(report_cfg)
                
                # Generate report
                report_output = report_prov.infer({
                    'system': "You are an expert research analyst specializing in bias detection and statistical analysis.",
                    'user': report_prompt
                })
                
                # Store report text
                st.session_state['ai_report'] = report_output
                
                # Generate PDF with all tables and figures
                pdf_buffer = create_pdf_report(
                    report_output,
                    has_bias_detection,
                    has_sensitivity,
                    has_stage2
                )
                st.session_state['ai_report_pdf'] = pdf_buffer.getvalue()
                
                st.success("✓ Report generated (text and PDF)!")
                
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        # Show report if available
        if 'ai_report' in st.session_state:
            st.markdown("---")
            st.markdown("### 📄 Generated Analysis Report")
            
            # Display in a nice formatted box
            st.markdown(st.session_state['ai_report'])
            
            # Download buttons
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                # Text version
                report_bytes = st.session_state['ai_report'].encode('utf-8')
                st.download_button(
                    label="⬇️ Download as Text (.txt)",
                    data=report_bytes,
                    file_name=f"bias_analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_report_txt"
                )
            
            with col_dl2:
                # PDF version (if available)
                if 'ai_report_pdf' in st.session_state:
                    st.download_button(
                        label="📥 Download as PDF (.pdf)",
                        data=st.session_state['ai_report_pdf'],
                        file_name=f"bias_analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key="download_report_pdf"
                    )

# ------------------------------
# SECTION 9: Exports & Downloads
# ------------------------------
st.header("Exports & Downloads")

# Excel Export with all results
st.subheader("📊 Comprehensive Excel Export")
st.write("Download all results in one Excel file with separate sheets for each analysis.")

# Check what data is available
available_sheets = []
if 'experiment_results' in st.session_state:
    available_sheets.append("Bias Detection Results")
if 'sensitivity_results' in st.session_state:
    available_sheets.append("Sensitivity Analysis")
if 'two_stage_stage1_df' in st.session_state:
    available_sheets.append("Stage 1 Channel Scores")
if 'two_stage_stage2_df' in st.session_state:
    available_sheets.append("Stage 2 Overall Predictions")
if 'text_masked' in st.session_state:
    available_sheets.append("Masked Data")
if 'text_summary' in st.session_state:
    available_sheets.append("Summary Data")

if available_sheets:
    st.write(f"**Available data:** {', '.join(available_sheets)}")
    
    if st.button("📥 Generate Excel Export", key="generate_excel"):
        with st.spinner("Creating Excel file with all results..."):
            try:
                excel_data = create_excel_export(df_raw=df_raw if 'df_raw' in locals() else None)
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                
                st.download_button(
                    label="⬇️ Download Excel File",
                    data=excel_data,
                    file_name=f"lookahead_bias_results_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
                st.success("✓ Excel file ready for download!")
                
                # Show sheet contents summary
                with st.expander("📋 Excel File Contents"):
                    if 'experiment_results' in st.session_state:
                        st.write(f"- **Bias_Detection**: {len(st.session_state['experiment_results'])} rows")
                    if 'sensitivity_results' in st.session_state:
                        st.write(f"- **Sensitivity_Analysis**: {len(st.session_state['sensitivity_results'])} rows")
                    if 'sensitivity_summary' in st.session_state:
                        st.write(f"- **Sensitivity_Summary**: {len(st.session_state['sensitivity_summary'])} rows")
                    if 'two_stage_stage1_df' in st.session_state:
                        st.write(f"- **Stage1_Channel_Scores**: {len(st.session_state['two_stage_stage1_df'])} rows")
                    if 'two_stage_stage2_df' in st.session_state:
                        st.write(f"- **Stage2_Overall_Scores**: {len(st.session_state['two_stage_stage2_df'])} rows")
                    if 'text_masked' in st.session_state:
                        st.write(f"- **Masked_Data**: {len(st.session_state['text_masked'])} rows (original + masked texts)")
                    if 'text_summary' in st.session_state:
                        st.write(f"- **Summary_Data**: {len(st.session_state['text_summary'])} rows (original + summaries)")
                    st.write("- **Configuration**: Metadata and settings")
                
            except Exception as e:
                st.error(f"Excel export failed: {e}")
                import traceback
                st.code(traceback.format_exc())
else:
    st.info("No results available yet. ")

st.write("---")  # Simple divider for older Streamlit versions

