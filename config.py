# -*- coding: utf-8 -*-
"""
Configuration file for Look-Ahead Bias Lab.
Contains global constants, regex patterns, model lists, and threshold settings.
"""

APP_VERSION = "0.982"
OUTPUT_DIR = "outputs"

# Random Seed for reproducibility
RANDOM_DEFAULT_SEED = 7

# Batch Processing Settings
BATCH_SIZE_DEFAULT = 100          
USE_BATCHING_DEFAULT = True

# SAFETY RAILS
MIN_DOCS_THRESHOLD = 5       # Minimum docs required to run stats
MAX_SUBSET_SIZE = 50         # Maximum docs per category (randomly sampled if exceeded)

# --- PRICING TABLE (Est. $ per 1M tokens as of late 2024/2025) ---
# Format: "Model Name": (Input Price, Output Price)
PRICING_TABLE = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    
    # Anthropic
    "claude-3-5-sonnet-20240620": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (1.00, 5.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    
    # Gemini
    "gemini/gemini-1.5-pro": (1.25, 5.00),
    "gemini/gemini-1.5-flash": (0.075, 0.30),
    
    # Fallback
    "default": (1.00, 2.00)
}

# --- AVAILABLE MODELS ---
MODEL_OPTIONS = {
    "OpenAI": [
        "gpt-4o", 
        "gpt-4o-mini", 
        "gpt-4-turbo", 
        "gpt-3.5-turbo"
    ],
    "Claude": [
        "claude-3-5-sonnet-20240620", 
        "claude-3-5-haiku-20241022", 
        "claude-3-opus-20240229"
    ],
    "Gemini": [
        "gemini/gemini-1.5-pro", 
        "gemini/gemini-1.5-flash"
    ],
    "Mock": [
        "mock-model-v1"
    ]
}

# Detailed instructions for the LLM to ensure aggressive masking
MASKING_DEFINITIONS = {
    "Time": "Dates, years (e.g., 2023), months, relative time (yesterday).",
    "Organizations": "Companies, agencies, institutions (Apple, FBI).",
    "Numbers": "Money, percentages, cardinals, quantities.",
    "Locations": "Countries, cities, states, regions.",
    "Person Names": "Specific people names.",
    "Gender": "He/She/Him/Her pronouns and gendered titles.",
    "Products": "Commercial products/services."
}

# List of keys for UI loops
ALL_MASK_CATEGORIES = list(MASKING_DEFINITIONS.keys())

# Diagnostic Feedback Messages
BIAS_FEEDBACK_MESSAGES = {
    "Time": "Potential **Temporal Leakage**. The model relies on dates, possibly accessing future knowledge (e.g. 2008 crash).",
    "Organizations": "Potential **Reputation Bias**. The model relies on company names, possibly using external knowledge of their success.",
    "Numbers": "**Magnitude Bias**. Financial figures are influencing the score excessively.",
    "Locations": "**Geographic Bias**. The model treats regions differently.",
    "Person Names": "**Authority Bias**. Specific individuals are swaying the analysis.",
    "Gender": "CRITICAL **Social Bias**. Gendered pronouns are altering the sentiment score.",
    "Products": "**Product Bias**. Specific brands are influencing the result."
}

# NLP & Regex Configuration
ENTITY_CONFIG = {
    "Time": {
        "spacy": ["DATE", "TIME"],
        "regex": r'\b(19|20)\d{2}\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b(yesterday|today|tomorrow|next|last|fiscal|quarter|q[1-4])\b'
    },
    "Organizations": {
        "spacy": ["ORG"],
        "regex": r'\b(Inc|Corp|Ltd|LLC|Group|Bank|Company|Systems|Technologies)\b'
    },
    "Numbers": {
        "spacy": ["MONEY", "PERCENT", "CARDINAL", "QUANTITY"],
        "regex": r'\b\d+(\.\d+)?(%|m|b|k)?\b|\$\d+'
    },
    "Locations": {
        "spacy": ["GPE", "LOC"],
        "regex": None
    },
    "Person Names": {
        "spacy": ["PERSON"],
        "regex": None
    },
    "Gender": {
        "spacy": [], 
        "regex": r'\b(he|she|his|her|him|man|woman|mr|ms|mrs|gentleman|lady|chairman|spokesman|spokeswoman)\b'
    },
    "Products": {
        "spacy": ["PRODUCT"], 
        "regex": None
    }
}