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
    "gpt-5.1": (1.25, 10.00),
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
        "gpt-5.1",
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
    "Time": (
        "### ⚠️ Temporal Leakage (Time Bias)\n\n"
        "**The Problem:** LLMs are trained on historical data. If a document mentions a specific date (e.g., 'September 2008'), "
        "the model 'remembers' what happened next (the crash) instead of analyzing the text objectively.\n\n"
        "**Risk:** The model performs artificially well on backtests but fails in production because it cannot know the future in real-time.\n\n"
        "**Recommendation:**\n"
        "* **Data Fix:** Replace specific dates with relative markers (e.g., replace '2020' with 'Year X').\n"
        "* **Prompt Fix:** Instruct the model to ignore year references and assume a 'timeless vacuum'."
    ),
    "Organizations": (
        "### ⚠️ Reputation Bias\n\n"
        "**The Problem:** The model knows the long-term fate of famous companies. It may assign a positive score to 'Apple' "
        "simply because it knows Apple became successful, ignoring specific risks mentioned in the text.\n\n"
        "**Risk:** The model is trading on brand recognition, not the content of the filing or news article.\n\n"
        "**Recommendation:**\n"
        "* **Data Fix:** Anonymize entities. Replace names with generic tokens like `[COMPANY_A]`.\n"
        "* **Prompt Fix:** Tell the model to treat all companies as unknown, hypothetical entities."
    ),
    "Numbers": (
        "### ⚠️ Magnitude Bias\n\n"
        "**The Problem:** LLMs often rely on heuristics, assuming 'large numbers' (Billions) are automatically 'Good' or 'Bad' "
        "depending on context, regardless of whether the number represents profit, debt, or a fine.\n\n"
        "**Risk:** The model ignores the semantic sentiment (the words) and just looks for the biggest number on the page.\n\n"
        "**Recommendation:**\n"
        "* **Data Fix:** Normalize financial figures (e.g., convert raw numbers to `% change`) or mask them with `[NUM]`.\n"
        "* **Prompt Fix:** Instruct the model to focus on qualitative descriptors (e.g., 'disappointing', 'strong') rather than digits."
    ),
    "Locations": (
        "### ⚠️ Geographic Bias\n\n"
        "**The Problem:** LLMs contain sociological biases about specific regions. A startup in 'Silicon Valley' might be scored "
        "more positively than one in a region associated with economic instability, solely based on the location name.\n\n"
        "**Risk:** You might miss opportunities in emerging markets or undervalue assets in specific regions due to prejudice.\n\n"
        "**Recommendation:**\n"
        "* **Data Fix:** Mask locations with `[LOCATION]`.\n"
        "* **Prompt Fix:** Explicitly instruct the model to ignore geographic context."
    ),
    "Person Names": (
        "### ⚠️ Authority Bias (The 'Halo Effect')\n\n"
        "**The Problem:** If a text quotes a celebrity CEO or famous investor (e.g., Buffett, Musk), the model may assign a "
        "positive score simply because it associates that person with success, ignoring what they actually said.\n\n"
        "**Risk:** The model acts as a fanboy rather than an objective analyst.\n\n"
        "**Recommendation:**\n"
        "* **Data Fix:** Replace specific names with titles (e.g., change 'Mark Zuckerberg' to 'The CEO').\n"
        "* **Prompt Fix:** Instruct the model to evaluate the statement objectively, regardless of the speaker's fame."
    ),
    "Gender": (
        "### ⛔ Social Bias (Critical Fairness Issue)\n\n"
        "**The Problem:** Research shows LLMs can assign lower competence scores to text containing female pronouns ('she/her') "
        "compared to male pronouns in business contexts, or stereotype certain industries.\n\n"
        "**Risk:** Ethical & Legal risk. Your model is making decisions based on gender rather than merit.\n\n"
        "**Recommendation:**\n"
        "* **Data Fix:** Neutralize the text. Replace 'He/She' with 'They' or 'The Executive'.\n"
        "* **Prompt Fix:** Maintain strict gender neutrality. Do not allow the gender of the subject to influence the sentiment."
    ),
    "Products": (
        "### ⚠️ Outcome Bias (Product Bias)\n\n"
        "**The Problem:** The model knows which products succeeded (iPhone) and which failed (Zune). It cannot objectively "
        "evaluate a historical review because it has access to future knowledge of the product's fate.\n\n"
        "**Risk:** Hindsight bias. The model cannot simulate the uncertainty of the moment the text was written.\n\n"
        "**Recommendation:**\n"
        "* **Data Fix:** Replace product names with 'The Product' or 'Device A'.\n"
        "* **Prompt Fix:** Evaluate the product solely on the features described, ignoring external knowledge of market success."
    )
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