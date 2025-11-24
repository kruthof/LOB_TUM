# Look-Ahead Bias Lab (v0.982)

Look-Ahead Bias Lab is an interactive **Streamlit** application for
detecting **look-ahead bias** and related information leakage in
**binary classification** tasks using large language models (LLMs).

It tests whether your prompt relies on information that *should not*
influence a model's prediction---such as dates, organization names,
locations, numbers, gendered terms, product names, or specific people.

The lab works by: 1. Scanning your corpus and identifying documents
containing these categories. 2. Creating masked variants of each
document (removing only one category at a time). 3. Running your LLM on
both the **Original** and **Masked** versions. 4. Measuring whether the
scores significantly change using statistical tests. 5. Reporting which
categories your model depends on and recommending prompt fixes.

------------------------------------------------------------------------

## Features

-   **Entity-driven document scanning** via spaCy + regex\
-   **Deterministic masking engine** for seven bias categories:\
    Time, Organizations, Numbers, Locations, Person Names, Gender,
    Products\
-   **Cost estimator** for input/output tokens per model\
-   **Self-healing inference engine** that retries missing documents\
-   **Paired statistical analysis** using two-pass t-tests\
-   **Support for multiple LLM providers**:
    -   OpenAI (official SDK)
    -   Anthropic + Gemini (via LiteLLM)
    -   Mock model for offline experiments\
-   **Full Excel export** with:
    -   Stats sheet\
    -   Raw scores\
    -   Per-category merged data

------------------------------------------------------------------------

## Project Structure

-   **lob.py** -- Main Streamlit application\
    Handles UI, masking, inference, statistical testing, and Excel
    export.

-   **config.py** -- Central configuration\
    Defines model lists, pricing, regex patterns, masking rules,
    thresholds, and category definitions.

------------------------------------------------------------------------

## Installation

### 1. Clone the repository

``` bash
git clone <your-repo-url>.git
cd LookAheadBiasLab
```

### 2. (Optional) Create a virtual environment

``` bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# or
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

Install via your own requirements file or manually:

``` txt
streamlit
pandas
numpy
scipy
spacy
openai
litellm
plotly
matplotlib
openpyxl
pyarrow
```

### 4. spaCy models

The application will automatically download: - `en_core_web_lg`
(preferred) - fallback `en_core_web_sm`

### 5. Run the app

``` bash
streamlit run lob.py
```

------------------------------------------------------------------------

## Usage Guide

### Step 1 --- Upload Dataset

Upload a CSV or Parquet file.\
Select the column containing your text.\
The app constructs a normalized dataframe of: - `doc_id` - `timestamp`
(placeholder) - `text`

### Step 2 --- Enter Prompt

Your prompt **must** return a floating-point score in **\[0.0, 1.0\]**.

Examples: - "What is the sentiment score between 0 and 1?" - "Rate the
hawkishness from 0 (dovish) to 1 (strongly hawkish)."

### Step 3 --- Configure LLM Provider

Select provider (OpenAI, Claude, Gemini, Mock), model, and API key.\
The app sets environment variables automatically: - `OPENAI_API_KEY` -
`ANTHROPIC_API_KEY` - `GOOGLE_API_KEY`

### Step 4 --- Scan & Estimate Cost

The system will: 1. Detect entities and regex matches per category\
2. Build subsets, capped at **MAX_SUBSET_SIZE**\
3. Estimate LLM cost using average length, token pricing, and two passes

You will see: - Per-category table: Found / Using / Status / Est. Cost\
- Total estimated cost\
- Indicator for subset capping

### Step 5 --- Run Analysis

For each category: - Create masked version of text\
- Run LLM on Original\
- Run LLM on Masked\
- Retry any missing docs\
- Merge into final results dataframe

------------------------------------------------------------------------

## Statistical Analysis

For each category: 1. Align Original vs Masked scores per doc\
2. Compute: - Mean(Original), Mean(Masked) - Paired t-test:
`scipy.stats.ttest_rel()`\
3. Interpretation: - **p \< 0.05 → Bias Detected** - otherwise → No
statistically significant effect

If bias is detected: - Category appears in a red alert\
- Human-readable explanation is shown\
- A **Debiased Prompt** is generated automatically

------------------------------------------------------------------------

## Masking Categories

Defined in `config.py`:

### Time

Dates, years, quarters, months, fiscal terms.

### Organizations

Companies and ORG-type entities.

### Numbers

Money, percentages, quantitative figures.

### Locations

Countries, regions, cities, GPE/LOC entities.

### Person Names

Proper PERSON entities.

### Gender

Pronouns and gendered titles.

### Products

Commercial products / PRODUCT entities.

Each category uses **spaCy NER** plus optional **regex** rules.

------------------------------------------------------------------------

## Providers & Inference Layer

### OpenAIProvider

-   Uses official SDK\
-   Retries on failure\
-   Deterministic parameters (temperature=0, seeded)

### LiteLLMProvider

-   Supports Anthropic + Gemini\
-   Automatic key routing via LiteLLM

### MockProvider

-   No external calls\
-   Returns random scores

------------------------------------------------------------------------

## Cost Estimation Details

Token estimation logic:

    input_tokens  ≈ avg_chars / 4 + 150
    output_tokens ≈ 50
    passes        = 2 (Original + Masked)

    total_tokens  = (input_tokens + output_tokens) * docs * passes
    cost          = total_tokens / 1e6 * price_per_million

Model pricing is fully configurable in `PRICING_TABLE`.

------------------------------------------------------------------------

## Export

The app exports a full Excel workbook:

### Sheet: `Stats`

-   Category\
-   N\
-   Means (Original vs Masked)\
-   p-value\
-   Bias flag

### Sheet: `Raw`

-   doc_id\
-   sentiment score\
-   version\
-   category_group

### Sheet: `{Category}_Data`

-   Text + masked text\
-   Merged Original + Masked scores

------------------------------------------------------------------------

## Security Notice

The app: - Does **not** persist API keys\
- Sends keys only for on-demand API calls\
- Should be run **locally** for sensitive datasets

------------------------------------------------------------------------

## Extending the System

Add new categories by extending: - `MASKING_DEFINITIONS` -
`ENTITY_CONFIG` - `BIAS_FEEDBACK_MESSAGES` - `ALL_MASK_CATEGORIES`

Add new providers by subclassing `ModelProvider` in `lob.py`.

------------------------------------------------------------------------

## Summary

The Look-Ahead Bias Lab provides:

-   Entity-based document scanning\
-   Deterministic masking\
-   Original vs Masked LLM scoring\
-   Paired statistical testing\
-   Cost forecasting\
-   Prompt repair suggestions\
-   Full Excel export

It transforms your dataset + prompt + LLM model into a **controlled
experimental setup**, letting you discover dependencies, biases, and
hidden correlations inside your LLM scoring pipeline.
