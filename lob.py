#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Look-Ahead Bias Lab v0.982 - Detailed Reporting & Fixes
Updates:
  - Step 4: Detailed table showing Found Docs vs. Used Docs (Capped).
  - Cost: Explicitly includes Output tokens in calculation.
  - Fix: Solved 'AttributeError: st.rerun' for older Streamlit versions.
  - Retains: Legacy compatibility (no column_config).

Author: You (Garvin)
License: MIT
"""

from __future__ import annotations
import os
import io
import json
import time
import re
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# --- CONFIG IMPORT ---
try:
    from config import *
except ImportError:
    st.error("Critical Error: 'config.py' not found. Please ensure it exists in the same directory.")
    st.stop()

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- COMPATIBILITY HELPER ---
def safe_rerun():
    """Handles rerun for different Streamlit versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# --- NLP SETUP ---
try:
    import spacy
    from spacy.cli import download
    try:
        if not spacy.util.is_package("en_core_web_lg"):
            print("Downloading spaCy large model...")
            download("en_core_web_lg")
        nlp = spacy.load("en_core_web_lg")
    except:
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
except Exception as e:
    nlp = None
    print(f"SpaCy error: {e}")

# Optional providers
try:
    import openai
except Exception:
    openai = None

try:
    import litellm
    litellm.suppress_debug_info = True
except Exception:
    litellm = None

# ------------------------------
# Data structures
# ------------------------------
@dataclass
class ProviderConfig:
    name: str; provider: str; model: str; 
    temperature: float = 0.0; top_p: float = 1.0; max_tokens: int = 256; seed: int = RANDOM_DEFAULT_SEED

# ------------------------------
# Provider Shims
# ------------------------------
class ModelProvider:
    def __init__(self, cfg: ProviderConfig): self.cfg = cfg
    def infer(self, prompt: Dict[str, Any]) -> str: raise NotImplementedError

class OpenAIProvider(ModelProvider):
    def __init__(self, cfg: ProviderConfig):
        super().__init__(cfg)
        if openai is None: raise RuntimeError("openai missing")
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    def infer(self, prompt):
        msgs = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]
        for _ in range(3):
            try:
                return self.client.chat.completions.create(
                    model=self.cfg.model, messages=msgs, temperature=0, seed=self.cfg.seed
                ).choices[0].message.content.strip()
            except Exception as e: time.sleep(1); last_err=e
        return f"Error: {last_err}"

class LiteLLMProvider(ModelProvider):
    def __init__(self, cfg: ProviderConfig):
        super().__init__(cfg)
        if litellm is None: raise RuntimeError("litellm missing")
        self.keys = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GOOGLE_API_KEY"}
        for k, v in self.keys.items():
            if k in cfg.model.lower() and os.environ.get(v): pass 
    def infer(self, prompt):
        msgs = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]
        for _ in range(3):
            try:
                return litellm.completion(model=self.cfg.model, messages=msgs, temperature=0).choices[0].message.content.strip()
            except Exception as e: time.sleep(1); last_err=e
        return f"Error: {last_err}"

class MockProvider(ModelProvider):
    def infer(self, prompt): return f"Signal: {np.random.rand():.3f}\nMock."

# ------------------------------
# 1. NLP & Subset Logic
# ------------------------------

def detect_entities_in_text(text: str) -> List[str]:
    """Scans a single text and returns which categories were found."""
    if not nlp or not text: return []
    found_cats = set()
    doc = nlp(text[:10000]) # Limit for speed
    for ent in doc.ents:
        for cat, config in ENTITY_CONFIG.items():
            if ent.label_ in config["spacy"]: found_cats.add(cat)
    for cat, config in ENTITY_CONFIG.items():
        if config["regex"] and re.search(config["regex"], text, re.IGNORECASE): found_cats.add(cat)
    return list(found_cats)

def create_smart_subsets(df_raw):
    """
    Creates subsets. Returns both the raw counts and the sampled subsets.
    """
    subsets = {}
    meta_counts = {} # Track raw counts vs used counts
    
    if 'detected_cats' not in df_raw.columns:
        with st.spinner("Scanning corpus for entities (spaCy)..."):
            df_raw['detected_cats'] = df_raw['text'].apply(detect_entities_in_text)
            
    for cat in ALL_MASK_CATEGORIES:
        # 1. Filter docs containing this category
        mask = df_raw['detected_cats'].apply(lambda x: cat in x)
        subset = df_raw[mask].copy()
        
        raw_count = len(subset)
        
        # 2. Enforce Max Size with Sampling
        if raw_count > MAX_SUBSET_SIZE:
            subset = subset.sample(n=MAX_SUBSET_SIZE, random_state=RANDOM_DEFAULT_SEED)
            
        subsets[cat] = subset
        meta_counts[cat] = {"found": raw_count, "used": len(subset)}
        
    return subsets, meta_counts

# ------------------------------
# 2. Deterministic Masking Engine
# ------------------------------

def mask_text_deterministic(text: str, category: str) -> str:
    if not text: return ""
    config = ENTITY_CONFIG.get(category)
    if not config: return text
    
    masked_text = text
    if config["regex"]:
        masked_text = re.sub(config["regex"], "[MASK]", masked_text, flags=re.IGNORECASE)
    
    if nlp and config["spacy"]:
        doc = nlp(masked_text)
        spans_to_mask = []
        for ent in doc.ents:
            if ent.label_ in config["spacy"]: spans_to_mask.append((ent.start_char, ent.end_char))
        spans_to_mask.sort(key=lambda x: x[0], reverse=True)
        text_chars = list(masked_text)
        for start, end in spans_to_mask: text_chars[start:end] = list("[MASK]")
        masked_text = "".join(text_chars)
        
    return masked_text

# ------------------------------
# 3. Analysis & Inference (Self-Healing)
# ------------------------------

def run_analysis_batch(batch_docs, provider, text_col, prompt_q):
    batch_results = []
    prompt_txt = f"Task: {prompt_q}\n\n"
    
    for idx, row in enumerate(batch_docs):
        txt_content = str(row[text_col]).replace("\n", " ")[:2000]
        prompt_txt += f"--- Doc {idx} ID:{row['doc_id']} ---\n{txt_content}\n"
    
    prompt_txt += "\nOUTPUT: JSON array of objects [{'doc_id': '...', 'signal': float 0-1}]"
    
    try:
        res = provider.infer({"system": "Analyst.", "user": prompt_txt})
        clean = res.replace("```json", "").replace("```", "").strip()
        start, end = clean.find('['), clean.rfind(']')
        if start != -1:
            data = json.loads(clean[start:end+1])
            return data
    except Exception as e:
        print(f"Batch Error: {e}")
        return []
    return []

def run_analysis_on_subset(df_subset, provider, text_col, prompt_q):
    final_results_map = {} 
    all_docs = df_subset.to_dict('records')
    
    # Pass 1
    for i in range(0, len(all_docs), 20):
        batch = all_docs[i:i+20]
        data = run_analysis_batch(batch, provider, text_col, prompt_q)
        for item in data:
            if 'doc_id' in item and 'signal' in item:
                try: final_results_map[str(item['doc_id'])] = float(item['signal'])
                except: pass

    # Pass 2 (Repair)
    missing = [d for d in all_docs if str(d['doc_id']) not in final_results_map]
    if missing:
        print(f"Repairing {len(missing)} docs...")
        for i in range(0, len(missing), 5):
            batch = missing[i:i+5]
            data = run_analysis_batch(batch, provider, text_col, prompt_q)
            for item in data:
                if 'doc_id' in item and 'signal' in item:
                    try: final_results_map[str(item['doc_id'])] = float(item['signal'])
                    except: pass

    results_list = [{"doc_id": k, "sentiment": v} for k, v in final_results_map.items()]
    return pd.DataFrame(results_list)

# ------------------------------
# 4. Cost Estimator (Detailed)
# ------------------------------

def estimate_cost(subsets, meta_counts, model_name):
    """
    Calculates detailed token usage including Input AND Output.
    """
    
    # Get Price
    price_in, price_out = PRICING_TABLE.get(model_name, PRICING_TABLE["default"])
    if "claude" in model_name and "haiku" in model_name:
        price_in, price_out = 1.0, 5.0 
        
    total_cost = 0.0
    breakdown = []
    
    for cat in ALL_MASK_CATEGORIES:
        df_sub = subsets.get(cat, pd.DataFrame())
        counts = meta_counts.get(cat, {"found": 0, "used": 0})
        used_count = counts['used']
        
        if used_count < MIN_DOCS_THRESHOLD:
            status = "❌ Skipped (Too few)" if used_count > 0 else "❌ Skipped (Empty)"
            est_cost_str = "$0.00"
        else:
            status = "✅ Analyzing"
            
            # Calculate tokens
            # Input: Text length + Prompt
            avg_chars = df_sub['text'].str.len().mean()
            avg_in_tokens = (avg_chars / 4) + 150 # + buffer
            
            # Output: JSON response (~50 tokens)
            avg_out_tokens = 50
            
            # Total = 2 Passes (Original + Masked) * N docs
            cat_input_tokens = used_count * 2 * avg_in_tokens
            cat_output_tokens = used_count * 2 * avg_out_tokens
            
            cost_in = (cat_input_tokens / 1_000_000) * price_in
            cost_out = (cat_output_tokens / 1_000_000) * price_out
            cat_total = cost_in + cost_out
            
            total_cost += cat_total
            est_cost_str = f"${cat_total:.4f}"

        breakdown.append({
            "Category": cat,
            "Found": counts['found'],
            "Using": used_count,
            "Status": status,
            "Est. Cost": est_cost_str
        })
        
    return total_cost, pd.DataFrame(breakdown)

# ------------------------------
# 5. Prompt Engineering
# ------------------------------

def generate_debiased_prompt(original_prompt, significant_categories):
    if not significant_categories: return original_prompt
    constraints = []
    for cat in significant_categories:
        if cat == "Time": constraints.append("- IGNORE all dates, years, and relative time.")
        elif cat == "Organizations": constraints.append("- IGNORE company names.")
        elif cat == "Numbers": constraints.append("- IGNORE financial figures.")
        elif cat == "Locations": constraints.append("- IGNORE locations.")
        elif cat == "Person Names": constraints.append("- IGNORE people's names.")
        elif cat == "Gender": constraints.append("- IGNORE gender.")
        elif cat == "Products": constraints.append("- IGNORE product names.")
    return f"{original_prompt}\n\n### DEBIASING CONSTRAINTS\nStrictly adhere to:\n" + "\n".join(constraints) + "\n\nFocus ONLY on remaining content."

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.set_page_config(page_title="Lab v0.982", layout="wide")
if os.path.exists("img/logo_tum.jpeg"): st.sidebar.image("img/logo_tum.jpeg")
st.title("Look-Ahead Bias Lab (v0.982)")

# --- INTRODUCTORY TEXT ---
st.markdown("""
### Do not use this tool! for internal use only!
### 🛡️ Look-Ahead Bias Detection for Binary Classification
This tool is designed to stress-test **Binary Classification** tasks (e.g., Positive/Negative, Buy/Sell) or other **2-Class Problems** where the output can be expressed as a **Score [0.0 - 1.0]**.

**How it works:**
1.  **Upload Corpus:** Load your text data (News, Filings, Transcripts).
2.  **Entity Detection:** The system uses local NLP to find documents containing **Time, Organizations, Numbers, Locations, Person Names, Gender, and Products**.
3.  **Targeted Masking:** It creates "masked" versions of your text where specific information is hidden (using deterministic spaCy + Regex patterns).
4.  **Score Comparison:** It runs your prompt on both the **Original** and **Masked** text using your chosen LLM.
5.  **Bias Diagnosis:** If the score shifts significantly when information is hidden (e.g., removing dates changes the score), the model likely has Look-Ahead Bias.
""")

# --- STEP 1: UPLOAD ---
st.header("Step 1: Upload Dataset")
uploaded = st.file_uploader("Upload Corpus (CSV/Parquet)", type=['csv', 'parquet'])

if uploaded:
    df_in = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_parquet(uploaded)
    st.write("Preview of uploaded data (first 3 rows):", df_in.head(3))
    
    c1, c2, c3 = st.columns(3)
    text_col = c1.selectbox("Select Text Column (Mandatory)", df_in.columns)
    
    df_raw = pd.DataFrame()
    df_raw['text'] = df_in[text_col].astype(str)
    df_raw['doc_id'] = df_in.index.astype(str)
    df_raw['timestamp'] = "N/A"
    

    
    st.success(f"✅ Loaded {len(df_raw)} documents.")

    
    with st.expander("🔎 Inspect First Document (Full Text)"):
        st.write(df_raw.iloc[0]['text'])
else:
    df_raw = pd.DataFrame({"doc_id":["1"],"timestamp":["N/A"],"text":["Sample Apple Q4 report 2023."]})
    st.info("Upload a file to begin.")

# --- STEP 2: CONFIG ---
st.header("Step 2: Define Question")
st.info("""
**📝 Prompt Guidelines:**
Your prompt MUST request a **floating point score between 0.0 and 1.0**.
* **Good:** "What is the sentiment score between 0 (Negative) and 1 (Positive)?"
* **Good:** "Rate the hawkishness from 0 to 1."
* **Bad:** "Is this positive? Answer Yes or No." (Statistical tests require continuous scores).
""")
user_prompt = st.text_area("Prompt", "What is the sentiment score between 0.0 (Negative) and 1.0 (Positive)?")

# --- STEP 3: PROVIDER ---
st.header("Step 3: Configure LLM")

st.warning("""
**🔐 Security & Privacy Notice**
API keys entered here are transmitted to the Streamlit server for execution. 
While this app does not permanently store your keys, sending credentials over the internet always carries some risk.

**For sensitive data or enterprise use:** We strongly recommend running this tool locally.
[Download Source Code from GitHub](https://github.com/YourRepo/LookAheadBiasLab)
""")

c1, c2 = st.columns(2)
prov_name = c1.selectbox("Provider", list(MODEL_OPTIONS.keys()))
model_name = c2.selectbox("Model", MODEL_OPTIONS[prov_name])
api_key = st.text_input("API Key", type="password")

# --- STEP 4: SCAN & ESTIMATE ---
st.header("Step 4: Scan & Estimate Cost")

if 'subsets' not in st.session_state:
    st.session_state['subsets'] = None

if st.button("🕵️ Scan Corpus & Estimate Cost"):
    if not nlp: st.error("NLP Engine missing.")
    else:
        # 1. Create Subsets & Get Metadata
        subsets, meta_counts = create_smart_subsets(df_raw)
        st.session_state['subsets'] = subsets
        
        # 2. Calculate Cost
        est_cost, breakdown = estimate_cost(subsets, meta_counts, model_name)
        
        # 3. Display Detailed Report
        st.write("### 📋 Scan Report")
        st.table(breakdown)
        st.info("This is an estimate only. Please refer to the LLM provider's pricing page for the most accurate cost.")
        c1, c2 = st.columns(2)
        c1.metric("Total Est. Cost", f"${est_cost:.4f}")
        c2.info(f"Docs Capped at {MAX_SUBSET_SIZE} per category.")
        
        st.session_state['scan_complete'] = True

# --- STEP 5: RUN ---
if st.session_state.get('scan_complete'):
    st.header("Step 5: Run Analysis")
    if st.button("🚀 Execute Analysis (Charge API)"):
        if not api_key and prov_name != "Mock":
            st.error("Enter API Key.")
        else:
            # Setup Provider
            if prov_name=="OpenAI": os.environ["OPENAI_API_KEY"] = api_key
            if prov_name=="Claude": os.environ["ANTHROPIC_API_KEY"] = api_key
            if prov_name=="Gemini": os.environ["GOOGLE_API_KEY"] = api_key
            
            cfg = ProviderConfig("main", prov_name.lower(), model_name)
            if prov_name=="OpenAI": prov = OpenAIProvider(cfg)
            elif prov_name=="Claude": prov = LiteLLMProvider(cfg)
            elif prov_name=="Gemini": prov = LiteLLMProvider(cfg)
            else: prov = MockProvider(cfg)
            
            # Run Loop
            subsets = st.session_state['subsets']
            results_container = []
            prog = st.progress(0)
            status = st.empty()
            
            active_cats = [k for k,v in subsets.items() if len(v) >= MIN_DOCS_THRESHOLD]
            
            for i, cat in enumerate(active_cats):
                df_sub = subsets[cat]
                status.markdown(f"### 🔄 Analyzing: **{cat}** ({len(df_sub)} docs)")
                
                # Mask
                df_sub[f"text_masked_{cat}"] = df_sub["text"].apply(lambda x: mask_text_deterministic(x, cat))
                
                # Analyze
                r_orig = run_analysis_on_subset(df_sub, prov, "text", user_prompt)
                r_orig['version'] = 'Original'; r_orig['category_group'] = cat
                
                r_mask = run_analysis_on_subset(df_sub, prov, f"text_masked_{cat}", user_prompt)
                r_mask['version'] = 'Masked'; r_mask['category_group'] = cat
                
                results_container.append(pd.concat([r_orig, r_mask]))
                st.session_state[f"subset_{cat}"] = df_sub
                prog.progress((i+1)/len(active_cats))
                
            if results_container:
                st.session_state['final_results'] = pd.concat(results_container).dropna()
                st.success("Analysis Complete!")
                safe_rerun() # Fixed for older Streamlit

# --- RESULTS ---
if 'final_results' in st.session_state:
    st.write("---")
    st.header("📊 Results")
    res = st.session_state['final_results']
    sig_cats = []
    stats = []
    
    for cat in res['category_group'].unique():
        sub = res[res['category_group'] == cat]
        piv = sub.pivot(index='doc_id', columns='version', values='sentiment').dropna()
        if len(piv) >= MIN_DOCS_THRESHOLD:
            t, p = scipy_stats.ttest_rel(piv['Original'], piv['Masked'])
            sig = p < 0.05
            if sig: sig_cats.append(cat)
            stats.append({
                "Category": cat, "N": len(piv),
                "Orig": f"{piv['Original'].mean():.3f}", "Mask": f"{piv['Masked'].mean():.3f}",
                "P-Val": f"{p:.4f}", "Bias?": "⚠️ YES" if sig else "✅ No"
            })
            
    st.table(pd.DataFrame(stats))
    
    if sig_cats:
        st.error(f"Bias found in: {', '.join(sig_cats)}")
        for c in sig_cats: st.warning(f"**{c}**: {BIAS_FEEDBACK_MESSAGES.get(c,'')}")
        st.text_area("Recommended Prompt", generate_debiased_prompt(user_prompt, sig_cats))
    else:
        st.success("No significant bias detected.")
        
    # Export
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        pd.DataFrame(stats).to_excel(w, sheet_name='Stats', index=False)
        res.to_excel(w, sheet_name='Raw', index=False)
        for cat in res['category_group'].unique():
            if f"subset_{cat}" in st.session_state:
                s = st.session_state[f"subset_{cat}"]
                p = res[res['category_group']==cat].pivot(index='doc_id', columns='version', values='sentiment').reset_index()
                # Ensure ID types match for merge
                s['doc_id'] = s['doc_id'].astype(str)
                p['doc_id'] = p['doc_id'].astype(str)
                pd.merge(s, p, on='doc_id').to_excel(w, sheet_name=f"{cat}_Data")
    st.download_button("⬇️ Download Excel", out.getvalue(), "results.xlsx")