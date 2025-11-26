import textwrap

def generate_remediation_prompt(original_prompt, biased_categories):
    """
    Returns a prompt that instructs an LLM to act as a redaction tool.
    It asks the LLM to rewrite the text, replacing biased entities with placeholders.
    """
    instructions = []
    
    # Map categories to specific masking/redaction instructions
    if "Time" in biased_categories:
        instructions.append("- DATES & TIME: Replace specific years, dates, and relative times (e.g., 'last Q4') with '[DATE]' or '[TIME]'.")
    if "Organizations" in biased_categories:
        instructions.append("- ENTITIES: Replace proper names of companies, agencies, and institutions with '[ORG]'.")
    if "Numbers" in biased_categories:
        instructions.append("- FINANCIALS: Replace specific numerical values, revenues, and percentages with '[NUM]'.")
    if "Locations" in biased_categories:
        instructions.append("- LOCATIONS: Replace specific countries, cities, and regions with '[LOC]'.")
    if "Person Names" in biased_categories:
        instructions.append("- NAMES: Replace names of specific individuals with '[PERSON]'.")
    if "Gender" in biased_categories:
        instructions.append("- GENDER: Replace gendered pronouns (he/she) with neutral terms like '[THEY]'.")
    if "Products" in biased_categories:
        instructions.append("- PRODUCTS: Replace specific product names (e.g., 'iPhone 15') with generic terms like '[PRODUCT]'.")

    instruction_block = "\n".join(instructions)

    # Note: We largely ignore 'original_prompt' here because the goal 
    # is to output a clean dataset, not to perform the analysis yet.
    new_prompt = f"""
    ### TEXT REDACTION TASK
    Your goal is to prepare a dataset for blind analysis by removing Look-Ahead Bias. 
    Please rewrite the provided text, keeping the original structure and non-sensitive content exactly as is, but apply the following masking rules:

    {instruction_block}

    ### INPUT TEXT
    [Paste your text here]

    ### OUTPUT
    Return ONLY the masked text. Do not summarize or analyze.
    """.strip()

    return new_prompt


def generate_remediation_code(biased_categories):
    """
    Returns a string containing a valid Python script using spaCy 
    to mask the specific categories found.
    """
    
    # Map your UI categories to standard spaCy ENT labels
    spacy_map = {
        "Time": ["DATE", "TIME"],
        "Organizations": ["ORG"],
        "Numbers": ["MONEY", "CARDINAL", "PERCENT", "QUANTITY"],
        "Locations": ["GPE", "LOC", "FAC"],
        "Person Names": ["PERSON"],
        "Products": ["PRODUCT"],
        "Gender": [] # SpaCy default models don't have a specific GENDER entity, usually handled via pronouns logic or custom components
    }

    target_labels = []
    for cat in biased_categories:
        if cat in spacy_map:
            target_labels.extend(spacy_map[cat])
            
    # Format the list for the python string
    labels_str = str(target_labels)

    code_snippet = f'''import spacy

def mask_bias_entities(text):
    """
    Standalone function to mask detected Look-Ahead Biases.
    Targeting specific entities: {", ".join(biased_categories)}
    """
    # Load model (ensure 'en_core_web_sm' is installed)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    
    # Entity labels identified as sources of bias
    TARGET_LABELS = {labels_str}
    
    # Filter entities
    spans_to_mask = [ent for ent in doc.ents if ent.label_ in TARGET_LABELS]
    
    # Sort spans in reverse order to mask without messing up indices
    spans_to_mask.sort(key=lambda x: x.start_char, reverse=True)
    
    text_chars = list(text)
    
    for ent in spans_to_mask:
        # Replace entity content with a generic [MASK] or [ENTITY_TYPE]
        replacement = f"[MASKED_{{ent.label_}}]" 
        text_chars[ent.start_char:ent.end_char] = list(replacement)
        
    return "".join(text_chars)

# Example Usage:
# clean_text = mask_bias_entities("Apple reported huge gains in Q4 2023.")
# print(clean_text)
'''
    return code_snippet