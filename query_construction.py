import pandas as pd
import numpy as np




def query_maker(n):
    df = pd.read_csv('liar_dataset/train.tsv', sep='\t', names=['label', 'statement', 'subjects',
                                                                'speaker', 'job_title', 'state_info',
                                                                'party', 'barely_true', 'false', 'half_true',
                                                                'mostly_true', 'pants_on_fire', 'context'])
    queries = {}
    for i in range(n):
        # Extract key components
        speaker = df["speaker"][i]
        statement = df["statement"][i]
        subject = df["subject"][i] if 'subject' in df.columns else ''
        location = df["state_info"][i] if 'state_info' in df.columns else ''
        context = df["context"][i] if 'context' in df.columns else ''

        # Simplified query focusing on key elements
        core_query = f"{speaker} claim: {statement}"

        #  Add subject or context for more specificity if needed
        if subject:
            core_query += f" on {subject}"
        if location:
            core_query += f" in {location}"
        if context:
            core_query += f" ({context})"

        queries[i] = core_query

    return queries

