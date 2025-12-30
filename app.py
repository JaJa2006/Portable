import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

st.title("Threat Event Grouping Application")


# embedding Model Loader
@st.cache_resource
def load_embedding_model():
    try:
        model_name = "intfloat/e5-base-v2"
        return SentenceTransformer(model_name)
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        st.stop()


model = load_embedding_model()


# LLM Loader
@st.cache_resource
def load_llm():
    try:
        repo_id = "SixOpen/Phi-3-mini-4k-instruct-Q4_K_M-GGUF"
        file_name = "phi-3-mini-4k-instruct-q4_k_m.gguf"

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            cache_dir="models",
            local_dir_use_symlinks=False
        )

        return Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.1,
            verbose=False
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        st.stop()


llm = load_llm()


# LLM-based classification helpers
def ai_check_group_match(threat_event, current_group, risk_info):
    prompt = f"""
            You are a SOC analyst specializing in threat clustering.

            Threat Event:
            {threat_event}

            Proposed Group:
            {current_group}

            Risk Context:
            {risk_info}

            Answer with exactly:
            GOOD or NOT GOOD
            """
            
    prompt_format = f"""<|user|>
                {prompt}<|end|>
                <|assistant|>"""

    res = llm(prompt_format, max_tokens=10)
    return res["choices"][0]["text"].strip().upper()


def ai_propose_new_group(threat_event, risk_info):
    
    prompt = f"""
    You are a SOC analyst.

    Generate ONE short cyber threat group name.

    Rules:
    - 15 words maximum
    - Use cybersecurity terminology
    - No punctuation
    - No explanation

    Threat Event:
    {threat_event}

    Risk Context:
    {risk_info}

    Existing Groups:
    {', '.join(groupings['Group Name'].unique().tolist())}

    Provide a new, concise threat group name in plain text:
    """
    
    prompt_format = f"""<|user|>
                    {prompt}<|end|>
                    <|assistant|>"""

    res = llm(prompt_format, max_tokens=40)
    return res["choices"][0]["text"].strip()


# file uploader
groupings_file = st.file_uploader(
    "Upload Groupings File",
    type=["xlsx"]
)

threat_file = st.file_uploader(
    "Upload Threat Events File",
    type=["xlsx"]
)

if groupings_file and threat_file:
    try:
        groupings = pd.read_excel(groupings_file)
    except Exception:
        st.error("Failed to read Groupings file. Ensure it is a valid XLSX.")
        st.stop()

    try:
        new_threats = pd.read_excel(threat_file)
    except Exception:
        st.error("Failed to read Threat file. Ensure it is a valid XLSX.")
        st.stop()

    # validate columns
    required_group_cols = {"Risk Scenario", "Threat Event", "Group Name"}
    required_threat_cols = {"Risk Scenario", "Threat Event"}

    if not required_group_cols.issubset(groupings.columns):
        st.error(
            f"Groupings file must contain columns: {', '.join(required_group_cols)}"
        )
        st.stop()

    if not required_threat_cols.issubset(new_threats.columns):
        st.error(
            f"Threat file must contain columns: {', '.join(required_threat_cols)}"
        )
        st.stop()

    groupings = groupings.dropna(subset=["Threat Event", "Group Name"])

    if groupings.empty:
        st.error("Groupings file contains no valid Threat Event data.")
        st.stop()

    # precompute embeddings
    group_embeddings = model.encode(
        groupings["Threat Event"].tolist(),
        normalize_embeddings=True
    )

    results = []

    total_rows = len(new_threats)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    with st.spinner(""):
        for idx, row in new_threats.iterrows():
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
            progress_text.write(f"Processing row {idx + 1} / {total_rows}")

            threat_event = row["Threat Event"]
            risk_info = row.get("Risk Scenario", "")

            if pd.isna(threat_event) or str(threat_event).strip() in ["", "NA"]:
                results.append({
                    "ThreatEvent": threat_event,
                    "BestGroup": "NA",
                    "HighestAvgScore": "NA",
                    "FinalGroup": "NA",
                    "Indicator": "No Threat Event"
                })
                continue

            event_emb = model.encode(
                [threat_event],
                normalize_embeddings=True
            )

            sims = util.cos_sim(event_emb, group_embeddings).numpy().flatten()
            groupings["Similarity"] = sims

            avg_scores = groupings.groupby("Group Name")["Similarity"].mean()
            best_group = avg_scores.idxmax()
            best_score = float(avg_scores.max() * 100)

            indicator = ""
            final_group = best_group

            if best_score >= 90:
                indicator = "Highly Accurate"
            elif best_score >= 85:
                indicator = "Moderately Accurate"
            elif best_score >= 75:
                llm_response = ai_check_group_match(
                    threat_event, best_group, risk_info
                )
                if "NOT GOOD" in llm_response:
                    final_group = ai_propose_new_group(
                        threat_event, risk_info
                    )
                    indicator = "AI Generated"
                else:
                    indicator = "Must Check"
            else:
                llm_response = ai_check_group_match(
                    threat_event, best_group, risk_info
                )
                if "NOT GOOD" in llm_response:
                    final_group = ai_propose_new_group(
                        threat_event, risk_info
                    )
                    indicator = "AI Generated"
                else:
                    indicator = "AI Verified"

            results.append({
                "ThreatEvent": threat_event,
                "BestGroup": best_group,
                "HighestAvgScore": round(best_score, 2),
                "FinalGroup": final_group,
                "Indicator": indicator
            })

    progress_bar.empty()
    progress_text.empty()

    st.subheader("Final Results")
    st.dataframe(pd.DataFrame(results))
