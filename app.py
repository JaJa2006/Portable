import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

st.title("Threat Event Grouping Application")


# auto-download model from huggingface hub
def download_hf_file(repo_id, filename, local_dir="models"):
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)

    if os.path.exists(local_path):
        return local_path

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=local_dir,
        local_dir_use_symlinks=False  # required for Streamlit Cloud
    )


# embedding Model Loader
@st.cache_resource
def load_embedding_model():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    return SentenceTransformer(model_name)


model = load_embedding_model()


# LLM Loader (GGUF models)
@st.cache_resource
def load_llm():
    repo_id = "Qwen/Qwen2-0.5B-Instruct-GGUF"
    file_name = "qwen2-0_5b-instruct-q8_0.gguf"

    model_path = download_hf_file(repo_id, file_name)

    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=8
    )


llm = load_llm()


# LLM-based classification helpers
def ai_check_group_match(threat_event, current_group, risk_info):
    prompt = f"""
    You are a cybersecurity assistant analyzing event clustering.

    Threat Event: {threat_event}
    Proposed Group: {current_group}
    Risk Context: {risk_info}

    Respond with exactly one word:
    - "GOOD" if the threat event fits well in this group
    - "NOT GOOD" if it does not fit
    """

    full_prompt = f"<|im_start|>system\nYou are an expert cybersecurity event clustering assistant.<|im_end|>\n" \
                  f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    res = llm(full_prompt, max_tokens=10)
    return res["choices"][0]["text"].strip().upper()


def ai_propose_new_group(threat_event, risk_info):
    prompt = f"""
    You are a cybersecurity assistant helping cluster threat events.

    Threat Event: {threat_event}
    Risk Context: {risk_info}
    Existing groups: {', '.join(groupings['Group Name'].unique().tolist())}

    Propose a short, clear name for a new threat group.
    Respond ONLY with the name — no explanations.
    """

    full_prompt = f"<|im_start|>system\nYou are an expert cybersecurity clustering assistant.<|im_end|>\n" \
                  f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    res = llm(full_prompt, max_tokens=40)
    return res["choices"][0]["text"].strip()


# upload file
groupings_file = st.file_uploader("Upload Groupings Excel", type="xlsx")

if groupings_file:
    groupings = pd.read_excel(groupings_file)
    group_embeddings = model.encode(
        groupings["Threat Event"].tolist(),
        normalize_embeddings=True
    )

uploaded_file = st.file_uploader("Upload Threat Events Excel", type="xlsx")

if uploaded_file and groupings_file:

    new_threats = pd.read_excel(uploaded_file)

    if "Threat Event" not in new_threats.columns:
        st.error("Uploaded file must have a 'Threat Event' column.")
        st.stop()

    results = []

    for idx, row in new_threats.iterrows():
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

        event_emb = model.encode([threat_event], normalize_embeddings=True)
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
            llm_response = ai_check_group_match(threat_event, best_group, risk_info)
            if "NOT GOOD" in llm_response:
                final_group = ai_propose_new_group(threat_event, risk_info)
                indicator = "AI Created"
            else:
                indicator = "Must Check"
        else:
            llm_response = ai_check_group_match(threat_event, best_group, risk_info)
            if "NOT GOOD" in llm_response:
                final_group = ai_propose_new_group(threat_event, risk_info)
                indicator = "AI Generated"
            else:
                final_group = f"AI_{best_group}"
                indicator = "AI Verified"

        results.append({
            "ThreatEvent": threat_event,
            "BestGroup": best_group,
            "HighestAvgScore": round(best_score, 2),
            "FinalGroup": final_group,
            "Indicator": indicator
        })

    st.subheader("Final Results")

    st.dataframe(pd.DataFrame(results))



