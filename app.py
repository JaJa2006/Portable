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
    try:
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)

        if os.path.exists(local_path):
            return local_path

        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=local_dir,
            local_dir_use_symlinks=False
        )
    except:
        st.error("Error downloading model file")
        st.stop()


# embedding Model Loader
@st.cache_resource
def load_embedding_model():
    try:
        model_name = "intfloat/e5-base-v2"
        return SentenceTransformer(model_name)
    except:
        st.error("Failed to load embedding model")
        st.stop()


model = load_embedding_model()


# LLM Loader
@st.cache_resource
def load_llm():
    try:
        repo_id = "Qwen/Qwen2-0.5B-Instruct-GGUF"
        file_name = "qwen2-0_5b-instruct-q8_0.gguf" 

        model_path = download_hf_file(repo_id, file_name)

        return Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8
        )
    except Exception as e:
        st.error("Failed to load LLM")
        st.stop()

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

# function to convert string data to dataframe
def string_to_df(data_string):
    try:
        rows = data_string.split("<ROW>")
        parsed = []
        for row in rows:
            row = row.strip()
            if not row:
                continue

            col_pairs = row.split("<COL>")
            row_dict = {}

            for pair in col_pairs:
                if "<SEP>" in pair:
                    colname, value = pair.split("<SEP>", 1)
                    row_dict[colname] = value
            parsed.append(row_dict)
        return pd.DataFrame(parsed)
    except:
        st.error("Failed to parse text into table")
        return pd.DataFrame()

# upload text
groupings_text = st.text_area("Enter Groupings Data:", height=200)

threat_text = st.text_area("Enter Threat Data:", height=200)

if threat_text and groupings_text:
    
    groupings_text = groupings_text.replace("\r", "").strip()
    groupings_text = groupings_text.replace("\n", "").strip()
    groupings = string_to_df(groupings_text)
    
    group_embeddings = model.encode(groupings["Threat Event"].tolist(), normalize_embeddings=True)
    
    if "Risk Scenario" not in groupings.columns or "Threat Event" not in groupings.columns or "Group Name" not in groupings.columns:
        st.error("Grouping data must have a 'Risk Scenario', 'Threat Event' and 'Group Name' column.")
        st.stop()

    threat_text = threat_text.replace("\r", "").strip()
    threat_text = threat_text.replace("\n", "").strip()
    new_threats = string_to_df(threat_text)

    if "Risk Scenario" not in new_threats.columns or "Threat Event" not in new_threats.columns:
        st.error("Threat data must have a 'Risk Scenario' and 'Threat Event' column.")
        st.stop()

    results = []
    
    # progress bar
    total_rows = len(new_threats)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    with st.spinner(""):
        for idx, row in new_threats.iterrows():
            
            # update progress bar
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
                    indicator = "AI Generated"
                else:
                    indicator = "Must Check"
            else:
                llm_response = ai_check_group_match(threat_event, best_group, risk_info)
                if "NOT GOOD" in llm_response:
                    final_group = ai_propose_new_group(threat_event, risk_info)
                    indicator = "AI Generated"
                else:
                    final_group = {best_group}
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
