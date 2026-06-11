import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
import re
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px

# Config
MODEL_PATH = 'distilbert-base-uncased'
THRESHOLD  = 0.46
URGENT_WORDS = [
    'urgent', 'asap', 'immediately', 'emergency', 'critical',
    'cannot access', 'not working', 'broken', 'error', 'failed',
    'crash', 'outage', 'down', 'blocked', 'fraud',
    'unauthorized', 'hacked', 'data loss', 'stolen', 'locked'
]
priority_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}

st.set_page_config(
    page_title="SIA — Support Integrity Auditor",
    page_icon="🔍",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ── Helper functions ──────────────────────────────────────────
def get_urgency_score(text):
    return min(sum(1 for w in URGENT_WORDS if w in str(text).lower()), 10)

def get_inferred_severity(rt):
    if rt <= 18:   return 'Critical'
    elif rt <= 34: return 'High'
    elif rt <= 44: return 'Medium'
    else:          return 'Low'

def build_input(subject, description, channel, rt, category, urgency):
    text = f"{subject.lower()} [SEP] {description.lower()}"
    return (f"{text} "
            f"[CHANNEL] {channel} "
            f"[CAT] {category} "
            f"[RT] {rt} "
            f"[URGENCY] {urgency}")

def predict(text_input):
    enc = tokenizer(
        text_input,
        return_tensors='pt',
        truncation=True,
        max_length=256,
        padding='max_length'
    )
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    conf  = probs[1].item()
    pred  = 1 if conf >= THRESHOLD else 0
    return pred, round(conf, 4)

def build_dossier(ticket_id, assigned, inferred, rt, sat, 
                   confidence, keywords):
    a_num = priority_map.get(assigned, 2)
    i_num = priority_map.get(inferred, 2)
    mismatch_type  = "Hidden Crisis" if i_num > a_num else "False Alarm"
    severity_delta = f"{assigned} → {inferred}"
    kw_str         = ", ".join(keywords) if keywords else "none detected"

    if i_num > a_num:
        analysis = (
            f"Ticket assigned '{assigned}' but signals indicate '{inferred}'. "
            f"Resolved in {rt}h with urgency indicators: {kw_str}. "
            f"Satisfaction {sat}/5 confirms severity was underestimated."
        )
    else:
        analysis = (
            f"Ticket assigned '{assigned}' but signals indicate '{inferred}'. "
            f"Resolution took {rt}h and satisfaction was {sat}/5, "
            f"suggesting severity was overestimated."
        )

    return {
        "ticket_id":           ticket_id,
        "assigned_priority":   assigned,
        "inferred_severity":   inferred,
        "mismatch_type":       mismatch_type,
        "severity_delta":      severity_delta,
        "feature_evidence": [
            {"signal": "keyword",
             "value":  kw_str,
             "weight": str(len(keywords))},
            {"signal": "resolution_time",
             "value":  str(rt),
             "interpretation": "fast — high severity" if rt <= 24
                               else "slow — low severity"},
            {"signal": "satisfaction_score",
             "value":  str(sat),
             "interpretation": "low — serious issue" if sat <= 3
                               else "high — minor issue"}
        ],
        "constraint_analysis": analysis,
        "confidence":          confidence
    }

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/inspection.png", width=80)
st.sidebar.title("SIA")
st.sidebar.markdown("**Support Integrity Auditor**")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🎫 Single Ticket", "📦 Batch CSV", "📊 Dashboard"]
)

# ══════════════════════════════════════════════════════════════
# PAGE 1: SINGLE TICKET
# ══════════════════════════════════════════════════════════════
if page == "🎫 Single Ticket":
    st.title("🔍 Single Ticket Analysis")
    st.markdown("Enter ticket details to check for priority mismatch.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        ticket_id   = st.text_input("Ticket ID", "TKT-001")
        subject     = st.text_input("Ticket Subject", 
                                     "Cannot access my account")
        description = st.text_area("Ticket Description",
                                    "I have been unable to login for 2 days. "
                                    "Getting error 403. This is urgent.",
                                    height=120)
        category    = st.selectbox("Issue Category",
                                    ["Technical", "Account", 
                                     "Billing", "Fraud", "General Inquiry"])

    with col2:
        assigned_priority = st.selectbox("Assigned Priority",
                                          ["Low", "Medium", "High", "Critical"])
        channel           = st.selectbox("Ticket Channel",
                                          ["Email", "Chat", "Web Form"])
        resolution_time   = st.number_input("Resolution Time (hours)",
                                             min_value=1.0,
                                             max_value=120.0,
                                             value=24.0)
        satisfaction      = st.number_input("Satisfaction Score (1-5)",
                                             min_value=1.0,
                                             max_value=5.0,
                                             value=3.0)

    st.markdown("---")

    if st.button("🔎 Analyze Ticket", type="primary"):
        urgency  = get_urgency_score(subject + ' ' + description)
        inferred = get_inferred_severity(resolution_time)
        keywords = [w for w in URGENT_WORDS
                    if w in (subject + ' ' + description).lower()][:3]

        text_input = build_input(
            subject, description, channel,
            resolution_time, category, urgency
        )
        pred, conf = predict(text_input)

        st.markdown("### Result")

        if pred == 1:
            a_num = priority_map.get(assigned_priority, 2)
            i_num = priority_map.get(inferred, 2)
            mtype = "Hidden Crisis" if i_num > a_num else "False Alarm"

            if mtype == "Hidden Crisis":
                st.error(f"⚠️ MISMATCH DETECTED — {mtype}")
            else:
                st.warning(f"⚠️ MISMATCH DETECTED — {mtype}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Assigned Priority", assigned_priority)
            col2.metric("Inferred Severity", inferred)
            col3.metric("Confidence",        f"{conf:.1%}")

            dossier = build_dossier(
                ticket_id, assigned_priority, inferred,
                resolution_time, satisfaction, conf, keywords
            )

            st.markdown("### Evidence Dossier")
            st.json(dossier)

        else:
            st.success(f"✅ Priority is CONSISTENT")
            col1, col2, col3 = st.columns(3)
            col1.metric("Assigned Priority", assigned_priority)
            col2.metric("Inferred Severity", inferred)
            col3.metric("Confidence",        f"{conf:.1%}")

# ══════════════════════════════════════════════════════════════
# PAGE 2: BATCH CSV
# ══════════════════════════════════════════════════════════════
elif page == "📦 Batch CSV":
    st.title("📦 Batch CSV Analysis")
    st.markdown("Upload a CSV file to analyze multiple tickets at once.")
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f"**Loaded {len(df)} tickets**")
        st.dataframe(df.head(5))

        if st.button("🚀 Run Batch Analysis", type="primary"):
            # Prepare data
            if 'combined_text' not in df.columns:
                df['combined_text'] = (
                    df['Ticket_Subject'].str.lower() +
                    ' [SEP] ' +
                    df['Ticket_Description'].str.lower()
                )

            if 'urgency_score' not in df.columns:
                df['urgency_score'] = df['combined_text'].apply(
                    get_urgency_score
                )

            if 'inferred_severity' not in df.columns:
                df['inferred_severity'] = df['Resolution_Time_Hours'].apply(
                    get_inferred_severity
                )

            # Build inputs
            inputs = []
            for _, row in df.iterrows():
                inputs.append(build_input(
                    str(row.get('Ticket_Subject', '')),
                    str(row.get('Ticket_Description', '')),
                    str(row.get('Ticket_Channel', 'Email')),
                    float(row.get('Resolution_Time_Hours', 24)),
                    str(row.get('Issue_Category', 'General Inquiry')),
                    int(row.get('urgency_score', 0))
                ))

            # Batch predict with progress bar
            BATCH_SIZE = 64
            all_preds, all_confs = [], []
            progress   = st.progress(0)
            status     = st.empty()

            for i in range(0, len(inputs), BATCH_SIZE):
                batch = inputs[i:i+BATCH_SIZE]
                enc   = tokenizer(
                    batch,
                    return_tensors='pt',
                    truncation=True,
                    max_length=256,
                    padding=True
                )
                with torch.no_grad():
                    logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                confs = probs[:, 1].numpy()
                preds = (confs >= THRESHOLD).astype(int)
                all_preds.extend(preds.tolist())
                all_confs.extend(confs.tolist())
                progress.progress(min((i + BATCH_SIZE) / len(inputs), 1.0))
                status.text(f"Processing {min(i+BATCH_SIZE, len(inputs))}"
                            f"/{len(inputs)} tickets...")

            df['predicted_mismatch'] = all_preds
            df['confidence']         = all_confs
            df['result']             = df['predicted_mismatch'].map(
                                           {0: 'Consistent', 1: 'Mismatch'}
                                       )

            status.text("Done!")

            # Show results
            total      = len(df)
            mismatches = sum(all_preds)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Tickets", total)
            col2.metric("Mismatches",    mismatches)
            col3.metric("Mismatch Rate", f"{mismatches/total:.1%}")

            st.markdown("### Results")
            result_df = df[['Ticket_ID', 'Ticket_Subject',
                             'Priority_Level', 'result',
                             'confidence']].copy()
            st.dataframe(result_df)

            # Download button
            csv = df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Results CSV",
                csv,
                "sia_predictions.csv",
                "text/csv"
            )

# ══════════════════════════════════════════════════════════════
# PAGE 3: DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 Priority Mismatch Dashboard")
    st.markdown("---")

    try:
    if os.path.exists('outputs/test_predictions.csv'):
        df = pd.read_csv('outputs/test_predictions.csv')
    else:
        st.warning("No predictions file found. Please use Batch CSV tab to generate predictions first.")
        st.stop()

    total      = len(df)
        mismatches = df['predicted_mismatch'].sum()
        consistent = total - mismatches
        avg_conf   = df['confidence'].mean()
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tickets",  f"{total:,}")
        col2.metric("Mismatches",     f"{mismatches:,}")
        col3.metric("Consistent",     f"{consistent:,}")
        col4.metric("Avg Confidence", f"{avg_conf:.1%}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        # Chart 1: Mismatch by Priority
        with col1:
            st.markdown("### Mismatch by Priority Level")
            priority_counts = df.groupby(
                ['Priority_Level', 'result']
            ).size().reset_index(name='count')
            fig1 = px.bar(
                priority_counts,
                x='Priority_Level',
                y='count',
                color='result',
                barmode='group',
                color_discrete_map={
                    'Mismatch':   '#ef4444',
                    'Consistent': '#22c55e'
                }
            )
            st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Mismatch by Channel
        with col2:
            st.markdown("### Mismatch by Channel")
            channel_counts = df.groupby(
                ['Ticket_Channel', 'result']
            ).size().reset_index(name='count')
            fig2 = px.bar(
                channel_counts,
                x='Ticket_Channel',
                y='count',
                color='result',
                barmode='group',
                color_discrete_map={
                    'Mismatch':   '#ef4444',
                    'Consistent': '#22c55e'
                }
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Chart 3: Confidence distribution
        st.markdown("### Confidence Score Distribution")
        fig3 = px.histogram(
            df, x='confidence',
            color='result',
            nbins=50,
            color_discrete_map={
                'Mismatch':   '#ef4444',
                'Consistent': '#22c55e'
            }
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Chart 4: Heatmap by Category and Channel
        if 'Issue_Category' in df.columns:
            st.markdown("### Severity Delta Heatmap")
            heat = df[df['result'] == 'Mismatch'].groupby(
                ['Issue_Category', 'Ticket_Channel']
            ).size().reset_index(name='count')
            fig4 = px.density_heatmap(
                heat,
                x='Ticket_Channel',
                y='Issue_Category',
                z='count',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig4, use_container_width=True)

    except FileNotFoundError:
        st.warning("No predictions found. Run predict.py first or "
                   "use Batch CSV to generate predictions.")
        st.code("python predict.py tickets_clean.csv "
                "outputs/test_predictions.csv")
