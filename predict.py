import pandas as pd
import numpy as np
import torch
import json
import sys
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ────────────────────────────────────────────────────
MODEL_PATH  = r'C:\Users\rawal\OneDrive\Desktop\mars\models\sia_classifier_final'
THRESHOLD   = 0.46
URGENT_WORDS = [
    'urgent', 'asap', 'immediately', 'emergency', 'critical',
    'cannot access', 'not working', 'broken', 'error', 'failed',
    'crash', 'outage', 'down', 'blocked', 'fraud',
    'unauthorized', 'hacked', 'data loss', 'stolen', 'locked'
]
priority_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
severity_map = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Critical'}

# ── Load model ────────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded!")

# ── Build input text ──────────────────────────────────────────
def build_input(row):
    channel  = str(row.get('Ticket_Channel', ''))
    rt       = str(round(float(row.get('Resolution_Time_Hours', 0)), 1))
    urgency  = str(int(row.get('urgency_score', 0)))
    cat      = str(row.get('Issue_Category', ''))
    text     = str(row.get('combined_text',
                   str(row.get('Ticket_Subject', '')) + ' [SEP] ' +
                   str(row.get('Ticket_Description', ''))))
    return (f"{text} "
            f"[CHANNEL] {channel} "
            f"[CAT] {cat} "
            f"[RT] {rt} "
            f"[URGENCY] {urgency}")

# ── Predict single ticket ─────────────────────────────────────
def predict_single(row):
    text = build_input(row)
    enc  = tokenizer(
        text,
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

# ── Build dossier ─────────────────────────────────────────────
def build_dossier(row, confidence):
    assigned   = str(row.get('Priority_Level', 'Unknown'))
    inferred   = str(row.get('inferred_severity', 'Unknown'))
    rt         = float(row.get('Resolution_Time_Hours', 0))
    sat        = float(row.get('Satisfaction_Score', 0))
    text       = str(row.get('combined_text', '')).lower()

    # Keywords
    keywords = [w for w in URGENT_WORDS if w in text][:3]
    evidence = []
    evidence.append({
        "signal":  "keyword",
        "value":   ", ".join(keywords) if keywords else "none detected",
        "weight":  str(len(keywords))
    })
    evidence.append({
        "signal":         "resolution_time",
        "value":          str(round(rt, 1)),
        "interpretation": "fast — suggests high severity" if rt <= 24
                          else "slow — suggests low severity"
    })
    evidence.append({
        "signal":         "satisfaction_score",
        "value":          str(round(sat, 1)),
        "interpretation": "low — confirms serious issue" if sat <= 3
                          else "high — suggests minor issue"
    })

    # Mismatch type
    a_num = priority_map.get(assigned, 2)
    i_num = priority_map.get(inferred, 2)
    mismatch_type  = "Hidden Crisis" if i_num > a_num else "False Alarm"
    severity_delta = f"{assigned} → {inferred}"

    # Constraint analysis
    kw_str = ", ".join(keywords) if keywords else "none detected"
    if i_num > a_num:
        analysis = (
            f"Ticket assigned '{assigned}' but signals indicate '{inferred}'. "
            f"Resolved in {round(rt,1)}h with keywords: {kw_str}. "
            f"Satisfaction {round(sat,1)}/5 confirms severity was underestimated."
        )
    else:
        analysis = (
            f"Ticket assigned '{assigned}' but signals indicate '{inferred}'. "
            f"Resolution took {round(rt,1)}h and satisfaction was {round(sat,1)}/5, "
            f"suggesting severity was overestimated."
        )

    return {
        "ticket_id":           str(row.get('Ticket_ID', 'N/A')),
        "assigned_priority":   assigned,
        "inferred_severity":   inferred,
        "mismatch_type":       mismatch_type,
        "severity_delta":      severity_delta,
        "feature_evidence":    evidence,
        "constraint_analysis": analysis,
        "confidence":          confidence
    }

# ── Main function ─────────────────────────────────────────────
def predict_csv(input_path, output_path):
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    # Add combined text if not present
    if 'combined_text' not in df.columns:
        df['Ticket_Subject']     = df['Ticket_Subject'].str.lower().str.strip()
        df['Ticket_Description'] = df['Ticket_Description'].str.lower().str.strip()
        df['combined_text']      = (df['Ticket_Subject'] +
                                    ' [SEP] ' +
                                    df['Ticket_Description'])

    # Add urgency score if not present
    if 'urgency_score' not in df.columns:
        df['urgency_score'] = df['combined_text'].apply(
            lambda t: min(sum(1 for w in URGENT_WORDS
                              if w in str(t).lower()), 10)
        )

    # Add inferred severity if not present
    if 'inferred_severity' not in df.columns:
        def rt_to_inferred(rt):
            if rt <= 18:   return 'Critical'
            elif rt <= 34: return 'High'
            elif rt <= 44: return 'Medium'
            else:          return 'Low'
        df['inferred_severity'] = df['Resolution_Time_Hours'].apply(rt_to_inferred)

    # Run predictions
    predictions, confidences, dossiers = [], [], []
    print(f"Running predictions on {len(df)} tickets...")

    for idx, row in df.iterrows():
        pred, conf = predict_single(row)
        predictions.append(pred)
        confidences.append(conf)
        if pred == 1:
            dossiers.append(build_dossier(row, conf))

    df['predicted_mismatch'] = predictions
    df['confidence']         = confidences
    df['result']             = df['predicted_mismatch'].map(
                                   {0: 'Consistent', 1: 'Mismatch'}
                               )

    # Save predictions CSV
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

    # Save dossiers JSON
    dossier_path = output_path.replace('.csv', '_dossiers.json')
    with open(dossier_path, 'w') as f:
        json.dump(dossiers, f, indent=2)
    print(f"Dossiers saved to: {dossier_path}")

    # Summary
    total     = len(df)
    mismatches = sum(predictions)
    print(f"\n=== RESULTS ===")
    print(f"Total tickets : {total}")
    print(f"Mismatches    : {mismatches} ({mismatches/total:.1%})")
    print(f"Consistent    : {total - mismatches}")
    print(f"Dossiers      : {len(dossiers)}")

# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python predict.py input.csv output.csv")
        print("Example: python predict.py data/tickets.csv outputs/results.csv")
        sys.exit(1)

    input_csv  = sys.argv[1]
    output_csv = sys.argv[2]
    predict_csv(input_csv, output_csv)