import pandas as pd
import numpy as np
import json

# Load predictions
df = pd.read_csv(r'C:\Users\rawal\OneDrive\Desktop\mars\outputs\predictions.csv')

print(f"Total tickets: {len(df)}")
print(f"Mismatches: {df['predicted_mismatch'].sum()}")

# Only process mismatched tickets
mismatched = df[df['predicted_mismatch'] == 1].copy().reset_index(drop=True)
print(f"Processing {len(mismatched)} dossiers...")

# Priority mappings
priority_map  = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
severity_map  = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Critical'}

# Urgent keywords for evidence
URGENT_WORDS = [
    'urgent', 'asap', 'immediately', 'emergency', 'critical',
    'cannot access', 'not working', 'broken', 'error', 'failed',
    'crash', 'outage', 'down', 'blocked', 'fraud',
    'unauthorized', 'hacked', 'data loss', 'stolen', 'locked'
]

def get_keywords(text):
    text  = str(text).lower()
    found = [w for w in URGENT_WORDS if w in text]
    return found[:3]

def get_mismatch_type(assigned, inferred):
    a = priority_map.get(str(assigned), 2)
    i = priority_map.get(str(inferred), 2)
    if i > a:
        return "Hidden Crisis"
    else:
        return "False Alarm"

def get_constraint_analysis(row):
    assigned = str(row['Priority_Level'])
    inferred = str(row['inferred_severity'])
    rt       = round(float(row['Resolution_Time_Hours']), 1)
    sat      = round(float(row['Satisfaction_Score']), 1)
    keywords = get_keywords(row['combined_text'])
    kw_str   = ', '.join(keywords) if keywords else 'none detected'

    a_num = priority_map.get(assigned, 2)
    i_num = priority_map.get(inferred, 2)

    if i_num > a_num:
        return (
            f"The ticket was assigned '{assigned}' priority but severity "
            f"signals indicate it is actually '{inferred}'. "
            f"The issue was resolved in {rt} hours with urgency indicators "
            f"'{kw_str}' found in the ticket text. "
            f"Customer satisfaction of {sat}/5 further confirms "
            f"this was a more serious issue than labeled."
        )
    else:
        return (
            f"The ticket was assigned '{assigned}' priority but signals "
            f"suggest actual severity is '{inferred}'. "
            f"Despite the high assigned priority, resolution took {rt} hours "
            f"and satisfaction score was {sat}/5, "
            f"indicating the issue was less critical than labeled."
        )

# Generate dossiers
dossiers = []

for idx, row in mismatched.iterrows():
    assigned    = str(row['Priority_Level'])
    inferred    = str(row['inferred_severity'])
    assigned_n  = priority_map.get(assigned, 2)
    inferred_n  = priority_map.get(inferred, 2)

    # Build evidence list
    evidence = []

    # Evidence 1: Keywords from text
    keywords = get_keywords(row['combined_text'])
    if keywords:
        evidence.append({
            "signal":  "keyword",
            "value":   ", ".join(keywords),
            "weight":  str(int(row.get('urgency_score', len(keywords))))
        })
    else:
        evidence.append({
            "signal":  "keyword",
            "value":   "no urgent keywords detected",
            "weight":  "0"
        })

    # Evidence 2: Resolution time
    rt = float(row['Resolution_Time_Hours'])
    evidence.append({
        "signal":         "resolution_time",
        "value":          str(round(rt, 1)),
        "interpretation": "fast resolution suggests high severity"
                          if rt <= 24
                          else "slow resolution suggests low severity"
    })

    # Evidence 3: Satisfaction score
    sat = float(row['Satisfaction_Score'])
    evidence.append({
        "signal":         "satisfaction_score",
        "value":          str(round(sat, 1)),
        "interpretation": "low satisfaction confirms serious issue"
                          if sat <= 3
                          else "high satisfaction suggests minor issue"
    })

    # Build severity delta
    severity_delta = f"{assigned} → {inferred}"

    # Build complete dossier
    dossier = {
        "ticket_id":           str(row.get('Ticket_ID', idx)),
        "assigned_priority":   assigned,
        "inferred_severity":   inferred,
        "mismatch_type":       get_mismatch_type(assigned, inferred),
        "severity_delta":      severity_delta,
        "feature_evidence":    evidence,
        "constraint_analysis": get_constraint_analysis(row),
        "confidence":          round(float(row['confidence']), 4)
    }
    dossiers.append(dossier)

# Save dossiers
output_path = r'C:\Users\rawal\OneDrive\Desktop\mars\outputs\dossiers.json'
with open(output_path, 'w') as f:
    json.dump(dossiers, f, indent=2)

print(f"\nGenerated {len(dossiers)} dossiers")
print(f"Saved to: {output_path}")

# Summary stats
hidden     = [d for d in dossiers if d['mismatch_type'] == 'Hidden Crisis']
false_alarm = [d for d in dossiers if d['mismatch_type'] == 'False Alarm']
avg_conf   = np.mean([d['confidence'] for d in dossiers])

print(f"\n=== DOSSIER SUMMARY ===")
print(f"Hidden Crisis : {len(hidden)}")
print(f"False Alarm   : {len(false_alarm)}")
print(f"Avg Confidence: {avg_conf:.2%}")

# Print 1 Hidden Crisis sample
print("\n=== SAMPLE: Hidden Crisis ===")
if hidden:
    print(json.dumps(hidden[0], indent=2))

# Print 1 False Alarm sample
print("\n=== SAMPLE: False Alarm ===")
if false_alarm:
    print(json.dumps(false_alarm[0], indent=2))