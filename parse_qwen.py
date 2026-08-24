import re
import json

with open("DISSERTATION_RESULTS.md", "r") as f:
    content = f.read()

# Extract Qwen2.5-VL-7B section
start_marker = "### 1.2 Qwen2.5-VL-7B"
end_marker = "### 1.3 VideoLLaVA"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

qwen_section = content[start_idx:end_idx]

results = []

# Parse collapsed tables
# Table format: | Frames | Trim | View | Overall | Novice acc | Expert acc | Predicted counts | File |
# or | View | Overall | Novice acc | Other classes | Predicted counts | File |
# or | Prompt | Frames | Trim | Overall | Novice acc | Other classes | Predicted counts | File |

# We can also parse the "Kept" list items which are formatted like:
# **Fourclass — 8 frames, entire, ego**
# - Overall: 24/100 = **24.0%**
# - Novice: 23/25 = 92.0% · Early Expert: 0/25 · Intermediate Expert: 1/25 = 4.0% · Late Expert: 0/25
# - Predicted counts: Novice 84, Unknown 9, Intermediate Expert 7

def parse_kept():
    blocks = re.split(r'\*\*([A-Z][a-z]+) — (.*?)\*\*', qwen_section)
    for i in range(1, len(blocks), 2):
        prompt = blocks[i]
        meta = blocks[i+1]
        body = blocks[i+2]
        
        # skip if this is not a result block
        if "- Overall:" not in body: continue
        
        # parse meta for nframes, trim, view
        nframes = "16" if "16 frames" in meta else "8" if "8 frames" in meta else "64" if "64 frames" in meta else "native"
        if "best-exo" in meta: nframes = "8" # from text
        trim = "trimmed" if "trimmed" in meta else "entire"
        view = "ego" if "ego" in meta else "exo"
        if "best-exo" in meta: view = "best-exo"
        if "n=389" in meta: prompt += " (n=389)"
        
        # overall
        m_overall = re.search(r'Overall:\s*([\d\.]+)/(\d+)\s*=\s*\*\*([\d\.]+)%\*\*', body)
        if not m_overall:
            m_overall = re.search(r'Overall:\s*([\d\.]+)/(\d+)\s*=\s*([\d\.]+)%', body)
        if not m_overall:
            m_overall = re.search(r'Overall\s*([\d\.]+)/(\d+)\s*=\s*\*\*([\d\.]+)%\*\*', body)
        
        accuracy = m_overall.group(3) + "%" if m_overall else ""
        
        # recalls
        recalls = {}
        # Novice: 23/25 = 92.0%
        for m_class in re.finditer(r'([A-Za-z\s]+):\s*(\d+)/(\d+)(?:\s*=\s*([\d\.]+)%)?', body):
            cls = m_class.group(1).strip()
            if cls in ['Overall', 'Predicted counts', 'File']: continue
            rec = m_class.group(4)
            if rec: recalls[cls] = rec + "%"
            else:
                if m_class.group(2) == '0': recalls[cls] = "0.0%"
        
        # predicted counts
        pred_counts = {}
        m_pred = re.search(r'Predicted counts:\s*(.*?)(?=\n- File|\n$)', body, re.DOTALL)
        if m_pred:
            preds_str = m_pred.group(1)
            for item in preds_str.split(','):
                item = item.strip()
                if item:
                    parts = item.rsplit(' ', 1)
                    if len(parts) == 2:
                        pred_counts[parts[0]] = parts[1]
        
        results.append({
            "Prompt": prompt,
            "nframes": nframes,
            "Trim": trim,
            "view": view,
            "accuracy": accuracy,
            "recall": recalls,
            "label_distribution": pred_counts
        })

parse_kept()
print(json.dumps(results, indent=2))
