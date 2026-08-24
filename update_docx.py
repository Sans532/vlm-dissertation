import re
from docx import Document
import sys

def parse_markdown():
    with open("DISSERTATION_RESULTS.md", "r") as f:
        content = f.read()

    # Extract Qwen2.5-VL-7B section
    start_marker = "### 1.2 Qwen2.5-VL-7B"
    end_marker = "### 1.3 VideoLLaVA"
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("Start marker not found.")
        sys.exit(1)
        
    end_idx = content.find(end_marker, start_idx)
    if end_idx == -1:
        end_idx = len(content)

    qwen_section = content[start_idx:end_idx]

    results = []

    # Parse markdown tables
    # Find all table rows
    lines = qwen_section.split('\n')
    in_table = False
    headers = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('|') and not line.startswith('|-'):
            cells = [cell.strip() for cell in line.split('|')][1:-1]
            if not in_table:
                headers = cells
                in_table = True
            elif line.startswith('|---'):
                continue
            else:
                row_data = dict(zip(headers, cells))
                if 'Overall' in row_data:
                    # Map to the format we need
                    prompt = row_data.get('Prompt', 'Binary')
                    nframes = row_data.get('Frames', '8')
                    trim = row_data.get('Trim', 'entire')
                    view = row_data.get('View', 'ego/exo')
                    overall = row_data.get('Overall', '')
                    acc = re.search(r'=\s*([\d\.]+%?)', overall)
                    accuracy = acc.group(1) if acc else overall
                    
                    # Recall/class acc
                    novice_acc = row_data.get('Novice acc', '0%')
                    other_acc = row_data.get('Other classes', '')
                    expert_acc = row_data.get('Expert acc', '')
                    recall = f"Novice: {novice_acc}"
                    if other_acc:
                        recall += f", Others: {other_acc}"
                    if expert_acc:
                        recall += f", Expert: {expert_acc}"
                        
                    label_dist = row_data.get('Predicted counts', '')
                    
                    # Precision is tricky to calculate generally, we'll leave it empty for collapsed
                    results.append({
                        'Prompts': prompt,
                        'nframes': nframes,
                        'Trim': trim,
                        'view': view,
                        'accuracy': accuracy,
                        'precision': 'N/A (Collapsed)',
                        'recall': recall,
                        'label distribution': label_dist
                    })
        else:
            in_table = False

    # Parse bulleted items
    # Example block:
    # **Fourclass — 8 frames, entire, exo**
    # - Overall: 25/100 = **25.0%**
    # - Novice: 22/25 = 88.0% · Early Expert: 0/25 · Intermediate Expert: 3/25 = 12.0% · Late Expert: 0/25
    # - Predicted counts: Novice 83, Unknown 9, Intermediate Expert 8
    
    current_item = {}
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('**') and '—' in line:
            m = re.match(r'\*\*(.*?)\s*—\s*(.*?)\*\*', line)
            if m:
                prompt = m.group(1).strip()
                meta = m.group(2).strip()
                
                nframes = "16" if "16 frames" in meta else "8" if "8 frames" in meta else "64" if "64 frames" in meta else "native"
                if "best-exo" in meta: nframes = "8"
                trim = "trimmed" if "trimmed" in meta else "entire"
                view = "ego" if "ego" in meta else "exo"
                if "best-exo" in meta: view = "best-exo"
                if "n=389" in meta: prompt += " (n=389)"
                
                current_item = {
                    'Prompts': prompt,
                    'nframes': nframes,
                    'Trim': trim,
                    'view': view,
                    'accuracy': '',
                    'precision': '',
                    'recall': '',
                    'label distribution': ''
                }
                
                # Look ahead for details
                j = i + 1
                while j < len(lines) and (lines[j].strip().startswith('-') or lines[j].strip().startswith('Predicted counts:') or lines[j].strip().startswith('ego:') or lines[j].strip().startswith('exo:')):
                    detail = lines[j].strip()
                    if detail.startswith('- Overall:'):
                        acc_m = re.search(r'=\s*\*\*?([\d\.]+%)', detail)
                        if acc_m:
                            current_item['accuracy'] = acc_m.group(1)
                    elif 'Overall' in detail and '=' in detail:
                        acc_m = re.search(r'Overall.*=\s*\*\*?([\d\.]+%)', detail)
                        if acc_m:
                            current_item['accuracy'] = acc_m.group(1)
                            
                    elif 'Novice:' in detail or 'Early Expert:' in detail:
                        # Extract recall string by removing bullets
                        current_item['recall'] = detail.lstrip('- ').strip()
                    elif 'Predicted counts:' in detail:
                        current_item['label distribution'] = detail.split('Predicted counts:')[1].strip()
                    
                    j += 1
                
                if current_item['accuracy']:
                    results.append(current_item)
                    current_item = {}

    return results

def write_to_docx(results):
    document = Document("Qwen2.5-VL-7B_Climbing_results.docx")
    table = document.tables[0]
    
    for res in results:
        cells = table.add_row().cells
        cells[0].text = res.get('Prompts', '')
        cells[1].text = res.get('nframes', '')
        cells[2].text = res.get('Trim', '')
        cells[3].text = res.get('view', '')
        cells[4].text = res.get('accuracy', '')
        cells[5].text = res.get('precision', '') # Not explicitly in markdown
        cells[6].text = res.get('recall', '')
        cells[7].text = res.get('label distribution', '')

    document.save("Qwen2.5-VL-7B_Climbing_results.docx")

if __name__ == "__main__":
    results = parse_markdown()
    write_to_docx(results)
    print(f"Added {len(results)} rows to the docx.")
