import pandas as pd
import re

# Read CSV
df = pd.read_csv('/Users/sans/Documents/Msc courses/vlm-dissertation/statistics/master_results_summary.csv')

# We want to extract accuracies from txt files and compare
def parse_txt(filepath):
    data = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    current_model = None
    current_prompt = None
    current_frames = None
    current_trim = None
    
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if 'QWEN' in line or 'VideoLLaVa' in line or 'VideoLLaVA' in line or 'Gemini' in line:
            if 'QWEN' in line: current_model = 'qwen'
            elif 'VideoLLaVa' in line or 'VideoLLaVA' in line: current_model = 'videollava'
            elif 'Gemini' in line: current_model = 'gemini'
            
            if 'TRIMMED' in line: current_trim = 'trimmed'
            else: current_trim = 'entire'
            
            if '16 frames' in line: current_frames = '16'
            else: current_frames = '8' if current_model != 'gemini' else '' # default for gemini
        
        # Match "Binary, Entire, 8f" or "Binary"
        elif 'Binary' in line or 'Fourclass' in line or 'Structured' in line or 'Reasoning' in line:
            if line.startswith('Binary'): current_prompt = 'binary'
            elif line.startswith('Fourclass'): current_prompt = 'fourclass'
            elif line.startswith('Structured'): current_prompt = 'structured'
            elif line.startswith('Reasoning'): current_prompt = 'reasoning'
            
            if '16f' in line: current_frames = '16'
            elif '8f' in line: current_frames = '8'
            
            if 'Trimmed' in line: current_trim = 'trimmed'
            elif 'Entire' in line: current_trim = 'entire'
            
            # Read EXO and EGO
            if 'gemini' in current_model:
                # Need to handle gemini
                pass
                
        i += 1

# Actually it's easier to just do a quick manual check of a few things and print the differences
print("Parsing differences...")
