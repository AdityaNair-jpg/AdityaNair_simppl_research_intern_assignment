import json
import pandas as pd

def get_column_names(filename="data.jsonl", num_lines=5):  # Adjust num_lines as needed
    columns = set()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for _ in range(num_lines):
                line = f.readline()
                if not line:
                    break  # End of file
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):  # Ensure it's a dictionary
                        columns.update(data.keys())
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                    continue
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        return []
    return list(columns)

column_names = get_column_names()
print("Column names:", column_names)

#Optional: Using pandas to read the jsonl file
try:
    df = pd.read_json('data.jsonl', lines=True)
    print(df.columns)
except Exception as e:
    print(f"Error reading JSONL file: {e}")