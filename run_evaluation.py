import json
import sqlite3
import os
import time
import ast
import re
from openai import OpenAI

# 1. Configurazione LLM
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" 
)
NOME_MODELLO = "llama3" 

# 2. Accesso al Database e Estrazione Schema
def get_db_connection(db_id):
    db_path = os.path.join("data", "database", db_id, f"{db_id}.sqlite")
    return sqlite3.connect(db_path)

def get_schema(db_id, oracle_tables):
    conn = get_db_connection(db_id)
    cursor = conn.cursor()
    
    schema_text = ""
    for table_name in oracle_tables:
        schema_text += f"\nTable: {table_name}\n"
        
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns_with_types = [f"{col[1]} ({col[2]})" for col in cursor.fetchall()]
        schema_text += "Columns: " + ", ".join(columns_with_types) + "\n"
        
        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        fk_info = cursor.fetchall()
        if fk_info:
            fks = []
            for fk in fk_info:
                fks.append(f"{fk[3]} -> {fk[2]}.{fk[4]}")
            schema_text += "Foreign Keys: " + ", ".join(fks) + "\n"
            
    conn.close()
    return schema_text

def execute_query(db_id, query):
    try:
        conn = get_db_connection(db_id)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return [list(r) for r in results], None
    except Exception as e:
        return None, str(e)

# 3. Identificazione e Serializzazione Tabelle
def get_oracle_tables(db_id, ground_truth_sql):
    conn = get_db_connection(db_id)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = [row[0].lower() for row in cursor.fetchall()]
    conn.close()
    
    sql_lower = ground_truth_sql.lower()
    oracle_tables = [t for t in all_tables if re.search(rf'\b{re.escape(t)}\b', sql_lower)]
    return oracle_tables if oracle_tables else all_tables

def serialize_tables(db_id, table_names):
    testo_tabelle = ""
    conn = get_db_connection(db_id)
    cursor = conn.cursor()
    for t in table_names:
        cursor.execute(f"PRAGMA table_info({t});")
        cols = [c[1] for c in cursor.fetchall()]
        cursor.execute(f"SELECT * FROM {t};")
        rows = cursor.fetchall()
        testo_tabelle += f"\n### Tabella: {t} ###\n"
        testo_tabelle += " | ".join(cols) + "\n"
        testo_tabelle += "-" * 40 + "\n"
        for r in rows:
            testo_tabelle += " | ".join([str(val) for val in r]) + "\n"
    conn.close()
    return testo_tabelle

def serialize_tables_csv(db_id, table_names):
    testo_tabelle = ""
    conn = get_db_connection(db_id)
    cursor = conn.cursor()
    for t in table_names:
        cursor.execute(f"PRAGMA table_info({t});")
        cols = [c[1] for c in cursor.fetchall()]
        cursor.execute(f"SELECT * FROM {t};")
        rows = cursor.fetchall()
        testo_tabelle += f"\nTable: {t}\n"
        testo_tabelle += ",".join(cols) + "\n"
        for r in rows:
            testo_tabelle += ",".join([str(val) for val in r]) + "\n"
    conn.close()
    return testo_tabelle

# 4. Pipeline di Generazione (Text-to-SQL e Table QA)
def safe_parse_json(clean_json):
    try:
        data = json.loads(clean_json)
        if isinstance(data, list) and len(data) > 0 and not isinstance(data[0], list):
            data = [[item] for item in data]
        righe = len(data) if isinstance(data, list) else 0
        return data, False, righe
    except Exception:
        try:
            data = ast.literal_eval(clean_json)
            if isinstance(data, list):
                if len(data) > 0 and not isinstance(data[0], list):
                    data = [[item] for item in data]
                return data, False, len(data)
        except Exception:
            pass
            
        matches = re.findall(r'\[\s*(?:"[^"]*"|\'[^\']*\'|[^\[\]])+?\s*\]', clean_json)
        recovered_data = []
        for m in matches:
            try:
                m_clean = m.replace("'", '"')
                row = json.loads(m_clean)
                if isinstance(row, list):
                    recovered_data.append(row)
            except:
                continue
        if recovered_data:
            return recovered_data, True, len(recovered_data)
        return None, True, 0

def pipeline_text_to_sql(question, schema):
    prompt = f"Schema:\n{schema}\n\nQuestion: {question}\n\nWrite the SQL query for SQLite. Return ONLY the query and nothing else."
    response = client.chat.completions.create(
        model=NOME_MODELLO, messages=[{"role": "user", "content": prompt}], temperature=0.0
    )
    raw_text = response.choices[0].message.content.strip()
    match = re.search(r'\bSELECT\b', raw_text, re.IGNORECASE)
    if not match:
        return raw_text.replace("```sql", "").replace("```", "").strip()
    
    text_to_parse = raw_text[match.start():]
    paren_count = 0
    in_single_quote = False
    in_double_quote = False
    for i, char in enumerate(text_to_parse):
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        if not in_single_quote and not in_double_quote:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ';' and paren_count == 0:
                return text_to_parse[:i+1].strip()
    sql_clean = text_to_parse.strip()
    if not sql_clean.endswith(";"):
        sql_clean += ";"
    return sql_clean

def pipeline_table_qa(question, table_text):
    prompt = f"""You are an intelligent data analyst.
Analyze the provided tables and perform the necessary calculations (such as counts, averages, sums, or filters) to answer the question.
You must return ONLY the final answer and NOT the intermediate table rows.

EXAMPLE:
Tables:
### Table: employees ###
name | salary
----------------------------------------
Anna | 2000
Luca | 3000

Question: What is the average salary of the employees?
Expected Result: [["2500"]]
END OF EXAMPLE.

Now answer this question:
Tables:
{table_text}

Question: {question}

Respond by returning EXCLUSIVELY a two-dimensional JSON array (e.g. [["Name", 30]]). No textual introduction."""
    response = client.chat.completions.create(
        model=NOME_MODELLO, messages=[{"role": "user", "content": prompt}], temperature=0.0
    )
    raw_text = response.choices[0].message.content.strip()
    json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
    clean_json = json_match.group(0).strip() if json_match else raw_text.replace("```json", "").replace("```", "").strip()
    
    data, is_truncated, recovered_rows = safe_parse_json(clean_json)
    
    error_msg = None
    if is_truncated:
        error_msg = f"JSON troncato. Recuperate {recovered_rows} righe valide." if data else "JSON troncato e irrecuperabile."
    return data, clean_json, error_msg, is_truncated, recovered_rows

# 5. Calcolo Metriche Qatch
def calcola_qatch_metrics(gt_data, pred_data):
    if not gt_data and not pred_data:
        return {"cell_precision": 1.0, "cell_recall": 1.0, "tuple_cardinality": 1.0, "tuple_order": 1.0, "exact_match": 1.0}
    if not pred_data or not gt_data:
        return {"cell_precision": 0.0, "cell_recall": 0.0, "tuple_cardinality": 0.0, "tuple_order": 0.0, "exact_match": 0.0}
    
    def normalize_row(row):
        return tuple(str(c).strip().lower() for c in (row if isinstance(row, (list, tuple)) else [row]))

    gt_norm = [normalize_row(row) for row in gt_data]
    pred_norm = [normalize_row(row) for row in pred_data]
    
    exact = 1.0 if gt_norm == pred_norm else 0.0
    cardinality = 1.0 if len(gt_norm) == len(pred_norm) else 0.0
    
    # Tuple Order posizionale
    lunghezza_max = max(len(gt_norm), len(pred_norm))
    righe_in_posizione = sum(1 for i in range(min(len(gt_norm), len(pred_norm))) if gt_norm[i] == pred_norm[i])
    order = round(righe_in_posizione / lunghezza_max, 2) if lunghezza_max > 0 else 0.0

    def flatten(table):
        return [str(c).strip().lower() for row in table for c in (row if isinstance(row, (list, tuple)) else [row])]
    
    gt_cells = flatten(gt_data)
    pred_cells = flatten(pred_data)
    matches = 0
    temp_gt = gt_cells.copy()
    for c in pred_cells:
        if c in temp_gt:
            matches += 1
            temp_gt.remove(c)
    
    prec = matches / len(pred_cells) if pred_cells else 0.0
    rec = matches / len(gt_cells) if gt_cells else 0.0
    
    return {
        "cell_precision": round(prec, 2), "cell_recall": round(rec, 2),
        "tuple_cardinality": cardinality, "tuple_order": order, "exact_match": exact
    }

# 6. Esecuzione Valutazione Principale
def run_evaluation():
    with open("data/dev.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    database_scelti = ["concert_singer", "pets_1"]
    test_set = [sample for sample in dataset if sample["db_id"] in database_scelti]
    results = []
    empty_metrics = {"cell_precision": 0.0, "cell_recall": 0.0, "tuple_cardinality": 0.0, "tuple_order": 0.0, "exact_match": 0.0}
    
    for i, sample in enumerate(test_set):
        print(f"\n--- Elaborazione {i+1}/{len(test_set)} ---")
        db_id, question, gt_sql = sample['db_id'], sample['question'], sample['query']
        gt_data, _ = execute_query(db_id, gt_sql)
        oracle_tabs = get_oracle_tables(db_id, gt_sql)

        # SQL Pipeline
        start = time.time()
        sql_gen = pipeline_text_to_sql(question, get_schema(db_id, oracle_tabs))
        res_sql, err_sql = execute_query(db_id, sql_gen)
        m_sql = calcola_qatch_metrics(gt_data, res_sql)
        time_sql = round(time.time() - start, 2)

        # Markdown Table QA Pipeline
        start = time.time()
        res_qa_md, raw_qa_md, err_qa_md, is_trunc_md, rec_rows_md = pipeline_table_qa(question, serialize_tables(db_id, oracle_tabs))
        m_qa_md = calcola_qatch_metrics(gt_data, res_qa_md) if res_qa_md else empty_metrics.copy()
        time_qa_md = round(time.time() - start, 2)
        
        # CSV Table QA Pipeline
        start = time.time()
        res_qa_csv, raw_qa_csv, err_qa_csv, is_trunc_csv, rec_rows_csv = pipeline_table_qa(question, serialize_tables_csv(db_id, oracle_tabs))
        m_qa_csv = calcola_qatch_metrics(gt_data, res_qa_csv) if res_qa_csv else empty_metrics.copy()
        time_qa_csv = round(time.time() - start, 2)

        results.append({
            "id": i, "db_id": db_id, "question": question, "gt_data": gt_data,
            "text_to_sql": {"generated_query": sql_gen, "data": res_sql, "metrics": m_sql, "error": err_sql, "time_seconds": time_sql},
            "table_qa_markdown": {"data": res_qa_md, "metrics": m_qa_md, "time_seconds": time_qa_md},
            "table_qa_csv": {"data": res_qa_csv, "metrics": m_qa_csv, "time_seconds": time_qa_csv}
        })
        print(f"SQL: P={m_sql['cell_precision']} | QA(MD): P={m_qa_md['cell_precision']} | QA(CSV): P={m_qa_csv['cell_precision']}")

    with open("evaluation_log.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    run_evaluation()
