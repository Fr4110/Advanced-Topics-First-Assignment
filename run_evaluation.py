import json
import sqlite3
import os
import time
import re
from openai import OpenAI

#configurazione ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" 
)
NOME_MODELLO = "llama3" 

#accesso al database
def get_db_connection(db_id):
    db_path = os.path.join("data", "database", db_id, f"{db_id}.sqlite")
    return sqlite3.connect(db_path)

def get_schema(db_id, oracle_tables):
    conn = get_db_connection(db_id)
    cursor = conn.cursor()
    
    schema_text = ""
    for table_name in oracle_tables:
        schema_text += f"\nTable: {table_name}\nColumns: "
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [col[1] for col in cursor.fetchall()]
        schema_text += ", ".join(columns) + "\n"
        
    conn.close()
    return schema_text

def execute_query(db_id, query):
    try:
        conn = get_db_connection(db_id)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        # Convertiamo in lista di liste per compatibilità con il JSON di Llama
        return [list(r) for r in results], None
    except Exception as e:
        return None, str(e)

#serializzazione
def get_oracle_tables(db_id, ground_truth_sql):
    conn = get_db_connection(db_id)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = [row[0].lower() for row in cursor.fetchall()]
    conn.close()
    
    sql_lower = ground_truth_sql.lower()
    oracle_tables = [t for t in all_tables if re.search(rf'\b{t}\b', sql_lower)]
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

#pipeline
def pipeline_text_to_sql(question, schema):
    prompt = f"Schema:\n{schema}\n\nDomanda: {question}\n\nScrivi la query SQL per SQLite. Restituisci SOLO la query e nient'altro."
    response = client.chat.completions.create(
        model=NOME_MODELLO, messages=[{"role": "user", "content": prompt}], temperature=0.0
    )
    raw_text = response.choices[0].message.content.strip()
    
    sql_match = re.search(r'(SELECT.*?)(;|\Z)', raw_text, re.IGNORECASE | re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip() + ";"
    return raw_text.replace("```sql", "").replace("```", "").strip()

def pipeline_table_qa(question, table_text):
    prompt = f"""Tabelle:\n{table_text}\n\nDomanda: {question}\n\nRispondi restituendo ESCLUSIVAMENTE un array JSON bidimensionale (es: [["Mario", 30], ["Luigi", 25]]). Nessuna introduzione testuale."""
    response = client.chat.completions.create(
        model=NOME_MODELLO, messages=[{"role": "user", "content": prompt}], temperature=0.0
    )
    raw_text = response.choices[0].message.content.strip()
    
    json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
    clean_json = json_match.group(0).strip() if json_match else raw_text.replace("```json", "").replace("```", "").strip()
        
    try:
        data = json.loads(clean_json)
        if isinstance(data, list) and len(data) > 0 and not isinstance(data[0], list):
            data = [[item] for item in data]
        return data, clean_json, None
    except Exception as e:
        return None, clean_json, str(e)

#metriche
def calcola_qatch_metrics(gt_data, pred_data):
    if not gt_data and not pred_data:
        return {"cell_precision": 1.0, "cell_recall": 1.0, "tuple_cardinality": 1.0, "tuple_order": 1.0, "exact_match": 1.0}
    if not pred_data or not gt_data:
        return {"cell_precision": 0.0, "cell_recall": 0.0, "tuple_cardinality": 0.0, "tuple_order": 0.0, "exact_match": 0.0}

    exact = 1.0 if gt_data == pred_data else 0.0
    cardinality = 1.0 if len(gt_data) == len(pred_data) else 0.0

    order = 0.0
    if sorted(str(gt_data)) == sorted(str(pred_data)):
        order = 1.0 if gt_data == pred_data else 0.0

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
        "cell_precision": round(prec, 2),
        "cell_recall": round(rec, 2),
        "tuple_cardinality": cardinality,
        "tuple_order": order,
        "exact_match": exact
    }

#valutazione
def run_evaluation():
    with open("data/dev.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    database_scelti = ["concert_singer", "pets_1"]
    test_set = [sample for sample in dataset if sample["db_id"] in database_scelti]
    
    results = []
    
    for i, sample in enumerate(test_set):
        print(f"\n--- Elaborazione {i+1}/{len(test_set)} ---")
        db_id = sample['db_id']
        question = sample['question']
        gt_sql = sample['query']
        
        gt_data, _ = execute_query(db_id, gt_sql)

        oracle_tabs = get_oracle_tables(db_id, gt_sql)

        start_time = time.time()
        schema_ridotto = get_schema(db_id, oracle_tabs)
        sql_gen = pipeline_text_to_sql(question, schema_ridotto)
        res_sql, err_sql = execute_query(db_id, sql_gen)
        m_sql = calcola_qatch_metrics(gt_data, res_sql)
        time_sql = round(time.time() - start_time, 2)

        start_time = time.time()
        tab_text = serialize_tables(db_id, oracle_tabs)
        res_qa, raw_qa, err_qa = pipeline_table_qa(question, tab_text)
        
        m_qa = calcola_qatch_metrics(gt_data, res_qa) if res_qa else {"cell_precision": 0.0, "cell_recall": 0.0, "tuple_cardinality": 0.0, "tuple_order": 0.0, "exact_match": 0.0}
        
        time_qa = round(time.time() - start_time, 2)

        results.append({
            "id": i,
            "db_id": db_id,
            "question": question,
            "gt_data": gt_data,
            "text_to_sql": {
                "generated_query": sql_gen,
                "data": res_sql,
                "metrics": m_sql,
                "error": err_sql,
                "time_seconds": time_sql
            },
            "table_qa": {
                "oracle_tables": oracle_tabs,
                "raw_response": raw_qa,
                "data": res_qa,
                "metrics": m_qa,
                "error": err_qa,
                "time_seconds": time_qa
            }
        })
        
        print(f"Metrics SQL: P={m_sql['cell_precision']} | R={m_sql['cell_recall']} | Ordine={m_sql['tuple_order']}")
        print(f"Metrics QA : P={m_qa['cell_precision']} | R={m_qa['cell_recall']} | Ordine={m_qa['tuple_order']}")

    with open("evaluation_log.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
if __name__ == "__main__":
    run_evaluation()