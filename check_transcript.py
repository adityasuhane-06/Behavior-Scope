import sqlite3
import json

conn = sqlite3.connect('data/audit/behavior_scope_audit.db')
cursor = conn.cursor()

cursor.execute("""
    SELECT transcript_text, transcript_json, transcript_speaker_count 
    FROM sessions 
    WHERE session_id = 'session_20260124_210550'
""")

row = cursor.fetchone()
if row:
    print("=== TRANSCRIPT TEXT ===")
    print(row[0])
    print("\n=== TRANSCRIPT JSON ===")
    if row[1]:
        transcript_json = json.loads(row[1])
        print(json.dumps(transcript_json, indent=2))
    else:
        print("No JSON data")
    print(f"\n=== SPEAKER COUNT: {row[2]} ===")
else:
    print("No data found")

conn.close()
