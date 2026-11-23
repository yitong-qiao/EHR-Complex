import abc
import logging
from copy import copy, deepcopy
from dataclasses import asdict

class EHRPrompt:
    def __init__(self):
        self.prompt_template = """{instruction}
You have access to exactly ONE action with params, and receive corresponding feedback after the action:
1. execute_code: Execute a Python script to explore data or calculate intermediate results.
    - params: code (str)
    - feedback: returns the standard output (stdout) or standard error (stderr) of your script.

Code requirements:
    - Explore the data first to understand the schema.
    - Use the variable 'answer' to store the answer of the code.
    - Code should be self-contained and not rely on any variables or state outside.
    
Response format requirements, strictly one of the following:
{format_output}
    - Must be valid JSON format
    - No additional text or formatting allowed"""
        self.format_output = """{
    "action": "execute_code",
    "params": {
        "code": "<code>"
    }
}"""


class DynamicPrompt:
    def __init__(self):


        self.prompt_template = """{instruction}
You are operating in an INTERACTIVE environment. You can explore the data by writing Python code (e.g., using pandas for .csv or sqlite3/duckdb for databases), executing them, and analyzing the results. 

You have access to action:
{action_definition}
    
Response format requirements:
When you are exploring and need to execute code, STRICTLY output the following valid JSON format and NOTHING else:
{format_output}

Submitting the Final Answer:
Only submit when you are certain the execution output you have seen is the correct answer to the question. If you have any doubt about whether the query is correct, run another query to verify before submitting.
Once you are confident, stop using JSON actions and submit your final answer by wrapping it in the `<Final_answer>` tag and appending `<DONE>`.

CRITICAL RULES FOR FINAL ANSWER:
1. COMPLETE BUT CONCISE: You MUST answer EVERY part of the question. If the question asks for multiple things (e.g., a date AND a list of IDs), provide ALL of them separated by commas (e.g., `2105-05-22, 198770, 179042`).
2. NO FULL SENTENCES: Output just the values.
3. EXACT RAW VALUES: You MUST output the exact raw value as it appears in the database. Do not expand abbreviations.

Example (the value below is purely illustrative — never copy it as your answer):
Based on my execution results, the patient count is 42.
<Final_answer> 42 </Final_answer>
<DONE>"""




        self.action_definition = {
            "execute_code": """execute_code: Execute a Python script to query databases (SQLite .db, .duckdb or .csv), explore EHR data, or calculate intermediate results.
    - params: code (str)
    - feedback: returns the standard output (stdout) or standard error (stderr) of your script.
    - Note: This action ONLY RUNS CODE, it does NOT submit the final answer. You MUST use print() to see the query results.
    - Note: Each code execution runs in a FULLY ISOLATED subprocess. Variables, imports, and data loaded in one execution are NOT available in the next. You MUST re-import libraries (e.g., pandas) and reload any data files at the beginning of EVERY code block."""
        }
        self.action_format = {
            "execute_code": """{
    "action": "execute_code",
    "params": {
        "code": "<code>"
    }
}"""
        }