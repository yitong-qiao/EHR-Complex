import os
import json
import math
import re
import random
from collections import defaultdict
from .base import AbstractEHRTask

overall_information = """Assume you have knowledge of several tables:
(1) Tables are linked by identifiers which usually have the suffix '_id'. For example, subject_id refers to a unique patient, hadm_id refers to a unique hospital admission, and stay_id refers to a unique admission to an intensive care unit.
(2) Charted events such as notes, laboratory tests, and fluid balance are stored in a series of 'events' tables. For example, the outputevents table contains all measurements related to output for a given patient, while the labevents table contains laboratory test results for a patient.
(3) Tables prefixed with 'd_' are dictionary tables and provide definitions for identifiers. For example, every row of chartevents is associated with a single itemid which represents the concept measured, but it does not contain the actual name of the measurement. By joining chartevents and d_items on itemid, it is possible to identify the concept represented by a given itemid. Other dictionary tables include d_hcpcs, d_icd_diagnoses, d_icd_procedures, and d_labitems.
(4) For the databases, five of them are used to define and track patient stays and movements: admissions, icustays, patients, services, and transfers. Another five tables are dictionaries for cross-referencing codes against their respective definitions: d_hcpcs, d_icd_diagnoses, d_icd_procedures, d_items, and d_labitems. The remaining 21 tables, including caregiver, chartevents, datetimeevents, diagnoses_icd, drgcodes, emar, emar_detail, hcpcsevents, ingredientevents, inputevents, labevents, microbiologyevents, omr, outputevents, pharmacy, poe, poe_detail, prescriptions, procedureevents, procedures_icd, and provider, contain data associated with patient care, such as physiological measurements, caregiver observations, medication administration, and billing information.
"""

table_information = """For different tables, they contain the following information:
(1) admissions: subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, admit_provider_id, admission_location, discharge_location, insurance, language, marital_status, race, edregtime, edouttime, hospital_expire_flag
(2) caregiver: caregiver_id
(3) chartevents: subject_id, hadm_id, stay_id, caregiver_id, charttime, storetime, itemid, value, valuenum, valueuom, warning
(4) d_hcpcs: code, category, long_description, short_description
(5) d_icd_diagnoses: icd_code, icd_version, long_title
(6) d_icd_procedures: icd_code, icd_version, long_title
(7) d_items: itemid, label, abbreviation, linksto, category, unitname, param_type, lownormalvalue, highnormalvalue
(8) d_labitems: itemid, label, fluid, category
(9) datetimeevents: subject_id, hadm_id, stay_id, caregiver_id, charttime, storetime, itemid, value, valueuom, warning
(10) diagnoses_icd: subject_id, hadm_id, seq_num, icd_code, icd_version
(11) drgcodes: subject_id, hadm_id, drg_type, drg_code, description, drg_severity, drg_mortality
(12) emar: subject_id, hadm_id, emar_id, emar_seq, poe_id, pharmacy_id, enter_provider_id, charttime, medication, event_txt, scheduletime, storetime
(13) emar_detail: subject_id, emar_id, emar_seq, parent_field_ordinal, administration_type, pharmacy_id, barcode_type, reason_for_no_barcode, complete_dose_not_given, dose_due, dose_due_unit, dose_given, dose_given_unit, will_remainder_of_dose_be_given, product_amount_given, product_unit, product_code, product_description, product_description_other, prior_infusion_rate, infusion_rate, infusion_rate_adjustment, infusion_rate_adjustment_amount, infusion_rate_unit, route, infusion_complete, completion_interval, new_iv_bag_hung, continued_infusion_in_other_location, restart_interval, side, site, non_formulary_visual_verification
(14) hcpcsevents: subject_id, hadm_id, chartdate, hcpcs_cd, seq_num, short_description
(15) icustays: subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime, los
(16) ingredientevents: subject_id, hadm_id, stay_id, caregiver_id, starttime, endtime, storetime, itemid, amount, amountuom, rate, rateuom, orderid, linkorderid, statusdescription, originalamount, originalrate
(17) inputevents: subject_id, hadm_id, stay_id, caregiver_id, starttime, endtime, storetime, itemid, amount, amountuom, rate, rateuom, orderid, linkorderid, ordercategoryname, secondaryordercategoryname, ordercomponenttypedescription, ordercategorydescription, patientweight, totalamount, totalamountuom, isopenbag, statusdescription, originalamount, originalrate
(18) labevents: labevent_id, subject_id, hadm_id, specimen_id, itemid, order_provider_id, charttime, storetime, value, valuenum, valueuom, ref_range_lower, ref_range_upper, flag, priority, comments
(19) microbiologyevents: microevent_id, subject_id, hadm_id, micro_specimen_id, order_provider_id, chartdate, charttime, spec_itemid, spec_type_desc, test_seq, storedate, storetime, test_itemid, test_name, org_itemid, org_name, isolate_num, quantity, ab_itemid, ab_name, dilution_text, dilution_comparison, dilution_value, interpretation, comments
(20) omr: subject_id, chartdate, seq_num, result_name, result_value
(21) outputevents: subject_id, hadm_id, stay_id, caregiver_id, charttime, storetime, itemid, value, valueuom
(22) patients: subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod
(23) pharmacy: subject_id, hadm_id, pharmacy_id, poe_id, starttime, stoptime, medication, proc_type, status, entertime, verifiedtime, route, frequency, disp_sched, infusion_type, sliding_scale, lockout_interval, basal_rate, one_hr_max, doses_per_24_hrs, duration, duration_interval, expiration_value, expiration_unit, expirationdate, dispensation, fill_quantity
(24) poe: poe_id, poe_seq, subject_id, hadm_id, ordertime, order_type, order_subtype, transaction_type, discontinue_of_poe_id, discontinued_by_poe_id, order_provider_id, order_status
(25) poe_detail: poe_id, poe_seq, subject_id, field_name, field_value
(26) prescriptions: subject_id, hadm_id, pharmacy_id, poe_id, poe_seq, order_provider_id, starttime, stoptime, drug_type, drug, formulary_drug_cd, gsn, ndc, prod_strength, form_rx, dose_val_rx, dose_unit_rx, form_val_disp, form_unit_disp, doses_per_24_hrs, route
(27) procedureevents: subject_id, hadm_id, stay_id, caregiver_id, starttime, endtime, storetime, itemid, value, valueuom, location, locationcategory, orderid, linkorderid, ordercategoryname, ordercategorydescription, patientweight, isopenbag, continueinnextdept, statusdescription, originalamount, originalrate
(28) procedures_icd: subject_id, hadm_id, seq_num, chartdate, icd_code, icd_version
(29) provider: provider_id
(30) services: subject_id, hadm_id, transfertime, prev_service, curr_service
(31) transfers: subject_id, hadm_id, transfer_id, eventtype, careunit, intime, outtime

All the tables are located within the provided database.
"""






instruction = """You are a biomedical expert handling EHR data to answer questions.
Your objective is to answer the user's question with given EHR data. You can execute python code interactively to explore the database, calculate intermediate results, and find the answer step by step.
{overall}

{EHR_tables}

CRITICAL INSTRUCTION FOR DATABASE ACCESS:
The database is a DuckDB file located at: {db_path}
The environment automatically provides a ready-to-use `conn` variable (a DuckDB connection) at the start of every code execution — you do NOT need to import duckdb or call duckdb.connect() yourself.

EXECUTION POLICY:
Use only the provided EHR database as source data. Benchmark files, reference answers, standard SQL, graders, and other trajectories are inaccessible. Prohibited access returns ILLEGAL_OPERATION; continue with a valid database query after a denial.

Syntax Example for querying (Note: Write this logic INSIDE your JSON action's "code" parameter):
df = conn.execute("SELECT * FROM admissions LIMIT 5").df()
print(df)

CRITICAL INSTRUCTION FOR FINAL ANSWER:
When submitting your final answer using the `<Final_answer>` tag, DO NOT output JSON objects or dicts (e.g., avoid {{"insurance": "Medicaid"}}). 
Instead, output the exact values in plain text.
- If the answer is a single value, just write the value (e.g., `<Final_answer> Medicaid </Final_answer>`).
- If the answer contains multiple values, row data, or multiple columns, write them as a comma-separated list or a simple Python list (e.g., `<Final_answer> Medicare, 21 </Final_answer>` or `<Final_answer> [6, 12] </Final_answer>`).
"""


class EHRComplexTask(AbstractEHRTask):
    permitted_actions = ['execute_code']

    def __init__(
        self,
        task_id: int,
        data_path: str = None,
        mode: str = "test",
        num_param_expansions: int = None,
        num_paraphrases: int = None,
    ) -> None:
        super().__init__(task_id=task_id)
        self.task_id = task_id
        self.task_list = None
        self.data_path = data_path if data_path else os.path.join(os.path.dirname(__file__), "../../../data/ehr_complex")
        self.mode = mode
        self.num_param_expansions = num_param_expansions
        self.num_paraphrases = num_paraphrases

        self.database_directory = os.environ.get("EHR_COMPLEX_DB", os.path.join(os.path.dirname(__file__), "../../../data/mimic_iv_full.duckdb"))

    @classmethod
    def get_task_id(cls):
        return "EHRGym.ehr_complex"

    def setup(self) -> tuple[str, dict]:
        if self.task_list is None:
            self.task_list = []
            rng = random.Random(42)
            subsets = sorted(["single_patient", "cohort_patient"])
            for subset in subsets:
                subset_dir = os.path.join(self.data_path, subset)
                if not os.path.exists(subset_dir):
                    continue

                categories = sorted(os.listdir(subset_dir))
                for category in categories:
                    cat_dir = os.path.join(subset_dir, category)
                    if not os.path.isdir(cat_dir):
                        continue

                    templates = sorted(os.listdir(cat_dir))
                    for template_file in templates:
                        if not template_file.endswith(".jsonl"):
                            continue

                        template_name = template_file.replace(".jsonl", "")
                        file_path = os.path.join(cat_dir, template_file)

                        all_rows = []
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    all_rows.append(json.loads(line))

                        groups = defaultdict(list)
                        for row in all_rows:
                            groups[row.get('original_question', '')].append(row)

                        sorted_groups = sorted(groups.items())  # [(oq, [row, ...]), ...]

                        if (self.num_param_expansions is not None
                                and len(sorted_groups) > self.num_param_expansions):
                            sorted_groups = rng.sample(sorted_groups, self.num_param_expansions)
                            sorted_groups = sorted(sorted_groups)

                        for oq, rows in sorted_groups:
                            rows_sorted = sorted(rows, key=lambda r: r.get('paraphrase_index', 0))

                            if (self.num_paraphrases is not None
                                    and len(rows_sorted) > self.num_paraphrases):
                                rows_sorted = rng.sample(rows_sorted, self.num_paraphrases)
                                rows_sorted = sorted(rows_sorted, key=lambda r: r.get('paraphrase_index', 0))

                            for line_idx, data in enumerate(rows_sorted):
                                self.task_list.append({
                                    "question": data.get("question", ""),
                                    "expected_result": data.get("expected_result", {}),
                                    "standard_sql": data.get("standard_sql", ""),
                                    "metadata": {
                                        "subset": subset,
                                        "category": category,
                                        "template": template_name,
                                        "line_idx": data.get("paraphrase_index", line_idx),
                                        "original_question": data.get("original_question", ""),
                                        "paraphrase_index": data.get("paraphrase_index", line_idx),
                                        "raw_question_template": data.get("raw_question_template", ""),
                                        "raw_sql_template": data.get("raw_sql_template", ""),
                                    }
                                })

        task_data = self.task_list[self.task_id]
        self.question = task_data['question']
        self.expected_result = task_data['expected_result']
        self.standard_sql = task_data.get('standard_sql', '')
        self.metadata = task_data['metadata']

        goal, info = self.setup_goal()
        return goal, info

    def setup_goal(self) -> tuple[str, dict]:
        super().setup_goal()
        self.goal = (
            f"You can execute python code to explore the database and solve the question.\n"
            f"Question: {self.question}\n\n"
        )

        self.context_pattern = "<<AGENT_CODE>>"
        self.context = (
            f"import duckdb\n"
            "conn = duckdb.connect('/data/mimic_iv_full.duckdb', read_only=True)\n"
            f"{self.context_pattern}"
        )

        return self.goal, {}

    def _get_obs(self) -> dict:
        obs = {"type": "initial_observation", "info": {}}
        obs["info"]["table_info"] = table_information
        obs["info"]["overall"] = overall_information
        obs["info"]["task_goal"] = self.goal
        obs["info"]["instruction"] = instruction.format(
            overall=overall_information,
            EHR_tables=table_information,
            db_path="/data/mimic_iv_full.duckdb"
        )
        obs["info"]["context_info"] = {
            "context": self.context,
            "context_pattern": self.context_pattern,
        }
        return obs

    def validate(self, chat_messages, obs):
        if obs["type"] == "submit":
            from dataset.grading.ehr_complex_grader import evaluate
            result = evaluate(self.question, self.expected_result, obs['env_message'])
            return (result, True,
                    'The answer is correct' if result else 'The answer is incorrect',
                    {'message': 'EHRComplex unified evaluator: ' + ('correct' if result else 'incorrect')})
        elif obs["type"] == "code_execution":
            return (0, False, "Code executed", {"message": f"Execution Output:\n{obs['env_message']}\n\nIf you found the answer, use <Final_answer> tag. Otherwise, continue exploring."})
            
        elif obs["type"] == "error_message":
            return (0, False, "The code encountered with errors", {"message": f"The code encountered errors. Please debug it.\nError Message:\n{obs['message']}"})
            
        else:
            return (0, False, "", {"message": obs.get('env_message', '')})