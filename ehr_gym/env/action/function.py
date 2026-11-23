import time
import os
from pathlib import Path
import traceback
from datetime import datetime
import logging
import uuid
from typing import Dict, Any, Optional, Tuple
from ehr_gym.utils.env_utils import parse_and_truncate_error
from ehr_gym.env.action.isolation import run_isolated
import re

logging.basicConfig(
    level=logging.INFO, format="%(name)s : %(levelname)-8s : %(message)s"
)
logger = logging.getLogger(__name__)

def _run_code_file(code_file: Path, timeout: Optional[int] = None, workspace=None) -> Tuple[int, str, str, float]:
    return run_isolated(code_file, timeout=timeout, workspace=workspace)

def execute_code(code: str, workspace=None) -> Dict[str, Any]:
    """
    Execute the generated python code.
    """
    success = True
    output = ""
    start_time = time.time()
    
    try:
        compile(code, '<string>', 'exec')
    except Exception as e:
        success = False
        error_msg = f"Compilation error occurred for the code: {str(e)}\nFull traceback:\n{traceback.format_exc()}"
        output = parse_and_truncate_error(error_msg)
        execution_time = time.time() - start_time
        logger.error("Compilation error during code execution.")
        return {
            "type": "code_execution",
            "status": "FAILED",
            "env_message": output,
            "execution_time": f"{execution_time:.2f}s"
        }
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uuid_str = str(uuid.uuid4())
    code_file = os.path.join("./cache", f"execution_code_{timestamp}_{uuid_str}.py")
    
    if "duckdb.connect" in code:
        def enforce_readonly(match):
            inner_args = match.group(1)
            if "read_only" not in inner_args:
                return f"duckdb.connect({inner_args}, read_only=True)"
            return match.group(0)
            
        code = re.sub(r"duckdb\.connect\((.*?)\)", enforce_readonly, code)

    code_file = Path(code_file)
    code_file.parent.mkdir(parents=True, exist_ok=True)
    if not 'print' in code and 'answer' in code:
        code += "\nprint(answer)"
    code_file.write_text(code)

    try:
        returncode, stdout, stderr, execution_time = _run_code_file(code_file, timeout=60, workspace=workspace)
        combined_output = stdout + ("\n" + stderr if stderr else "")
        if returncode != 0:
            success = False
            output = parse_and_truncate_error(combined_output)
        else:
            output = combined_output
    except Exception as e:
        success = False
        error_msg = f"Error executing code: {str(e)}\nFull traceback:\n{traceback.format_exc()}"
        output = parse_and_truncate_error(error_msg)
        execution_time = time.time() - start_time
        logger.error("Error executing code.")
    
    try:
        os.remove(code_file)
    except Exception as e:
        logger.error(f"Error deleting code file: {e}")

    
    return {
        "type": "code_execution",
        "status": "SUCCESS" if success else "FAILED",
        "env_message": output,
        "execution_time": f"{execution_time:.2f}s"
    }
