import json
import re
from typing import Tuple, Any

START = '<|tool_calls_section_begin|>'
ENVELOPE = re.compile(
    r'<\|tool_calls_section_begin\|>\s*'
    r'<\|tool_call_begin\|>\s*functions\.execute_code:\d+\s*'
    r'<\|tool_call_argument_begin\|>(.*)'
    r'<\|tool_call_end\|>\s*<\|tool_calls_section_end\|>', re.DOTALL)


def normalize_action_text(content):
    if not isinstance(content, str) or not content.lstrip().startswith(START):
        return content
    match = ENVELOPE.fullmatch(content.strip())
    if match is None:
        raise ValueError('Unsupported or incomplete serialized tool call; expected one execute_code call')
    try:
        params = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError('Serialized tool call must contain one complete JSON argument object') from exc
    # Some responses wrap the complete official action instead of only its args.
    if isinstance(params, dict) and set(params) == {'action', 'params'}:
        if params['action'] != 'execute_code':
            raise ValueError('Serialized execute_code call contains a different action')
        params = params['params']
    if not isinstance(params, dict) or set(params) != {'code'} or not isinstance(params['code'], str):
        raise ValueError('execute_code arguments must contain only a string code field')
    return json.dumps({'action': 'execute_code', 'params': params}, ensure_ascii=False)


def parse_llm_response(response: str) -> Tuple[str, Any]:
    """
    Parse the response from the LLM model.
    """
    try:
        if "<Final_answer>" in response and "</Final_answer>" in response:
            start = response.find("<Final_answer>") + len("<Final_answer>")
            end = response.find("</Final_answer>")
            answer = response[start:end].strip()
            return "submit", {"answer": answer}

        if '</think>' in response:
            response = response.split('</think>')[-1].strip()

        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if start > 6 and end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if start > 2 and end > start:
                candidate = response[start:end].strip()
                if candidate.startswith("{"):
                    response = candidate

        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1:
            response = response[start:end + 1].strip()
            
        if '"""' in response:
            left = response.find('"""')
            right = response.rfind('"""')
            if left != -1 and right != -1 and left != right:
                code = response[left + 3:right].strip()
                code = code.replace("\n", "\\n")
                response = response[:left].strip() + '\"' + code + '\"' + response[right + 3:].strip()

        if not response.strip():
            raise ValueError("Empty response")
            
        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}\nResponse: {response}")
            
        if not isinstance(response_dict, dict):
            raise ValueError(f"Response must be a JSON object, got {type(response_dict)}")
            
        required_keys = {"action", "params"}
        actual_keys = set(response_dict.keys())
        if not required_keys.issubset(actual_keys):
            raise ValueError(f"Response must contain at least these keys: {required_keys}, got: {actual_keys}")
            
        action = response_dict["action"]
        params = response_dict["params"]
        
        if action == "validate_code":
            action = "execute_code"
            
        valid_actions = {"execute_code"}
        if action not in valid_actions:
            if action in ["request_info", "debug", "terminal"]:
                return "execute_code", {"code": f"print('Action {action} is deprecated. Please write Python code to accomplish your goal.')"}
            raise ValueError(f"Invalid action '{action}'. Must be: {valid_actions}")
            
        if not isinstance(params, dict):
            raise ValueError(f"'params' must be a dict, got {type(params)}")
            
        allowed_params_keys = {"code"}
        actual_params_keys = set(params.keys())
        if actual_params_keys != allowed_params_keys:
            raise ValueError(f"'{action}' params must contain only: {allowed_params_keys}, got: {actual_params_keys}")
        if not isinstance(params.get("code"), str):
            raise ValueError(f"'code' must be string, got type {type(params.get('code'))}")

        return action, params
        
    except Exception as e:
        raise ValueError(f"Failed to fix parse error: {e}")