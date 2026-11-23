from typing import Any, Dict, Union, List, Tuple
from dataclasses import dataclass
import json
import time
import tiktoken
from ehr_gym.llm.chat_api import OpenAIModelArgs, AzureModelArgs, VLLMModelArgs, make_system_message, make_user_message, make_assistant_message
from ehr_gym.agent.prompt import EHRPrompt, DynamicPrompt
from ehr_gym.agent.parser import parse_llm_response

@dataclass
class LLMConfig:
    """Configuration for the LLM agent."""
    model_path: str
    tokenizer_path: str
    max_length: int
    device: str
    port: int = 8000

class EHRAgent:
    """Agent class to interact with EHREnvironment using LLM"""
    def __init__(self, agent_config, permitted_actions):
        self.conversation_history = []
        self.agent_config = agent_config
        self.llm_config = agent_config['llm']

        if self.llm_config["model_type"] == 'OpenAI':
            self.llm = OpenAIModelArgs(
                model_name=self.llm_config["model_name"],
                max_total_tokens=self.llm_config["max_total_tokens"],
                max_input_tokens=self.llm_config["max_input_tokens"],
                max_new_tokens=self.llm_config["max_new_tokens"],
                vision_support=False,
                temperature=self.llm_config["temperature"],
                top_p=self.llm_config.get("top_p", None),
                top_k=self.llm_config.get("top_k", None),
                enable_thinking=self.llm_config.get("enable_thinking", None),
                reasoning_effort=self.llm_config.get("reasoning_effort", None),
            ).make_model()
        elif self.llm_config["model_type"] == 'Azure':
            self.llm = AzureModelArgs(
                model_name=self.llm_config["model_name"],
                temperature=self.llm_config["temperature"],
                max_new_tokens=self.llm_config["max_new_tokens"],
                deployment_name=self.llm_config["deployment_name"],
                log_probs=self.llm_config["log_probs"]
            ).make_model()
        elif self.llm_config["model_type"] == 'vLLM':
            self.llm = VLLMModelArgs(
                model_name=self.llm_config["model_name"],
                temperature=self.llm_config["temperature"],
                max_new_tokens=self.llm_config["max_new_tokens"],
                port=self.llm_config["port"],
            ).make_model()
        else:
            raise ValueError("Model type {} not supported.".format(self.llm_config["model_type"]))
        self.conversation_history = []
        self.prompt = DynamicPrompt()
        self.parser = parse_llm_response
        self.cost = []
        self.permitted_actions = permitted_actions

    def act(self, obs: Any) -> Tuple[str, Dict[str, Any]]:
        """Main interface to get action from agent."""
        if self.conversation_history == []:
            instruction = obs['info']['instruction']
            action_definitions = []
            action_formats = []
            for action in self.permitted_actions:
                action_definitions.append(self.prompt.action_definition[action])
                action_formats.append(self.prompt.action_format[action])
            action_definitions = "\n".join(action_definitions)
            action_formats = "\nor\n".join(action_formats)
            system_msg = self.prompt.prompt_template.format(instruction=instruction, action_definition=action_definitions, format_output=action_formats)
            if self.llm_config.get("model_name", "").lower() == "gpt-5.4":
                system_msg += (
                    "\n\nGPT-5.4 FORMAT OVERRIDE:\n"
                    "When you need to execute code, your entire response must be exactly one JSON object.\n"
                    "Do not output more than one JSON object in the same response.\n"
                    "Do not include a Final_answer in the same response as an execute_code JSON object.\n"
                    "Do not use markdown fences, prose, comments outside JSON, or multiple actions.\n"
                    "The only valid action response is exactly: {\"action\": \"execute_code\", \"params\": {\"code\": \"<code>\"}}\n"
                )
            user_msg = obs['info']['task_goal']
            if self.llm_config.get("model_name", "").lower() == "gpt-5.4":
                user_msg += (
                    "\nGPT-5.4 next-response constraint: you have not executed code yet, so do not answer. "
                    "Output exactly one JSON object for execute_code and nothing else. "
                    "Do not include <Final_answer>, markdown, prose, or a second JSON object in this response."
                )
            if self.llm_config["model_type"] == 'vLLM':
                user_msg = make_user_message(content=system_msg+'\n'+user_msg)
            else:
                system_msg = make_system_message(content=system_msg)
                self.conversation_history.append(system_msg)
                user_msg = make_user_message(content=user_msg)
            self.conversation_history.append(user_msg)
        elif 'env_message' in obs:
            user_msg = obs['env_message']
            if not user_msg or not user_msg.strip():
                user_msg = "(No output)"
            user_msg = make_user_message(content=user_msg)
            self.conversation_history.append(user_msg)
        for i in range(self.agent_config['n_retry']):
            try:
                # Preserve returned reasoning for history and subsequent model turns.
                response, cost = self.llm(self.conversation_history)
                reasoning_content = getattr(response, "reasoning_content", None)
                if reasoning_content is None:
                    reasoning_content = getattr(response, "reasoning", None)
                response = response.content
                self.cost.append(cost)
            except Exception as e:
                err_str = str(e)
                if "longer than the model's context length" in err_str \
                        or "context_length_exceeded" in err_str \
                        or "Requested token count exceeds" in err_str:
                    self.context_length_exceeded = True
                    return "error", {}
                time.sleep(self.agent_config["retry_delay"])
                action, params = "error", {}
                continue
            try:
                action, params = self.parser(response)
            except Exception as e:
                time.sleep(self.agent_config["retry_delay"])
                action, params = f"error: str({e})", {}
                assistant_msg = make_assistant_message(content=response)
                if reasoning_content is not None:
                    assistant_msg["reasoning_content"] = reasoning_content
                self.conversation_history.append(assistant_msg)
                error_hint_content = (
                    f"Your response could not be parsed. Error: {e}. "
                    "Please output ONLY a valid JSON action block or a <Final_answer> tag. "
                    "Do NOT include any other text outside the JSON."
                )
                if self.llm_config.get("model_name", "").lower() == "gpt-5.4":
                    error_hint_content += (
                        " For GPT-5.4, if you are executing code, output exactly one JSON object "
                        "and nothing else. Do not output multiple JSON objects, markdown, prose, "
                        "or a Final_answer together with an execute_code JSON object."
                    )
                error_hint = make_user_message(content=error_hint_content)
                self.conversation_history.append(error_hint)
                continue
            assistant_msg = make_assistant_message(content=response)
            if reasoning_content is not None:
                assistant_msg["reasoning_content"] = reasoning_content
            self.conversation_history.append(assistant_msg)
            break

        return action, params