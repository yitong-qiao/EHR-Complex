import typing
from ehr_gym.env.action.function import execute_code
from ehr_gym.env.action.base import AbstractActionSet
from dataclasses import dataclass

@dataclass
class BasicAction:
    signature: str
    description: str
    examples: list[str]

class BasicActionSet(AbstractActionSet):
    def __init__(self, custom_actions: typing.Optional[list[callable]] = None, strict: bool = False):
        super().__init__(strict)
        
        allowed_actions = [execute_code]
        if custom_actions:
            allowed_actions.extend(custom_actions)
        allowed_actions = list(dict.fromkeys(allowed_actions).keys())

        self.action_set = {
            "execute_code": execute_code,
        }
    
    def example_action(self, action) -> str:
        if action not in self.action_set:
            raise ValueError(f"Action {action} not found in action set.")
        return self.action_set[action].examples
    
    def describe(self, with_long_description: bool = True, with_examples: bool = True):
        description = f"\n{len(self.action_set)} different types of actions are available.\n\n"
        for _, action in self.action_set.items():
            description += f"{action.signature}\n"
            if with_long_description:
                description += f"    Description: {action.description}\n"
            if with_examples and getattr(action, 'examples', None):
                description += f"    Examples:\n"
                for example in action.examples:
                    description += f"        {example}\n"
        return description