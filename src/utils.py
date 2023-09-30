from typing import Any, List, Optional
import openai
import json
import time
from dataclasses import dataclass

price_table = {
    "gpt-4" : [0.03, 0.06],
    "gpt-4-32k": [0.06, 0.12],
    "gpt-4-0314": [0.03, 0.06],
    "gpt-4-32k-0314": [0.06, 0.12],
    "gpt-4-0613": [0.03, 0.06],
    "gpt-4-32k-0613": [0.06, 0.12],
    "gpt-3.5-turbo": [0.0015, 0.002],
    "gpt-3.5-turbo-16k": [0.003, 0.004],
    "gpt-3.5-turbo-16k-0613": [0.003, 0.004],
    "gpt-3.5-turbo-0301": [0.0015, 0.002],
    "text-davinci-003": [0.02, 0.02],
    "text-davinci-002": [0.02, 0.02],
    "davinci": [0.02, 0.02],
}

is_chat = {
    "gpt-4" : True,
    "gpt-4-32k": True,
    "gpt-4-0314": True,
    "gpt-4-32k-0314": True,
    "gpt-4-0613": True,
    "gpt-4-32k-0613": True,
    "gpt-3.5-turbo": True,
    "gpt-3.5-turbo-16k": True,
    "gpt-3.5-turbo-16k-0613": True,
    "gpt-3.5-turbo-0301": True,
    "text-davinci-003": False,
    "text-davinci-002": False,
    "davinci": False,
}

@dataclass
class Prompt:
    def __init__(self, content: str, role: str, **kwargs):
        self.content = content
        self.role = role
        self.kwargs = kwargs
    
    def __getattribute__(self, __name: str) -> Any:
        if __name == "content":
            return self.content
        if __name == "role":
            return self.role
        return ""

class ChatNode:
    def __init__(self, role: str = "system", content: str = ""):
        self.role = role  # either "system", "user", or "assistant"
        self.content = content  # the content of the message
        self.children: List[ChatNode] = []  # a list of ChatNode objects
        self.parent: Optional[ChatNode] = None  # the parent node

    def complete(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens=512,
        is_chat: bool = True,
        **kwargs,
    ):
        # append the completion of the current branch to the child
        messages = self.get_messages()  # get the messages from the root to this node
        retry = 3
        while retry:
            try:
                if is_chat:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    retry = 0
                    message = response["choices"][0]["message"]
                    child = ChatNode(message["role"], message["content"])
                else:
                    response = openai.Completion.create(
                        model=model,
                        prompt="\n".join([m["content"] for m in messages]),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    retry = 0
                    message = response["choices"][0]["text"]
                    child = ChatNode("assistant", message)
            except Exception as e:
                time.sleep(15)
                # If last try then raise the error.
                print(f"Warning: {e}")
                if retry == 1:
                    raise e
                retry -= 1
        self.children.append(child)
        child.parent = self
        return child

    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        return child

    def get_messages(self) -> List[dict]:
        # get the messages from the root to this node
        messages: List[dict] = []
        node = self
        while node:
            messages.append({"role": node.role, "content": node.content})
            node = node.parent
        messages.reverse()
        return messages

    def get_root(self):
        node = self
        while node.parent:
            node = node.parent
        return node

    def get_last_child(self):
        node = self
        while len(node.children) > 0:
            node = node.children[-1]
        return node


def format_prompt(prompt: str, **kwargs):
    return prompt.format(**kwargs)

def make_chat_tree(prompts: List[Prompt] or str, **kwargs) -> ChatNode:
    if isinstance(prompts, str):
        data = json.load(open(prompts, "r"))
        for required in data["required_kwargs"]:
            assert required in kwargs, f"{required} is not in kwargs"
        prompts = data["prompts"]

    root = None

    for prompt in prompts:
        assert "role" in prompt, f"role is not in prompt: {prompt}"
        assert "content" in prompt, f"content is not in prompt: {prompt}"
        assert prompt["role"] in ["user", "assistant", "system"], f"role is not valid : {prompt['role']}"

        current_node = ChatNode(prompt["role"], format_prompt(prompt["content"], **kwargs))
        if root is None:
            root = current_node
        else:
            current_node.parent = root
            root.children.append(current_node)
            root = current_node
    
    return root

def merge_chat_trees(parent: ChatNode, child: ChatNode):
    # Merge the root of the child tree to the parent ChatNode
    parent.children.append(child.get_root())
    child.get_root().parent = parent

    # Merge while root and first child are role "system"
    root = child.get_root()
    while root.children and root.children[0].role == "system" and len(root.children) == 1:
        root.content += "\n" + root.children[0].content
        root.children = root.children[0].children
        if root.children:
            root.children[0].parent = root

    # return last first child
    while root.children:
        root = root.children[0]
    return root
