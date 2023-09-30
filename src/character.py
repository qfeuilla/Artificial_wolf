from dataclasses import dataclass
from utils import make_chat_tree, merge_chat_trees, ChatNode
from typing import List
import re

@dataclass
class Player:
    name: str
    role: str
    extra: List[str]
    current_context: ChatNode = ChatNode()
    model: str = "gpt-4"

    def init_player(self, players: List[str]):
        self.current_context = make_chat_tree("../prompts/game_introduction.json", player_names=", ".join(players))
        
        self.current_context.content += f"You are going to play the role of {self.name}."

        self.current_context = merge_chat_trees(self.current_context, make_chat_tree(f"../prompts/characters/{self.role}.json"))

        if len(self.extra) > 0:
            self.current_context = merge_chat_trees(self.current_context, make_chat_tree(f"../prompts/prompt_extra.json", extra_info="- " + "\n- ".join(self.extra)))
        
        self.current_context = merge_chat_trees(self.current_context, make_chat_tree("../prompts/rules_and_game_start.json", you=self.name))

    def get_player_text(self):
        self.current_context.content += f"\n{self.name}:"
        self.current_context = self.current_context.complete(self.model, temperature=0.85, max_tokens=512)
    
        player_speech_uncensored = f"{self.name}: " + self.current_context.content
        player_speech = re.sub(r"\[[^\]]*\]", "", player_speech_uncensored)
    
        self.current_context = self.current_context.add_child(ChatNode("user", ""))

        return player_speech, player_speech_uncensored

    def add_other_text(self, other_speech : str):
        self.current_context.content += "\n" + other_speech


def get_next_speaker(conversation_history: List[str], player_names=List[str]):
    moderator_prompt = make_chat_tree("../prompts/moderator_prompt.json", player_names=",".join(player_names), current_debate="\n".join(conversation_history))
    # print(moderator_prompt.content)

    return moderator_prompt.complete("gpt-4", temperature=0.5, max_tokens=5).content