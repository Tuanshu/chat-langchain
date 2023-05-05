from typing import List, Union

from langchain.agents import Tool
from langchain.prompts import StringPromptTemplate

tools=[]
# Set up the base template
template = """Transcript of a dialog, where the User interacts with an Assistant. Assistant is kind, honest, and never fails to answer 
the User's requests immediately and with concision. Assistant can only speak English.

User: Hello, 請問你是誰？
Assistant: I am an Assistant. I am here to help you with your tasks. What can I do for you?
User: {input}
"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        return self.template.format(**kwargs)
    

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input"]
)