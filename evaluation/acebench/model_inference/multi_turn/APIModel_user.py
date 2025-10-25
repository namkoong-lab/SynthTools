from openai import OpenAI
import os

SYSTEM_PROMPT_TRAVEL_ZH = """您是一名与agent互动的用户。

Instruction: {instruction}

规则：
- 每次只生成一行内容，以模拟用户的消息。
- 不要一次性透露所有说明内容。只提供当前步骤所需的信息。
- 不要臆测说明中未提供的信息。例如，如果agent询问订单ID，但说明中没有提到，请不要编造订单ID，而是直接表示不记得或没有。
- 当遇到需要信息确认的时候，根据Instruction 中的内容决定是否确认。
- 不要在对话中重复说明内容，而是使用您自己的话来表达相同的信息。
- 尽量使对话自然，保持说明中描述的用户个性。
- 如果说明目标已达成，生成单独一行的 'finish conversation' 消息以结束对话。
- 如果Instruction中要求预定往返航班，则需要在最开始说明意图"预定往返航班"。
"""

SYSTEM_PROMPT_BASE_ZH = """您是一名与agent互动的用户。

Instruction: {instruction}

规则：
- 每次只生成一行内容，以模拟用户的消息。
- 不要一次性透露所有说明内容。只提供当前步骤所需的信息。
- 需要将当前步骤所需的信息提供完整。例如，添加提醒时需要提供提醒的描述，标题和时间等。
- 不要臆测说明中未提供的信息。例如，Instruction中并没有直接指明外卖内容，而随意编造外卖内容。
- 当被询问是否还需要帮助时，一定要确保Instruction中的主要任务是否都已被完成，如果没有，则继续向agent提出下一步任务。
- Instructiuon中出现的名字，即默认用户全名。
- 当agent询问需要删除哪一条短信时，需要按照Instruction中的要求删除短信。
- 你不能主动向agent提供帮助，按 Instruction中的要求回复agent问题，不能编造任何你未知的信息。
- 如果所有任务已完成，生成单独一行的 'finish conversation' 消息以结束对话。
"""

SYSTEM_PROMPT_TRAVEL_EN = """You are a user interacting with an agent.

Instruction: {instruction}

Rules:
- Generate only one line of content each time to simulate the user's message.
- Do not reveal all instruction content at once. Only provide information needed for the current step.
- Do not speculate information not provided in the instructions. For example, if the agent asks for an order ID but it is not mentioned in the instructions, do not fabricate an order ID; instead, directly state that you do not remember or do not have it.
- When information confirmation is needed, decide whether to confirm based on the content in the Instruction.
- Do not repeat instruction content in the conversation; instead, express the same information in your own words.
- Keep the dialogue natural and maintain the user's personality as described in the instructions.
- If the goal in the instructions has been achieved, generate a separate line with the message 'finish conversation' to end the dialogue.
- If the Instruction requires booking a round-trip flight, you need to state the intention "Book a round-trip flight" at the very beginning.
"""

SYSTEM_PROMPT_BASE_EN = """You are a user interacting with an agent.

Instruction: {instruction}

Rules:
- Generate only one line of content each time to simulate the user's message.
- Do not reveal all instruction content at once. Only provide information needed for the current step.
- Ensure that all information needed for the current step is provided completely. For example, when adding a reminder, you need to provide the reminder's description, title, and time, etc.
- Do not speculate information not provided in the instructions. For example, if the Instruction does not directly specify takeout content, do not fabricate takeout content.
- When asked if you need further assistance, make sure whether all main tasks in the Instruction have been completed. If not, continue to provide the next step task to the agent.
- Names appearing in the Instruction are assumed to be the user's full names.
- When the agent asks which message to delete, follow the Instruction's requirements to delete the message.
- You cannot proactively offer help to the agent. Respond to the agent's questions as per the Instruction's requirements, and do not fabricate any information you do not know.
- If all tasks are completed, generate a separate line with the message 'finish conversation' to end the dialogue.
"""



def remove_prefix(text):
    if text.startswith('user:'):
        return text[5:] 
    elif text.startswith('agent:'):
        return text[6:]  
    else:
        return text  


class APIUSER():

    def __init__(self, model_name, involved_class, temperature=0.001, top_p=1, max_tokens=1000, language="zh") -> None:
        
        self.model_name = model_name.lower()
        if "gpt" in self.model_name:
            api_key = os.getenv("GPT_API_KEY")
            base_url = os.getenv("GPT_BASE_URL")
        elif "deepseek" in self.model_name:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL")
        elif "qwen" in self.model_name:
            api_key = os.getenv("QWEN_API_KEY")
            base_url = os.getenv("QWEN_BASE_URL")
        elif "kimi" in self.model_name:
            api_key = os.getenv("KIMI_API_KEY")
            base_url = os.getenv("KIMI_BASE_URL")
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
            
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.involved_class = involved_class
        self.messages = []
        self.language = language

    def get_init_prompt(self,question):

        if self.language == "zh":
            if "BaseApi" in self.involved_class:
                system_prompt = SYSTEM_PROMPT_BASE_ZH
            elif "Travel" in self.involved_class:
                system_prompt = SYSTEM_PROMPT_TRAVEL_ZH
            self.messages = [
                {
                    "role": "system",
                    "content": system_prompt.format(instruction = question)
                },
                {
                    "role": "user",
                    "content": "今天有什么需要帮助的吗？",
                }]
        
        elif self.language == "en":
            if "BaseApi" in self.involved_class:
                system_prompt = SYSTEM_PROMPT_BASE_EN
            elif "Travel" in self.involved_class:
                system_prompt = SYSTEM_PROMPT_TRAVEL_EN
            self.messages = [
                {
                    "role": "system",
                    "content": system_prompt.format(instruction = question)
                },
                {
                    "role": "user",
                    "content": "Is there anything you need help with today?",
                }]

        response = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        response = response.choices[0].message.content
        message = {"role":"system",
                   "content":response}
        self.messages.append(message)
        return response

    
    def step(self, message):
        message = remove_prefix(message)
        self.messages.append({"role": "user", "content": message})


    def respond(self) -> None:

        current_message = {}
        
        response = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        response = response.choices[0].message.content
        self.messages.append({"role": "system", "content": response})

        current_message = {"sender":"user", "recipient": "agent", "message": response}

        return current_message