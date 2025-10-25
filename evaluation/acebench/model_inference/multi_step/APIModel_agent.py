
from openai import OpenAI
import os
import re 


MULTI_TURN_AGENT_PROMPT_SYSTEM_ZH = """你是一个AI系统，你的角色为system，请根据给定的API说明和对话历史1..t，为角色system生成在步骤t+1中生成相应的内容。
1 如果上一步提供的信息完整，能够正常进行api的调用，你应该调用的API请求，API请求以[ApiName(key1='value1', key2='value2', ...)]的格式输出，将ApiName替换为实际的API名称，将key1、key2等替换为实际的参数名称，将value1、value2替换为实际参数取值。输出应以方括号"["开头，以方括号"]"结尾。API请求有多个时以英文逗号隔开，比如[ApiName(key1='value1', key2='value2', ...), ApiName(key1='value1', key2='value2', ...), ...]。不要在输出中输出任何其他解释或提示或API调用的结果。\n 
如果API参数描述中没有特殊说明，则该参数为非必选参数（用户输入中提及的参数需要包含在输出中，如果未提及，则不需要包含在输出中）。\n如果API参数描述未指定取值格式要求，则该参数取值使用用户原文。
2 当一个任务需要多个步骤才能完成(步骤之间有严格的前后关系)，你需要一步步执行，并根据每一轮execution返回的结果决定下一步如何执行。
3 一般不使用并行调用的方法，也就是一次只调用一个函数。

请注意，如果需要进行api调用，请严格遵守调用规则[ApiName(key1='value1', key2='value2', ...)]，此时不得输出其他内容。
当你认为任务已经完成，请返回"finish conversation"以结束对话。

角色说明：
user: 用户 
agent: 进行API请求调用的AI系统角色 
execution: 执行api调用并返回结果
"""

MULTI_TURN_AGENT_PROMPT_USER_ZH = """以下是你可以调用的API列表（JSON格式）：{functions}。对话历史：{history}\n"""

FOOD_SYSTEM_ZH = """下面是不同用户的账号信息和密码，需要时使用: 
{
            "Eve": {"user_name": "Eve", "password": "password123"},
            "Frank": {"user_name": "Frank", "password": "password456"},
            "Grace": {"user_name": "Grace", "password": "password789"},
            "Helen": {"user_name": "Helen", "password": "password321"},
            "Isaac": {"user_name": "Isaac", "password": "password654"},
            "Jack": {"user_name": "Jack", "password": "password654"},
}"""

MULTI_TURN_AGENT_PROMPT_SYSTEM_EN = """You are an AI system with the role of 'system'. Based on the provided API documentation and the conversation history from steps 1 to t, generate the corresponding content for the 'system' role in step t+1.
1. If the information provided in the previous step is complete and allows for a successful API call, you should output the API request(s) to be called in the format [ApiName(key1='value1', key2='value2', ...)]. Replace ApiName with the actual API name, key1, key2, etc., with the actual parameter names, and value1, value2, etc., with the actual parameter values. The output should start with a square bracket "[" and end with a square bracket "]". If there are multiple API requests, separate them with commas, for example, [ApiName(key1='value1', key2='value2', ...), ApiName(key1='value1', key2='value2', ...), ...]. Do not include any additional explanations, prompts, or API call results in the output.
   - If the API parameter description does not specify otherwise, the parameter is optional (only include parameters mentioned in the user input; if not mentioned, do not include them).
   - If the API parameter description does not specify a required value format, use the user's original input for the parameter value.
2. If a task requires multiple steps to complete (with strict sequential relationships between steps), execute them step by step, and decide how to proceed based on the results returned from each execution.
3. Generally do not use parallel calls, meaning only one function is called at a time.

Please note that if an API call is needed, strictly adhere to the calling rules [ApiName(key1='value1', key2='value2', ...)] and do not output any other content.
When you believe the task is completed, return "finish conversation" to end the dialogue.

Role Descriptions:
user: The user
agent: The AI system role that performs API requests
execution: Executes API calls and returns results
"""

MULTI_TURN_AGENT_PROMPT_USER_EN = """Below is the list of APIs you can call (in JSON format): {functions}. Conversation history: {history}\n"""

FOOD_SYSTEM_EN = """Below is the account information and passwords for different users, to be used when needed: 
{
    "Eve": {"user_name": "Eve", "password": "password123"},
    "Frank": {"user_name": "Frank", "password": "password456"},
    "Grace": {"user_name": "Grace", "password": "password789"},
    "Helen": {"user_name": "Helen", "password": "password321"},
    "Isaac": {"user_name": "Isaac", "password": "password654"},
    "Jack": {"user_name": "Jack", "password": "password654"}
}"""


class APIAgent_step():

    def __init__(self, model_name, time, functions, temperature=0.001, top_p=1, max_tokens=1000, language="zh") -> None:
        self.model_name = model_name.lower()
        
        if "gpt" in self.model_name:
            api_key = os.getenv("GPT_AGENT_API_KEY")
            base_url = os.getenv("GPT_AGENT_BASE_URL")
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
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.time = time
        self.functions = functions
        self.language = language


    def respond(self, history) -> None:

        current_message = {}
        if self.language == "zh":
            system_prompt = MULTI_TURN_AGENT_PROMPT_SYSTEM_ZH.format(time = self.time)
            user_prompt = MULTI_TURN_AGENT_PROMPT_USER_ZH.format(functions = self.functions, history = history)
        elif self.language == "en":
            system_prompt = MULTI_TURN_AGENT_PROMPT_SYSTEM_EN.format(time = self.time)
            user_prompt = MULTI_TURN_AGENT_PROMPT_USER_EN.format(functions = self.functions, history = history)

        if "o1" not in self.model_name:
            message = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]

            response = self.client.chat.completions.create(
                messages=message,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            response = response.choices[0].message.content
        else:
            message = [{
                "role": "user",
                "content": system_prompt+"\n\n"+user_prompt,
            }]
            response = self.client.chat.completions.create(
                messages=message,
                model=self.model_name,
            )
            response = response.choices[0].message.content
            
        current_message["sender"] = "agent"

        pattern = r"\[.*?\]"
        match = re.match(pattern, response)

        if match:
            current_message["recipient"] = "execution"
        else:
            current_message["recipient"] = "user"
        current_message["message"] = response

        return current_message