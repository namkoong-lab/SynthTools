from openai import OpenAI
import os
import re 
import ast
from model_inference.prompt_en import TRAVEL_PROMPT_EN, BASE_PROMPT_EN
from model_inference.prompt_zh import TRAVEL_PROMPT_ZH, BASE_PROMPT_ZH


MULTI_TURN_AGENT_PROMPT_SYSTEM_ZH = """你是一个AI系统，你的角色为system，请根据给定的API说明和对话历史1..t，为角色system生成在步骤t+1中生成相应的内容。
1 如果上一步提供的信息完整，能够正常进行api的调用，你应该调用的API请求，API请求以[ApiName(key1='value1', key2='value2', ...)]的格式输出,不要在输出中输出任何其他解释或提示或API调用的结果。
如果API参数描述中没有特殊说明，则该参数为非必选参数（用户输入中提及的参数需要包含在输出中，如果未提及，则不需要包含在输出中）。\n如果API参数描述未指定取值格式要求，则该参数取值使用用户原文。
2 如果你得到的信息不完整，需要向user提问，以获得完整的信息。你不能扮演user去回答一些文职的问题，应该及时像user询问。

请注意，如果需要进行api调用，请严格遵守调用规则[ApiName(key1='value1', key2='value2', ...)]，此时不得输出其他文本内容。

角色说明：
user: 用户 
agent: 进行API请求调用的AI系统角色 
execution: 执行api调用并返回结果

你需要遵循的规则如下：\n
"""

MULTI_TURN_AGENT_PROMPT_USER_ZH = """下面是你可使用的api列表:\n {functions}\n\n对话历史1..t:\n{history}"""

MULTI_TURN_AGENT_PROMPT_SYSTEM_EN = """You are an AI system with the role name "system." Based on the provided API specifications and conversation history from steps 1 to t, generate the appropriate content for step t+1 for the "system" role.
1. If the information provided in the previous step is complete and the API call can be executed normally, you should generate the API request. The API request should be output in the format [ApiName(key1='value1', key2='value2', ...)]. Do not include any other explanations, prompts, or API call results in the output.
   - If the API parameter description does not specify otherwise, the parameter is optional (parameters mentioned in the user input need to be included in the output; if not mentioned, they do not need to be included).
   - If the API parameter description does not specify the required format for the value, use the user's original text for the parameter value.
2. If the information you received is incomplete, you need to ask the user for more information to obtain the complete details. You should not pretend to be the user to answer some clerical questions; instead, promptly ask the user for clarification.

Please note that if an API call is required, strictly adhere to the call format rules [ApiName(key1='value1', key2='value2', ...)] and do not output any other text content.

Role Descriptions:
user: User
agent: The AI system role that makes API requests
execution: Executes the API call and returns the result

The rules you need to follow are as follows:\n
"""

MULTI_TURN_AGENT_PROMPT_USER_EN = """Below is the list of APIs you can use:\n {functions}\n\nConversation history 1..t:\n{history}"""

class APIAgent_turn():

    def __init__(self, model_name, time, functions, involved_class, temperature=0.001, top_p=1, max_tokens=1000, language="zh") -> None:
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
        self.involved_class = involved_class
        self.language = language
        self.model_name = model_name

    def decode_function_list(self, result):
        func = result
        if " " == func[0]:
            func = func[1:]
        if not func.startswith("["):
            func = "[" + func
        if not func.endswith("]"):
            func = func + "]"
        decoded_output = self.ast_parse(func)
        return self.decoded_output_to_execution_list(decoded_output)

    def ast_parse(self, input_str, language="Python"):
        if language == "Python":
            cleaned_input = input_str.strip("[]'")  
            parsed = ast.parse(cleaned_input, mode="eval")
            extracted = []
            
            
            if isinstance(parsed.body, ast.Call):
                extracted.append(self.resolve_ast_call(parsed.body))
            elif isinstance(parsed.body, (ast.Tuple, ast.List)):  
                for elem in parsed.body.elts:
                    if isinstance(elem, ast.Call):
                        extracted.append(self.resolve_ast_call(elem))
                    else:
                        return False
            return extracted
        
    def resolve_ast_call(self,elem):
        # Handle nested attributes for deeply nested module paths
        func_parts = []
        func_part = elem.func
        while isinstance(func_part, ast.Attribute):
            func_parts.append(func_part.attr)
            func_part = func_part.value
        if isinstance(func_part, ast.Name):
            func_parts.append(func_part.id)
        func_name = ".".join(reversed(func_parts))
        args_dict = {}
        for arg in elem.keywords:
            output = self.resolve_ast_by_type(arg.value)
            args_dict[arg.arg] = output
        return {func_name: args_dict}


    def resolve_ast_by_type(self,value):
        if isinstance(value, ast.Constant):
            if value.value is Ellipsis:
                output = "..."
            else:
                output = value.value
        elif isinstance(value, ast.UnaryOp):
            output = -value.operand.value
        elif isinstance(value, ast.List):
            output = [self.resolve_ast_by_type(v) for v in value.elts]
        elif isinstance(value, ast.Dict):
            output = {
                self.resolve_ast_by_type(k): self.resolve_ast_by_type(v)
                for k, v in zip(value.keys, value.values)
            }
        elif isinstance(
            value, ast.NameConstant
        ):  # Added this condition to handle boolean values
            output = value.value
        elif isinstance(
            value, ast.BinOp
        ):  # Added this condition to handle function calls as arguments
            output = eval(ast.unparse(value))
        elif isinstance(value, ast.Name):
            output = value.id
        elif isinstance(value, ast.Call):
            if len(value.keywords) == 0:
                output = ast.unparse(value)
            else:
                output = self.resolve_ast_call(value)
        elif isinstance(value, ast.Tuple):
            output = tuple(self.resolve_ast_by_type(v) for v in value.elts)
        elif isinstance(value, ast.Lambda):
            output = eval(ast.unparse(value.body[0].value))
        elif isinstance(value, ast.Ellipsis):
            output = "..."
        elif isinstance(value, ast.Subscript):
            try:
                output = ast.unparse(value.body[0].value)
            except:
                output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
        else:
            raise Exception(f"Unsupported AST type: {type(value)}")
        return output
    
    
    def decoded_output_to_execution_list(self,decoded_output):

        execution_list = []
        for function_call in decoded_output:
            for key, value in function_call.items():
                args_str = ", ".join(
                    f"{k}={self.parse_nested_value(v)}" for k, v in value.items()
                )
                execution_list.append(f"{key}({args_str})")
        return execution_list
    
    def parse_nested_value(self,value):

        if isinstance(value, dict):
            func_name = list(value.keys())[0]
            args = value[func_name]
            args_str = ", ".join(f"{k}={self.parse_nested_value(v)}" for k, v in args.items())
            return f"{func_name}({args_str})"
        return repr(value)



    def respond(self, history) -> None:

        current_message = {}

        if self.language == "zh":
            system_prompt = MULTI_TURN_AGENT_PROMPT_SYSTEM_ZH
            user_prompt = MULTI_TURN_AGENT_PROMPT_USER_ZH.format(functions = self.functions, history = history)
            if "Travel" in self.involved_class:
                system_prompt += TRAVEL_PROMPT_ZH
            if "BaseApi" in self.involved_class:
                system_prompt += BASE_PROMPT_ZH
        elif self.language == "en":
            system_prompt = MULTI_TURN_AGENT_PROMPT_SYSTEM_EN
            user_prompt = MULTI_TURN_AGENT_PROMPT_USER_EN.format(functions = self.functions, history = history)
            if "Travel" in self.involved_class:
                system_prompt += TRAVEL_PROMPT_EN
            if "BaseApi" in self.involved_class:
                system_prompt += BASE_PROMPT_EN

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

        match = re.search(r"\[.*\]", response)
        if match:
            try:
                self.decode_function_list(response)
                current_message["recipient"] = "execution"
                current_message["message"] = response
            except Exception as e:
                current_message["recipient"] = "user"
                current_message["message"] = response
        else:
            current_message["recipient"] = "user"
            current_message["message"] = response

        return current_message