
from model_inference.multi_step.multi_step_utils import *
import ast


class EXECUTION_STEP():

    def __init__(self, agent_model_name, initial_config, involved_classes, test_id, language) -> None:

        self.agent_model_name = agent_model_name
        self.initial_config = initial_config
        self.involved_classes = involved_classes
        self.test_id = test_id
        self.language = language

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
            cleaned_input = input_str.strip("[]'")  # Clean input as needed
            parsed = ast.parse(cleaned_input, mode="eval")
            extracted = []
            
            # Check if it's a single function call or tuple/list
            if isinstance(parsed.body, ast.Call):
                extracted.append(self.resolve_ast_call(parsed.body))
            elif isinstance(parsed.body, (ast.Tuple, ast.List)):  # Modified to support tuple or list
                for elem in parsed.body.elts:
                    if isinstance(elem, ast.Call):
                        extracted.append(self.resolve_ast_call(elem))
                    else:
                        raise ValueError("Element is not a function call")
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
        current_message["sender"] = "execution"
        current_message["recipient"] = "agent"
        message = history[-1]["message"]
        try:
            function_call_list = self.decode_function_list(message)
        except Exception as e:
            current_message["message"] = "Please do not ask me any questions, use the known conditions to solve the problem"
            return current_message,{}  # Return error message when exception is caught

        single_turn_execution_results, result_instances = (
            execute_agent_func_call(
                func_call_list=function_call_list,
                initial_config=self.initial_config,
                involved_classes=self.involved_classes,
                model_name=self.agent_model_name,
                test_entry_id= self.test_id,
                language=self.language,
            )
        )

    
        parsed_results = []
        for item in single_turn_execution_results:
            try:
                parsed_item = json.loads(item)
                parsed_results.append(parsed_item)
            except json.JSONDecodeError as e:
                parsed_results.append(item)

        current_message["message"] = parsed_results

        return current_message, result_instances