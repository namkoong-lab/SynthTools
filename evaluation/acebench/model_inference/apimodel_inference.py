from model_inference.base_inference import BaseHandler


from model_inference.prompt_zh import *
from model_inference.prompt_en import *
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
import re
from model_inference.multi_turn.APIModel_agent import APIAgent_turn
from model_inference.multi_turn.APIModel_user import APIUSER
from model_inference.multi_turn.execution_role import EXECUTION
from model_inference.multi_turn.multi_turn_scene import Scene
from model_inference.multi_step.multi_step_scene import Mulit_Step_Scene
from model_inference.multi_step.APIModel_agent import APIAgent_step
from model_inference.multi_step.execution_role_step import EXECUTION_STEP

SAVED_CLASS = {
                "BaseApi": ["wifi","logged_in"],
                "MessageApi": ["inbox"],
                "ReminderApi": ["reminder_list"],
                "FoodPlatform":["users","logged_in_users","orders"],
                "Finance":["user_accounts", "is_logged_in","deposit_history","withdrawal_history","loan_history","orders","holdings"],
                "Travel": ["users","reservations"],
               }




class APIModelInference(BaseHandler):
    def __init__(self, model_name, model_path=None, temperature=0.001, top_p=1, max_tokens=1000, max_dialog_turns=40, user_model="gpt-4o", language="zh") -> None:
        super().__init__(model_name, model_path, temperature, top_p, max_tokens, language)

        load_dotenv()

        if "gpt" in self.model_name:
            api_key = os.getenv("GPT_AGENT_API_KEY")
            base_url = os.getenv("GPT_BASE_URL")
        elif "deepseek-r1" in self.model_name:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL")
        elif "o1" in self.model_name:
            api_key = os.getenv("GPT_AGENT_API_KEY")
            base_url = os.getenv("GPT_BASE_URL")
            
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_dialog_turns = max_dialog_turns
        self.language = language
        self.user_model = user_model


    def inference(self, question, functions, time, profile, test_case, id):
        category = id.rsplit("_", 1)[0]
        if "agent" in category :
            initial_config = test_case["initial_config"]
            involved_classes = test_case["involved_classes"]
            test_id = test_case["id"].split("_")[-1]
            if "multi_turn" in category:
                result, process_list = self.multi_turn_inference(question, initial_config, functions, involved_classes, test_id, time)
            elif "multi_step" in category:
                result, process_list = self.multi_step_inference(question, initial_config, functions, involved_classes, test_id, time)
            return result, process_list

        else:
            result = self.single_turn_inference(question, functions, category, time, profile, id)
            return result


    def single_turn_inference(self, question, functions, test_category, time, profile, id):

        if self.language == "zh":
            if "special" in test_category:
                system_prompt = SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH.format(time = time ,function = functions)
            elif "preference" in test_category:
                system_prompt = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_ZH.format(profile = profile ,function = functions)
            else:
                system_prompt = SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH.format(time = time ,function = functions)
            user_prompt = USER_PROMPT_ZH.format(question = question)

        elif self.language == "en":
            if "special" in test_category:
                system_prompt = SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN.format(time=time, function=functions)
                
            elif "preference" in test_category:
                system_prompt = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN.format(profile=profile, function=functions)
            else:
                system_prompt = SYSTEM_PROMPT_FOR_NORMAL_DATA_EN.format(time = time ,function = functions)
            user_prompt = USER_PROMPT_EN.format(question=question)

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
        
        attempt = 0
        while attempt < 6:
            try:
                response = self.client.chat.completions.create(
                    messages=message,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )
                result = response.choices[0].message.content

                if "deepseek-r1" in self.model_name:
                    match = re.search(r'</think>\s*(.*)$', result, re.DOTALL)
                    result = match.group(1).strip()
                break  # If successful, break the loop
            except Exception as e:
                attempt += 1
                # Check if it's a specific error type, skip current iteration
                if 'data_inspection_failed' in str(e):
                    print(id)
                    continue  # Skip current iteration, continue to next attempt
                elif attempt == 6:
                    raise e  # If maximum attempts reached, raise exception

        return result
    

    def multi_turn_inference(self, question, initial_config, functions, involved_classes, test_id, time):

        agent = APIAgent_turn(model_name=self.model_name, time=time, functions=functions, involved_class=involved_classes, language=self.language)
        user = APIUSER(model_name=self.user_model, involved_class=involved_classes, language=self.language)
        execution = EXECUTION(agent_model_name=self.model_name, initial_config=initial_config, involved_classes=involved_classes, test_id=test_id, language=self.language)

        init_message = user.get_init_prompt(question)
        
        scene = Scene(initial_state=initial_config, functions=functions, agent_role=agent, user_role=user, init_message=init_message, language=  self.language)
        message_history = scene.dialogue_history
        result_list = []
        
        result_instance_list = []
        mile_stone = []
        with tqdm(total=self.max_dialog_turns, desc="Processing Messages") as pbar:
            for index in range(self.max_dialog_turns):

                last_recipient = message_history[-1]["recipient"]
                if last_recipient == "user":
                    inference_message = scene.get_inference_message()
                    user.step(message_history[-1]["message"])
                    current_message = user.respond()
                elif last_recipient == "agent":
                    inference_message = scene.get_inference_message()
                    current_message = agent.respond(inference_message)
                else:
                    # Catch exceptions from execution.respond(message_history)
                    inference_message = scene.get_inference_message()
                    mile_stone_message = message_history[-1]["message"]
                    mile_stone.append(mile_stone_message)
                    current_message, result_instance = execution.respond(message_history)
                    if isinstance(result_instance,dict):
                        if result_instance not in result_instance_list:
                            result_instance_list.append(result_instance)

                scene.add_dialogue(current_message)

                if index > 1 and "finish conversation" in current_message["message"]:
                    break
                pbar.update(1)
            scene.write_message_history(test_id, self.model_name)
            
        
        
        for result_instance in result_instance_list:
            for name, instance in result_instance.items():
                item_dict = {}
                for item in instance.__dict__:
                    if item in SAVED_CLASS[name]:
                        item_dict[item] = instance.__dict__[item]
                result_list.append({name: item_dict})

        # Return instance names for subsequent testing of property conformance
        return result_list, mile_stone
    

    def multi_step_inference(self, question, initial_config, functions, involved_classes, test_id, time):
        agent = APIAgent_step(model_name=self.model_name, time=time, functions=functions)
        scene = Mulit_Step_Scene(question = question, initial_state=initial_config, functions=functions, agent_role=agent, language=self.language)
        execution = EXECUTION_STEP(agent_model_name=self.model_name, initial_config=initial_config, involved_classes=involved_classes, test_id=test_id, language=self.language)
        message_history = scene.dialogue_history
        result_list = []
        
        result_instance_list = []
        mile_stone = []
        with tqdm(total=self.max_dialog_turns, desc="Processing Messages") as pbar:
            for index in range(self.max_dialog_turns):
                
                last_sender = message_history[-1]["sender"]
                if index == 0 or last_sender == "execution":
                    inference_message = scene.get_inference_message()
                    current_message = agent.respond(inference_message)
                else:
                    # Catch exceptions from execution.respond(message_history)
                    current_message, result_instance = execution.respond(message_history)
                    mile_stone_message = message_history[-1]["message"]
                    mile_stone.append(mile_stone_message)
                    if result_instance not in result_instance_list:
                        result_instance_list.append(result_instance)

                scene.add_dialogue(current_message)

                if index > 1 and "finish conversation" in current_message["message"]:
                    break
                pbar.update(1)

            
        scene.write_message_history(test_id, self.model_name)
        
        for result_instance in result_instance_list:
            for name, instance in result_instance.items():
                item_dict = {}
                for item in instance.__dict__:
                    if item in SAVED_CLASS[name]:
                        item_dict[item] = instance.__dict__[item]
                result_list.append({name: item_dict})

        # Return instance names for subsequent testing of property conformance
        return result_list, mile_stone


    
