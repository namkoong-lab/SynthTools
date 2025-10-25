from model_inference.base_inference import BaseHandler

from model_inference.prompt_zh import *
from model_inference.prompt_en import *

from tqdm import tqdm

from model_inference.multi_turn.common_agent import CommonAgent
from model_inference.model_infer import get_model
from model_inference.multi_turn.APIModel_user import APIUSER
from model_inference.multi_turn.execution_role import EXECUTION
from model_inference.multi_turn.multi_turn_scene import Scene
from model_inference.multi_step.multi_step_scene import Mulit_Step_Scene
from model_inference.multi_step.common_agent_step import CommonAgent_Step
from model_inference.multi_step.execution_role_step import EXECUTION_STEP

SAVED_CLASS = {
                "BaseApi": ["wifi","logged_in"],
                "MessageApi": ["inbox"],
                "ReminderApi": ["reminder_list"],
                "FoodPlatform":["users","logged_in_users","orders"],
                "Finance":["user_accounts", "is_logged_in","deposit_history","withdrawal_history","loan_history","orders","holdings"],
                "Travel": ["users","reservations"],
               }




class CommonInference(BaseHandler):
    def __init__(self, model_name, model_path=None, temperature=0.001, top_p=1, max_tokens=1000, max_dialog_turns=40, user_model="gpt-4o", language="zh") -> None:
        super().__init__(model_name, model_path, temperature, top_p, max_tokens, language)

        self.model_name = model_name
        self.model_path = model_path
        self.max_message_index = max_dialog_turns
        self.language = language
        self.user_model = user_model
        self.tokenizer = self.initialize_tokenizer(model_path)
        self.model = get_model(model_name=model_name, model_path=model_path)

    def initialize_tokenizer(self, model_path):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer
      
    def inference(self, question, functions, time, profile, test_case, id):
        category = id.rsplit("_", 1)[0]
        if "multi_turn" in category and "agent" in category:
            initial_config = test_case["initial_config"]
            involved_classes = test_case["involved_classes"]
            test_id = test_case["id"].split("_")[-1]
            result, process_list = self.multi_turn_inference(question, initial_config, functions, involved_classes, test_id, time)
            return result, process_list
        elif "multi_step" in category:
            initial_config = test_case["initial_config"]
            involved_classes = test_case["involved_classes"]
            test_id = test_case["id"].split("_")[-1]
            result, process_list = self.multi_step_inference(question, initial_config, functions, involved_classes, test_id, time)
            return result,process_list
        else:
            result = self.single_turn_inference(question, functions, time, profile, id)
        
        return result
    
    def single_turn_inference(self, question, functions, time, profile, id):
        category = id.rsplit("_", 1)[0]
        if self.language == "zh":
            if "special" in category:
                system_prompt = SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH.format(time = time ,function = functions)
            elif "preference" in category:
                system_prompt = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_ZH.format(profile = profile ,function = functions)
            else:
                system_prompt = SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH.format(time = time ,function = functions)
            user_prompt = USER_PROMPT_ZH.format(question = question)

        elif self.language == "en":
            if "special" in category:
                system_prompt = SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN.format(time=time, function=functions)
                
            elif "preference" in category:
                system_prompt = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN.format(profile=profile, function=functions)
            else:
                system_prompt = SYSTEM_PROMPT_FOR_NORMAL_DATA_EN.format(time=time, function=functions)
            user_prompt = USER_PROMPT_EN.format(question=question)

      
        result = self.model.inference(system_prompt, user_prompt)
        return result

    def multi_turn_inference(self, question, initial_config, functions, involved_classes, test_id, time):
        agent = CommonAgent(model=self.model, time=time, functions=functions, involved_class=involved_classes, language=self.language)
        user = APIUSER(model_name=self.user_model, involved_class=involved_classes, language=self.language)
        execution = EXECUTION(agent_model_name=self.model_name, initial_config=initial_config, involved_classes=involved_classes, test_id=test_id, language=self.language)
        
        init_message = user.get_init_prompt(question)

        scene = Scene(initial_state=initial_config, functions=functions, agent_role=agent, user_role=user, init_message=init_message, language=self.language)
        message_history = scene.dialogue_history
        result_list = []
        
        result_instance_list = []
        mile_stone = []
        with tqdm(total=self.max_message_index, desc="Processing Messages") as pbar:
            for index in range(self.max_message_index):
              
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

        # Return the instance name - its properties will be tested against expectations later
        return result_list, mile_stone
    

    def multi_step_inference(self, question, initial_config, functions, involved_classes, test_id, time):
        agent = CommonAgent_Step(model = self.model, language=self.language, functions = functions)
        execution = EXECUTION_STEP(agent_model_name = self.model_name, initial_config = initial_config, involved_classes = involved_classes, test_id = test_id, language=self.language )

        scene = Mulit_Step_Scene(question=question, initial_state=initial_config, functions = functions, agent_role = agent, language = self.language)
        
        message_history = scene.dialogue_history
        result_list = []
        
        result_instance_list = []
        mile_stone = []
        with tqdm(total=self.max_message_index, desc="Processing Messages") as pbar:
            for index in range(self.max_message_index):
                last_sender = message_history[-1]["sender"]
                if index == 0 or last_sender == "execution":
                    inference_message = scene.get_inference_message()
                    current_message = agent.respond(inference_message)
                else:
                    
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
        
        # Return the instance name - its properties will be tested against expectations later
        return result_list, mile_stone
    

    
