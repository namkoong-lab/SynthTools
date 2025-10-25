
import json
import os
from wcwidth import wcswidth

class Mulit_Step_Scene:
    def __init__(self, question, initial_state, functions, agent_role, language):
        self.initial_state = initial_state     
        self.dialogue_history = [{"sender": "user", "recipient": "agent", "message": question}]  # Initialize the first message as dialogue history
        self.final_state = None 
        self.functions = functions              
        self.agent_role = agent_role
        self.inference_data = ""
        self.question = question
        self.language = language
    
    
    def get_inference_message(self):
        if self.dialogue_history[-1]["sender"] == "user":
            self.inference_data += "user:" + str(self.dialogue_history[-1]["message"]) + "\n"
        elif self.dialogue_history[-1]["sender"] == "agent":
            self.inference_data += "agent:" + self.dialogue_history[-1]["message"] + "\n"
        elif self.dialogue_history[-1]["sender"] == "execution":
            self.inference_data += "execution result:" + str(self.dialogue_history[-1]["message"]) + "\n"
        return self.inference_data



    def add_dialogue(self, dialogue):
        self.dialogue_history.append(dialogue)  # Add dialogue to history

    def set_final_state(self, final_state):
        pass


    
    def ljust_with_width(self, s, width):
        """Left align based on character display width"""
        fill_width = width - wcswidth(s)
        return s + ' ' * fill_width

    def write_message_history(self, test_id, model_name):
        # Define fixed column widths
        index_width = 15
        sender_width = 10
        recipient_width = 25
        content_width = 120

        # Build table header
        header = f"| {'message_index'.ljust(index_width)} | {'sender'.ljust(sender_width)} | {'recipient'.ljust(recipient_width)} | {'content'.ljust(content_width)} |"
        separator = f"|{'-' * (index_width + 2)}|{'-' * (sender_width + 2)}|{'-' * (recipient_width + 2)}|{'-' * (content_width + 2)}|"

        # Build table content
        rows = []
        for i, dialogue in enumerate(self.dialogue_history):
            sender = self.ljust_with_width(dialogue['sender'], sender_width)
            recipient = self.ljust_with_width(dialogue['recipient'], recipient_width)

            # Handle message field, if it's a list, merge it into a string
            content = dialogue['message']
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, dict):
                        # Convert dictionary key-value pairs to string
                        dict_str = " ".join([f"{k}: {v}" for k, v in item.items()])
                        new_content.append(dict_str)
                    else:
                        new_content.append(str(item))  # If not a dictionary, convert directly to string
                content = " ".join(new_content)
            elif not isinstance(content, str):
                content = str(content)

            # If content is escaped Unicode characters, process with json.loads
            try:
                content = json.loads(f'"{content}"')
            except json.JSONDecodeError:
                pass  # If parsing fails, keep original content

            # If content exceeds specified width, handle line wrapping
            wrapped_content = []
            while wcswidth(content) > content_width:
                wrapped_content.append(self.ljust_with_width(content[:content_width], content_width))
                content = content[content_width:]
            wrapped_content.append(self.ljust_with_width(content, content_width))

            # Add all segmented content to rows
            first_row = f"| {str(i).ljust(index_width)} | {sender} | {recipient} | {wrapped_content[0]} |"
            rows.append(first_row)
            for extra_line in wrapped_content[1:]:
                extra_row = f"| {' '.ljust(index_width)} | {' '.ljust(sender_width)} | {' '.ljust(recipient_width)} | {extra_line} |"
                rows.append(extra_row)

        # Join table content
        table_content = header + "\n" + separator + "\n" + "\n".join(rows)

        # Set output filename and path
        # Check if directory exists, create if not
        if self.language == "zh":
            directory = f"./data_all/data_zh/dialogue_history/multi_step/{model_name}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_name = f"{test_id}_dialogue_history.txt"
            file_path = f"{directory}/{file_name}"
        elif self.language == "en":
            directory = f"./data_all/data_en/dialogue_history/multi_step/{model_name}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_name = f"{test_id}_dialogue_history.txt"
            file_path = f"{directory}/{file_name}"

        # Write table to txt file
        with open(file_path, 'w', encoding="utf-8") as f:
            f.write(table_content)
