#!/usr/bin/env python3
"""
Test script for JudgeSimulatorClient to verify it's properly handling template formatting
and API calls for judging tool simulator responses.
"""

import json
from pathlib import Path
from simulator.judge_simulator import JudgeSimulatorClient, JudgmentResult


def test_judge_client():
    """Test the JudgeSimulatorClient with sample data"""
    
    print("=" * 60)
    print("Testing JudgeSimulatorClient")
    print("=" * 60)
    
    sample_tool_config = {
        "tool_name": "WiFi_Tool",
        "tool_description": "A tool for managing WiFi connections",
        "parameters": {
            "ssid": {
                "type": "string",
                "description": "The SSID of the WiFi network"
            },
            "password": {
                "type": "string", 
                "description": "The password for the WiFi network"
            }
        },
        "states": {
            "ssid_available": ["Home WiFi", "Office WiFi", "Guest WiFi"],
            "ssid_password": {
                "Home WiFi": "home_wifi_password",
                "Office WiFi": "office_wifi_password", 
                "Guest WiFi": "guest_wifi_password"
            }
        }
    }
    
    print("Sample tool config:")
    print(json.dumps(sample_tool_config, indent=2))
    print()
    
    
    test_cases = [
        {
            "name": "Valid WiFi connection - should succeed",
            "user_message": "Connect to Home WiFi with password home_wifi_password",
            "tool_call": "WiFi_Tool[ssid='Home WiFi', password='home_wifi_password']",
            "simulator_response": "Successfully connected to Home WiFi",
            "expected_judgment": "Should be CORRECT"
        },
        {
            "name": "Valid WiFi but wrong password - should fail", 
            "user_message": "Connect to Home WiFi with password wrong_password",
            "tool_call": "WiFi_Tool[ssid='Home WiFi', password='wrong_password']",
            "simulator_response": "Successfully connected to Home WiFi",
            "expected_judgment": "Should be SHOULD_ERROR_GOT_SUCCESS"
        },
        {
            "name": "Invalid SSID - should fail",
            "user_message": "Connect to NonExistent WiFi with password anything",
            "tool_call": "WiFi_Tool[ssid='NonExistent WiFi', password='anything']",
            "simulator_response": "Error: SSID not found",
            "expected_judgment": "Should be CORRECT"
        },
        {
            "name": "Valid parameters but simulator returns error",
            "user_message": "Connect to Office WiFi with password office_wifi_password",
            "tool_call": "WiFi_Tool[ssid='Office WiFi', password='office_wifi_password']",
            "simulator_response": "Error: Connection failed", 
            "expected_judgment": "Should be SHOULD_SUCCESS_GOT_ERROR"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print("-" * 50)
        print(f"Tool Call: {test_case['tool_call']}")
        print(f"Simulator Response: {test_case['simulator_response']}")
        print(f"Expected: {test_case['expected_judgment']}")
        print()
        

        try:
            sample_tool_config["user_message"] = test_case['user_message']
            sample_tool_config["tool_call_message"] = test_case['tool_call']
            sample_tool_config["simulator_response"] = test_case['simulator_response']
            print(sample_tool_config)
            judge_client = JudgeSimulatorClient(
                prompt_template_file=Path(__file__).parent / "prompt_templates" / "judge_template.yml"
            )
            judgment, confidence, reasoning = judge_client.judge_tool_response(
                user_message=test_case['user_message'],
                tool_call_message=test_case['tool_call'],
                simulator_response=test_case['simulator_response'], 
                tool_config=sample_tool_config
            )
            
            print(f"API call successful")
            print(f"Judgment: {judgment.value}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Reasoning: {reasoning}")
            
        except Exception as e:
            print(f"✗ API call failed: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        print("=" * 60)
        print()


def test_template_formatting():
    """Test that the template formatting works correctly without API calls"""
    
    print("Testing Template Formatting (No API calls)")
    print("=" * 60)
    
    try:
        judge_client = JudgeSimulatorClient(
            tool_config_file=Path(__file__).parent / "tool_configs" / "example_wifi_tool.json",
            prompt_template_file=Path(__file__).parent / "prompt_templates" / "judge_template.yml"
        )
        
        evaluation_data = {
            'tool_name': 'Test Tool',
            'tool_description': 'A test tool',
            'parameters': {'param1': 'value1'},
            'states': {'state1': 'active'},
            'tool_call_message': 'test call',
            'simulator_response': 'test response'
        }
        
        formatted_prompt = judge_client.prompt_template.format(**evaluation_data)
        
        print("Template formatting successful")
        print("Formatted prompt:")
        print("-" * 40)
        print(formatted_prompt)
        print("-" * 40)
        
        for key in evaluation_data:
            if f"{{{key}}}" in formatted_prompt:
                print(f"✗ Template variable {{{key}}} not substituted!")
                return
        
        print("All template variables properly substituted")
        
    except Exception as e:
        print(f"✗ Template formatting failed: {e}")


if __name__ == "__main__":
    
    print("WARNING: The following test will make actual API calls to Anthropic")
    user_input = input("Continue? (y/N): ")
    
    if user_input.lower() in ['y', 'yes']:
        test_judge_client()
    else:
        print("Skipping API call tests")