import json
import jsonschema
from dataset_export.tool_schemas import to_json_schema, tool_to_function

TOOL = {
    "tool_name": "Save",
    "tool_description": "Save items.",
    "parameters": {
        "query": {"type": "string", "required": True, "description": "Query."},
        "limit": {"type": "integer", "required": False, "default": 10, "minimum": 1, "maximum": 100},
        "ratio": {"type": "float", "required": False, "default": 0.5},
        "mode": {"type": "string", "required": False, "enum": ["a", "b"], "format": "x"},
        "items": {"type": "array", "required": True, "minItems": 1,
                  "items": {"type": "object", "properties": {
                      "id": {"type": "string", "required": True},
                      "score": {"type": "float", "required": False},
                      "tags": {"type": "array", "items": {"type": "float"}}}}},
        "profile": {"type": "object", "required": ["age", "city"],
                    "properties": {"age": {"type": "integer"}, "city": {"type": "string"}, "zip": {"type": "string", "required": True}}},
        "weights": {"type": "object", "additionalProperties": {"type": "float"}},
        "flags": {"type": "object", "additionalProperties": False},
        "maybe": {"type": ["float", "null"], "nullable": True, "default": None},
        "ref": {"$ref": "#/definitions/thing"},
        "anything": {},
    },
    "error_messages": ["Boom"], "usage": "Do it.", "output_details": {"x": {"type": "float"}},
}


def test_tool_to_function_shape_and_required():
    fn = tool_to_function(TOOL)
    assert fn["type"] == "function" and fn["function"]["name"] == "Save" and fn["function"]["description"] == "Save items."
    p = fn["function"]["parameters"]
    assert p["type"] == "object" and p["required"] == ["query", "items"]
    assert "required" not in p["properties"]["query"]
    assert p["properties"]["limit"] == {"type": "integer", "default": 10, "minimum": 1, "maximum": 100}
    assert p["properties"]["ratio"] == {"type": "number", "default": 0.5}
    assert p["properties"]["mode"] == {"type": "string", "enum": ["a", "b"], "format": "x"}
    text = json.dumps(fn)
    for leaked in ("error_messages", "usage", "output_details", "Boom", '"float"'):
        assert leaked not in text


def test_nested_items_properties_and_flags():
    items = tool_to_function(TOOL)["function"]["parameters"]["properties"]["items"]
    assert items["type"] == "array" and items["minItems"] == 1
    inner = items["items"]
    assert inner["type"] == "object" and inner["required"] == ["id"]
    assert inner["properties"]["score"] == {"type": "number"}
    assert inner["properties"]["tags"] == {"type": "array", "items": {"type": "number"}}


def test_list_required_is_kept_and_merged_with_flags():
    prof = tool_to_function(TOOL)["function"]["parameters"]["properties"]["profile"]
    assert prof["required"] == ["age", "city", "zip"]
    assert "required" not in prof["properties"]["zip"]
    # the parameter itself carried no boolean flag, so it is not required at the tool level
    assert "profile" not in tool_to_function(TOOL)["function"]["parameters"]["required"]


def test_additional_properties_type_lists_refs_and_passthrough():
    p = tool_to_function(TOOL)["function"]["parameters"]["properties"]
    assert p["weights"] == {"type": "object", "additionalProperties": {"type": "number"}}
    assert p["flags"] == {"type": "object", "additionalProperties": False}
    assert p["maybe"] == {"type": ["number", "null"], "nullable": True, "default": None}
    assert p["ref"] == {"$ref": "#/definitions/thing"} and p["anything"] == {}


def test_list_required_without_properties_passes_through():
    assert to_json_schema({"type": "object", "required": ["a"]}) == {"type": "object", "required": ["a"]}


def test_result_is_a_valid_json_schema():
    jsonschema.Draft202012Validator.check_schema(tool_to_function(TOOL)["function"]["parameters"])


def test_empty_and_missing_parameters():
    assert tool_to_function({"tool_name": "X"})["function"]["parameters"] == {"type": "object", "properties": {}}


def test_boolean_exclusive_bounds_become_numeric():
    assert to_json_schema({"type": "float", "minimum": 0.0, "exclusiveMinimum": True}) == {"type": "number", "exclusiveMinimum": 0.0}
    assert to_json_schema({"type": "integer", "maximum": 10, "exclusiveMaximum": False}) == {"type": "integer", "maximum": 10}
    assert to_json_schema({"type": "integer", "exclusiveMaximum": True}) == {"type": "integer"}
    jsonschema.Draft202012Validator.check_schema(to_json_schema({"type": "float", "minimum": 0.0, "exclusiveMinimum": True}))
