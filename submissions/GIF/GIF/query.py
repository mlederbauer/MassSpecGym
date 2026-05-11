import sys

from openai import OpenAI
import json
import time
import subprocess


def get_masked_selfies_schema():
    return {
        "type": "object",
        "properties": {
          "completed_selfies": {
            "type": "string"
          }
        },
        "required": ["completed_selfies"],
        "additionalProperties": False
    }

def get_intensity_frag_prediction_schema():
    return {
        "type": "object",
        "properties": {
            "intensities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "mz": {"type": "number"},
                        # "frag": {"type": "string"},
                        "int": {"type": "integer", "description": "Intensity of peak on a scale of 1 to 10"}
                    },
                    # "required": ["frag", "mz", "int"],
                    "required": ["mz", "int"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["intensities"],
        "additionalProperties": False
    }

def get_intensity_prediction_schema():
    return {
        "type": "object",
        "properties": {
            "intensities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "mz": {"type": "number"},
                        "subformula": {"type": "string"},
                        "int": {"type": "integer", "description": "Intensity of peak on a scale of 1 to 10"}
                    },
                    "required": ["subformula", "mz", "int"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["intensities"],
        "additionalProperties": False
    }

def get_fragment_list_prediction_schema():
    return {
        "type": "object",
        "properties": {
            "fragments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        # "mz": {"type": "number"},
                        "frag": {"type": "string", "description": "fragment in SELFIES format of given molecule"}
                    },
                    # "required": ["mz", "fragment"]
                    "required": ["frag"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["fragments"],
        "additionalProperties": False
    }

def get_subformula_prediction_schema():
    return {
        "type": "object",
        "properties": {
            "subformulae": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        # "mz": {"type": "number"},
                        "subformula": {"type": "string", "description": "subformula label of mass spectra peak of given molecule"}
                    },
                    # "required": ["mz", "fragment"]
                    "required": ["subformula"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["subformulae"],
        "additionalProperties": False
    }

def get_masked_intensity_schema():
    return {
        "type": "object",
        "properties": {
            "completed_spectrum": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "mz": {"type": "number"},
                        "frag": {"type": "string"},
                        "int": {
                            "type": "integer",# "minimum": 1, "maximum": 10
                            "description": "Intensity of peak on a scale of 1 to 10"
                            # "oneOf": [
                            #     {"type": "integer", "minimum": 1, "maximum": 10},
                            #     {"type": "string", "const": "[MASK]"}
                            # ]
                        }
                    },
                    "required": ["frag", "mz", "int"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["completed_spectrum"],
        "additionalProperties": False
    }

def get_masked_fragment_schema():
    return {
        "type": "object",
        "properties": {
            "completed_fragments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        # "mz": {"type": "number"},
                        "frag": {
                            "type": "string",
                            "description": "fragment in SELFIES format of given molecule"
                            # "oneOf": [
                            #     {"type": "string"},
                            #     {"type": "string", "const": "[MASK]"}
                            # ]
                        }
                    },
                    "required": ["frag"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["completed_fragments"],
        "additionalProperties": False
    }

def query_model_feedback(client, model_name, test_data, temperature = 0.1):
    time.sleep(2)

    if model_name[:2] == 'o4':
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=test_data['messages']
            )
        except:
            response = {}
    else:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=test_data['messages'],
                temperature=temperature
            )
        except:
            response = {}

    return response

def query_model(client, model_name, task_name, test_data, temperature = 0.1):
    time.sleep(2)
    task_schemas = {
        "MaskedSELFIES": get_masked_selfies_schema(),
        "IntensityFragPrediction": get_intensity_frag_prediction_schema(),
        "FragmentListPrediction": get_fragment_list_prediction_schema(),
        "MaskedIntensity": get_masked_intensity_schema(),
        "MaskedFragment": get_masked_fragment_schema(),
        "SubformulaPrediction": get_subformula_prediction_schema(),
        "IntensityPrediction": get_intensity_prediction_schema(),
        "IterativeFragmentListPrediction": get_fragment_list_prediction_schema(),
        "IterativeIntensityFragPrediction": get_intensity_frag_prediction_schema(),
    }

    if model_name[:6] == 'gpt-5':
        temperature = 1.0

    if task_name not in task_schemas:
        raise ValueError(f"Unsupported task: {task_name}")

    schema = task_schemas[task_name]

    if model_name[:2] == 'o4':
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=test_data['messages'],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"{task_name}_response",
                        "strict": True,
                        "schema": schema
                    }
                }
            )
        except:
            response = {}
    else:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=test_data['messages'],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"{task_name}_response",
                        "strict": True,
                        "schema": schema
                    }
                },
                temperature=temperature
                # # timeout in seconds #TODO
                # request_timeout = 30
            )
        except:
            response = {}



    return response

def mp_query(api_key, model_name, task, data, save_path, temperature=None, save_task="None"):
    p_start = time.time()
    PYTHON = sys.executable
    if temperature==None:
        cmd = [PYTHON, "single_query.py", "--api_key", api_key, "--model_name", model_name, "--task", task, "--save_path", save_path, "--save_task", save_task, "--from-stdin"]
    else:
        cmd = [PYTHON, "single_query.py", "--api_key", api_key, "--model_name", model_name, "--task", task, "--save_path", save_path, "--save_task", save_task, "--temperature", str(temperature), "--from-stdin"]
    proc = subprocess.run(cmd, input=json.dumps(data), text=True, capture_output=True, check=False)

    return

