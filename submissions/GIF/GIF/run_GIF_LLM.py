import os
import sys
import pickle
import csv
import json
import statistics
import torch
import json
import random
import copy
import yaml
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM, GenerationConfig
# from transformers import AutoTokenizer, AutoModelForCausalLM
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

from peft import PeftModel
import subprocess
import yaml
from pathlib import Path
import argparse


from prompts import get_prompt_templates, get_generated_fragments, get_generated_fragments_smi
from preprocess_data import load_test_data, load_train_data, load_val_data, add_method_features, load_cands_dict
from query import get_fragment_list_prediction_schema, get_intensity_frag_prediction_schema, get_subformula_prediction_schema
from metrics import subformula_prediction_evaluation, intensity_frag_prediction_evaluation, fragment_list_prediction_evaluation, masked_intensity_evaluation, masked_fragment_evaluation, cumulative_fragment_list_prediction_evaluation, baseline_metrics, calculate_regen_mz, smi_fragment_list_prediction_evaluation
from metrics import is_substructure, is_substructure_smi, selfies_to_smiles



# parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", default="llama")
parser.add_argument("--data_path", default='./data/')
parser.add_argument("--save_path", default='./output/')
parser.add_argument("--temperature", default=0.9)
parser.add_argument("--disable_training", action="store_true", default=False)
parser.add_argument("--disable_inference", action="store_true", default=False)
parser.add_argument("--disable_evaluation", action="store_true", default=False)
parser.add_argument("--disable_iterate", action="store_true", default=False)
args = vars(parser.parse_args())

model_type = args["model_type"]
data_path = args["data_path"]
save_path = args["save_path"]
temp = float(args["temperature"])
training = not args["disable_training"]
inference = not args["disable_inference"]
evaluation = not args["disable_evaluation"]
iterate = not args["disable_iterate"]


tasks = ["FragmentListPrediction"]


if args["model_type"] == "llama":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # Initialize model
    encoding = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16)
    if training:
        gpt_train_file = "./data/processed/prompts/prompts_with_metadata/train_queries_IntensityFragPrediction_subset_1kpts_auto.jsonl"
        gpt_val_file = "./data/processed/prompts/prompts_with_metadata/val_queries_IntensityFragPrediction_subset_1kpts_auto.jsonl"
        suffix = '1kpts_auto'
        len_limits = 1000
    if inference:
        # intensity_model = "finetuned_models/llama3-lora-1kpts_auto"
        intensity_model = "base"
elif args["model_type"] == "chemdfm":
    model_name = "OpenDFM/ChemDFM-v1.0-13B"
    encoding = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    if training:
        gpt_train_file = "./data/processed/prompts/prompts_with_metadata/train_queries_IntensityFragPrediction_subset_1kpts_auto.jsonl"
        gpt_val_file = "./data/processed/prompts/prompts_with_metadata/val_queries_IntensityFragPrediction_subset_1kpts_auto.jsonl"
        suffix = '1kpts_auto'
        len_limits = 1000
    if inference:
        intensity_model = "base"
else:
    sys.exit("Exiting due to invalid model type.")

if not os.path.isdir('./output/'):
    os.mkdir('./output/')
if not os.path.isdir('./output/'+model_type):
    os.mkdir('./output/'+model_type)
if not os.path.isdir(save_path):
    os.mkdir(save_path)
if not os.path.isdir(save_path+'raw/'):
    os.mkdir(save_path+'raw/')
if not os.path.isdir(save_path+'raw/json/'):
    os.mkdir(save_path+'raw/json/')
if not os.path.isdir(save_path+'raw/pkl/'):
    os.mkdir(save_path+'raw/pkl/')
if not os.path.isdir(save_path+'raw/txt/'):
    os.mkdir(save_path+'raw/txt/')
for task in tasks:
    if not os.path.isdir(save_path + 'raw/json/'+task):
        os.mkdir(save_path + 'raw/json/'+task)
for task in tasks:
    if not os.path.isdir(save_path + 'raw/pkl/'+task):
        os.mkdir(save_path + 'raw/pkl/'+task)
for task in tasks:
    if not os.path.isdir(save_path + 'raw/txt/'+task):
        os.mkdir(save_path + 'raw/txt/'+task)
if not os.path.isdir(save_path+'postprocessed/'):
    os.mkdir(save_path+'postprocessed/')
if not os.path.isdir(save_path+'figures/'):
    os.mkdir(save_path+'figures/')


if not os.path.isdir(data_path+'processed/'):
    os.mkdir(data_path+'processed/')

if not os.path.isdir(data_path+'processed/' + model_type + '_prompts/'):
    os.mkdir(data_path+'processed/' + model_type + '_prompts/')

cands_dict = load_cands_dict(data_path)

def get_fragment_list_prediction_desired_format():
    return {"fragments":[{"frag": "..."},{"frag": "..."},{"frag": "..."},{"frag": "..."}]}

def get_intensity_frag_prediction_desired_format():
    return {"intensities":[{"mz": "...", "int": "..."},{"mz": "...", "int": "..."},{"mz": "...", "int": "..."},{"mz": "...", "int": "..."}]}

def dict_to_yaml(d: dict) -> str:
    return yaml.safe_dump(
        d,
        sort_keys=False,        # keep key order
        allow_unicode=True,     # don't escape non-ASCII
        default_flow_style=False,
        width=10_000            # avoid line wrapping
    )

def create_test_prompts(task, data, model_type):
    if model_type == "llama":
        full_prompt = "### System:\n[[SYSTEM_PROMPT]]\n\n### Instruction:\n[[USER_PROMPT]]\n\n### Response:"
    elif model_type == "chemdfm":
        full_prompt = "[Round 0]\nHuman: [[USER_PROMPT]]\nAssistant:"

    prompts = get_prompt_templates()
    queries = []
    if "FragmentListPrediction" == task:
        if model_type == "llama":
            schema = get_fragment_list_prediction_desired_format()
            json_instruction = " Respond only in the following JSON format (Do not include any commentary):\n" + json.dumps(schema, indent=2) + "\n"
        elif model_type == "chemdfm":
            json_instruction = " Respond only in the JSON schema format above (Do not include any commentary)."  # + json.dumps(schema, indent=2) + "\n"
        for pt in data:
            temp = copy.deepcopy(prompts[task])
            if model_type == "chemdfm":
                user_prompt = temp["user"].replace("[[SELFIES]]", pt["smiles"]).replace("[[EXP_SETTINGS]]",
                                                                                         dict_to_yaml(pt[
                                                                                                          "exp_settings"]).replace(
                                                                                             '"', ''))
                user_prompt = user_prompt.replace("SELFIES", "SMILES") #CHEMDFM uses SMILES
            else:
                user_prompt = temp["user"].replace("[[SELFIES]]", pt["selfies"]).replace("[[EXP_SETTINGS]]",
                                                                                     json.dumps(pt["exp_settings"],
                                                                                                indent=None).replace(
                                                                                         '"', ''))
            user_prompt = user_prompt + json_instruction
            if model_type == "llama":
                temp_prompt = full_prompt.replace("[[SYSTEM_PROMPT]]",temp["system"]).replace("[[USER_PROMPT]]",user_prompt)
            elif model_type == "chemdfm":
                user_prompt = temp["system"] + " " + user_prompt
                temp_prompt = full_prompt.replace("[[USER_PROMPT]]", user_prompt)
            data_point = {"message": temp_prompt,
                          "identifier": pt['identifier'],
                          "frags": pt['sorted_frags'],
                          "answer": json.dumps(pt["frags_str"], indent=None)}
            queries.append(data_point)
    elif task == "IntensityFragPrediction":
        if model_type == "llama":
            schema = get_intensity_frag_prediction_desired_format()
            json_instruction = " Respond only in the following JSON format (Do not include any commentary):\n" + json.dumps(
                schema, indent=2) + "\n"
        elif model_type =="chemdfm":
            json_instruction = " Respond only in the JSON schema format above (Do not include any commentary)."  # + json.dumps(schema, indent=2) + "\n"
        for pt in data:
            temp = copy.deepcopy(prompts[task])
            user_prompt = temp["user"].replace("[[SELFIES]]",pt["selfies"]).replace("[[EXP_SETTINGS]]",json.dumps(pt["exp_settings"], indent=None).replace('"','')).replace("[[FRAG_LIST_MZ]]", json.dumps(pt["frags_mz"], indent=None).replace('"',''))
            user_prompt = user_prompt + json_instruction
            assistant_prompt = temp["assistant"].replace("[[LIST_MZ_INT]]",json.dumps(pt["str_mz_int"], indent=None).replace('"',''))
            if model_type == "llama":
                temp_prompt = full_prompt.replace("[[SYSTEM_PROMPT]]", temp["system"]).replace("[[USER_PROMPT]]",
                                                                                           user_prompt)
            elif model_type == "chemdfm":
                user_prompt = temp["system"] + " " + user_prompt
                temp_prompt = full_prompt.replace("[[USER_PROMPT]]", user_prompt)
            data_point = {"message": temp_prompt,
                          "identifier": pt['identifier'],
                          "frags": pt['sorted_frags'],
                          "answer": assistant_prompt}
            queries.append(data_point)
    elif task == "SubformulaPrediction":
        if model_type == "llama":
            schema = get_fragment_list_prediction_desired_format()
            json_instruction = " Respond only in the following JSON format (Do not include any commentary):\n" + json.dumps(schema, indent=2) + "\n"
        elif model_type == "chemdfm":
            json_instruction = " Respond only in the JSON schema format above (Do not include any commentary)."  # + json.dumps(schema, indent=2) + "\n"
        for pt in data:
            temp = copy.deepcopy(prompts[task])
            user_prompt = temp["user"].replace("[[SMILES]]", pt["smiles"]).replace("[[FORMULA]]", pt["formula"]).replace("[[EXP_SETTINGS]]",
                                                                                     json.dumps(pt["exp_settings"],
                                                                                                indent=None).replace(
                                                                                         '"', ''))
            user_prompt = user_prompt + json_instruction
            if model_type == "llama":
                temp_prompt = full_prompt.replace("[[SYSTEM_PROMPT]]",temp["system"]).replace("[[USER_PROMPT]]",user_prompt)
            elif model_type == "chemdfm":
                user_prompt = temp["system"] + " " + user_prompt
                temp_prompt = full_prompt.replace("[[USER_PROMPT]]", user_prompt)
            data_point = {"message": temp_prompt,
                          "identifier": pt['identifier'],
                          "frags": pt['sorted_subforms'],#pt['sorted_frags'],
                          "answer": json.dumps(pt["subforms_str"], indent=None)}
            queries.append(data_point)


    return queries

def create_test_prompts_chemdfm(task, data, model_type):
    if model_type == "chemdfm":
        full_prompt = "[Round 0]\nHuman: [[USER_PROMPT]]\nAssistant:"
    prompts = get_prompt_templates()
    queries = []
    json_instruction = " Respond only in the JSON schema format above (Do not include any commentary)."  # + json.dumps(schema, indent=2) + "\n"
    if "FragmentListPrediction" == task:
        for pt in data:
            temp = copy.deepcopy(prompts[task])
            user_prompt = temp["user"].replace("[[SELFIES]]", pt["selfies"]).replace("[[EXP_SETTINGS]]",
                                                                                     json.dumps(pt["exp_settings"],
                                                                                                indent=None).replace(
                                                                                         '"', ''))
            user_prompt = user_prompt + json_instruction

            user_prompt = temp["system"] + " " + user_prompt
            temp_prompt = full_prompt.replace("[[USER_PROMPT]]",user_prompt)
            data_point = {"message": temp_prompt,
                          "identifier": pt['identifier'],
                          "frags": pt['sorted_frags'],
                          "answer": json.dumps(pt["frags_str"], indent=None)}
            queries.append(data_point)
    elif task == "IntensityFragPrediction":
        for pt in data:
            temp = copy.deepcopy(prompts[task])
            user_prompt = temp["user"].replace("[[SELFIES]]",pt["selfies"]).replace("[[EXP_SETTINGS]]",json.dumps(pt["exp_settings"], indent=None).replace('"','')).replace("[[FRAG_LIST_MZ]]", json.dumps(pt["frags_mz"], indent=None).replace('"',''))
            user_prompt = user_prompt + json_instruction
            user_prompt = temp["system"] + " " + user_prompt
            assistant_prompt = temp["assistant"].replace("[[LIST_MZ_INT]]",json.dumps(pt["str_mz_int"], indent=None).replace('"',''))
            temp_prompt = full_prompt.replace("[[USER_PROMPT]]",user_prompt)
            data_point = {"message": temp_prompt,
                          "identifier": pt['identifier'],
                          "frags": pt['sorted_frags'],
                          "answer": assistant_prompt}
            queries.append(data_point)

    return queries

def create_llama_test_prompts_from_gen_frags(response, data, model_type):
    task = "IntensityFragPrediction"
    if model_type == "llama":
        full_prompt = "### System:\n[[SYSTEM_PROMPT]]\n\n### Instruction:\n[[USER_PROMPT]]\n\n### Response:"
    elif model_type == "chemdfm":
        full_prompt = "[Round 0]\nHuman: [[USER_PROMPT]]\nAssistant:"
    prompts = get_prompt_templates()
    if model_type == "llama":
        schema = get_intensity_frag_prediction_desired_format()
        json_instruction = " Respond only in the following JSON format (Do not include any commentary):\n" + json.dumps(
            schema, indent=2) + "\n"
    elif model_type == "chemdfm":
        json_instruction = " Respond only in the JSON schema format above (Do not include any commentary)."  # + json.dumps(schema, indent=2) + "\n"

    if model_type == "chemdfm":
        frags_mz_str = get_generated_fragments_smi(response, data)
    else:
        frags_mz_str = get_generated_fragments(response, data)

    temp = copy.deepcopy(prompts[task])
    if model_type == "llama":
        user_prompt = temp["user"].replace("[[SELFIES]]", data["selfies"]).replace("[[EXP_SETTINGS]]",
                                                                              json.dumps(data["exp_settings"],
                                                                                         indent=None).replace('"',
                                                                                                              '')).replace(
       "[[FRAG_LIST_MZ]]", json.dumps(frags_mz_str, indent=None).replace('"', ''))
    elif model_type == "chemdfm":
        user_prompt = temp["user"].replace("[[SELFIES]]", data["smiles"]).replace("[[EXP_SETTINGS]]",
                                                                                   json.dumps(data["exp_settings"],
                                                                                              indent=None).replace('"',
                                                                                                                   '')).replace(
            "[[FRAG_LIST_MZ]]", json.dumps(frags_mz_str, indent=None).replace('"', ''))

    user_prompt = user_prompt + json_instruction
    if model_type == "llama":
        temp_prompt = full_prompt.replace("[[SYSTEM_PROMPT]]", temp["system"]).replace("[[USER_PROMPT]]", user_prompt)
    elif model_type == "chemdfm":
        user_prompt = temp["system"] + " " + user_prompt
        temp_prompt = full_prompt.replace("[[USER_PROMPT]]", user_prompt)

    assistant_prompt = temp["assistant"].replace("[[LIST_MZ_INT]]",
                                                 json.dumps(data["str_mz_int"], indent=None).replace('"', ''))
    data_point = {"message": temp_prompt,
                  "identifier": data['identifier'],
                  "frags": data['sorted_frags'],
                  "answer": assistant_prompt}

    return data_point

def create_train_prompts(task, data, len_limit = None):
    full_prompt = "### System:\n[[SYSTEM_PROMPT]]\n\n### Instruction:\n[[USER_PROMPT]]\n\n### Response:"
    prompts = get_prompt_templates()
    queries = []
    if "FragmentListPrediction" == task:
        schema = get_fragment_list_prediction_desired_format()
        json_instruction = " Respond only in the following JSON format (Do not include any commentary):\n" + json.dumps(schema, indent=2) + "\n"
        for pt in data:
            temp = copy.deepcopy(prompts[task])
            user_prompt = temp["user"].replace("[[SELFIES]]", pt["selfies"]).replace("[[EXP_SETTINGS]]",
                                                                                     json.dumps(pt["exp_settings"],
                                                                                                indent=None).replace(
                                                                                         '"', ''))
            user_prompt = user_prompt + json_instruction
            assistant_prompt = temp["assistant"].replace("[[FRAGMENTS]]",
                                                         json.dumps(pt["frags_str"], indent=None).replace('"', ''))
            temp_prompt = full_prompt.replace("[[SYSTEM_PROMPT]]",temp["system"]).replace("[[USER_PROMPT]]",user_prompt)
            data_point = {"instruction": temp_prompt,
                          "output": assistant_prompt}

            queries.append(data_point)
    elif task == "IntensityFragPrediction":
        schema = get_intensity_frag_prediction_desired_format()
        json_instruction = " Respond only in the following JSON format (Do not include any commentary):\n" + json.dumps(
            schema, indent=2) + "\n"
        for pt in data:
            temp = copy.deepcopy(prompts[task])
            user_prompt = temp["user"].replace("[[SELFIES]]",pt["selfies"]).replace("[[EXP_SETTINGS]]",json.dumps(pt["exp_settings"], indent=None).replace('"','')).replace("[[FRAG_LIST_MZ]]", json.dumps(pt["frags_mz"], indent=None).replace('"',''))
            user_prompt = user_prompt + json_instruction
            assistant_prompt = temp["assistant"].replace("[[LIST_MZ_INT]]",json.dumps(pt["str_mz_int"], indent=None).replace('"',''))
            temp_prompt = full_prompt.replace("[[SYSTEM_PROMPT]]", temp["system"]).replace("[[USER_PROMPT]]",
                                                                                           user_prompt)
            data_point = {"instruction": temp_prompt,
                          "output": assistant_prompt}

            queries.append(data_point)
            if len_limit != None:
                if len(queries) >= len_limit:
                    break


    return queries

def create_train_prompts_from_gpt(task, data, file, model_type="llama"):
    with open(file, 'r') as f:
        gpt_queries = [json.loads(line) for line in f]
    if model_type == "llama":
        full_prompt = "### System:\n[[SYSTEM_PROMPT]]\n\n### Instruction:\n[[USER_PROMPT]]\n\n### Response:"
    elif model_type == "chemdfm":
        full_prompt = "[[USER_PROMPT]]"
    prompts = get_prompt_templates()
    queries = []
    if "FragmentListPrediction" == task:
        assert model_type == "llama"
        schema = get_fragment_list_prediction_desired_format()
        json_instruction = " Respond only in the following JSON format (Do not include any commentary):\n" + json.dumps(schema, indent=2) + "\n"
        for gpt_pt in gpt_queries:
            pt = [item for item in data if item.get("identifier") == gpt_pt["identifier"]]
            assert len(pt) == 1
            pt = pt[0]
            temp = copy.deepcopy(prompts[task])
            user_prompt = temp["user"].replace("[[SELFIES]]", pt["selfies"]).replace("[[EXP_SETTINGS]]",
                                                                                     json.dumps(pt["exp_settings"],
                                                                                                indent=None).replace(
                                                                                         '"', ''))
            user_prompt = user_prompt + json_instruction
            assistant_prompt = temp["assistant"].replace("[[FRAGMENTS]]",
                                                         json.dumps(pt["frags_str"], indent=None).replace('"', ''))
            temp_prompt = full_prompt.replace("[[SYSTEM_PROMPT]]",temp["system"]).replace("[[USER_PROMPT]]",user_prompt)
            data_point = {"instruction": temp_prompt,
                          "output": assistant_prompt}
            queries.append(data_point)
    elif task == "IntensityFragPrediction":
        if model_type == "llama":
            schema = get_intensity_frag_prediction_desired_format()
            json_instruction = " Respond only in the following JSON format (Do not include any commentary):\n" + json.dumps(
                schema, indent=2) + "\n"
        elif model_type == "chemdfm":
            json_instruction = " Respond only in the JSON schema format above (Do not include any commentary)."
        for gpt_pt in gpt_queries:
            pt = [item for item in data if item.get("identifier") == gpt_pt["identifier"]]
            assert len(pt) == 1
            pt = pt[0]

            temp = copy.deepcopy(prompts[task])
            if model_type == "llama":
                user_prompt = temp["user"].replace("[[SELFIES]]",pt["selfies"]).replace("[[EXP_SETTINGS]]",json.dumps(pt["exp_settings"], indent=None).replace('"','')).replace("[[FRAG_LIST_MZ]]", json.dumps(pt["frags_mz"], indent=None).replace('"',''))
                user_prompt = user_prompt + json_instruction
                temp_prompt = full_prompt.replace("[[SYSTEM_PROMPT]]", temp["system"]).replace("[[USER_PROMPT]]",
                                                                                               user_prompt)
            elif model_type == "chemdfm":
                frags_mz_smi = []
                for frag_entry in pt["frags_mz"]:
                    smi_frag = selfies_to_smiles(frag_entry['frag'])
                    frags_mz_smi.append({'frag':smi_frag, 'mz':frag_entry['mz']})

                user_prompt = temp["user"].replace("[[SELFIES]]",pt["smiles"]).replace("[[EXP_SETTINGS]]",json.dumps(pt["exp_settings"], indent=None).replace('"','')).replace("[[FRAG_LIST_MZ]]", json.dumps(frags_mz_smi, indent=None).replace('"',''))
                user_prompt = user_prompt + json_instruction
                user_prompt = temp["system"] + " " + user_prompt
                temp_prompt = full_prompt.replace("[[USER_PROMPT]]", user_prompt)

            assistant_prompt = temp["assistant"].replace("[[LIST_MZ_INT]]",json.dumps(pt["str_mz_int"], indent=None).replace('"',''))

            data_point = {"instruction": temp_prompt,
                          "output": assistant_prompt}
            queries.append(data_point)



    return queries

def query_llama(model, message, temp=0.1):
    inputs = encoding(message, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        eos_token_id=encoding.eos_token_id,
        pad_token_id=encoding.pad_token_id,
        temperature=temp,
        top_p=0.9
    )

    assert len(outputs) == 1
    decoded = encoding.decode(outputs[0], skip_special_tokens=True)
    json_candidate = decoded.split("### Response:")[-1].strip()
    try:
        return json.loads(json_candidate), decoded
    except json.JSONDecodeError:
        try:
            json_candidate = json_candidate.split('###')[0].strip()
            return json.loads(json_candidate), decoded
        except json.JSONDecodeError:
            try:
                return json.loads(json_candidate + '}'), decoded
            except json.JSONDecodeError:
                try:
                    return json.loads(json_candidate + '"}'), decoded
                except json.JSONDecodeError:
                    try:
                        return json.loads(json_candidate + '"}]}'), decoded
                    except json.JSONDecodeError:
                        if "###" in json_candidate:
                            try:
                                return json.load(json_candidate.split('###')[0].strip()), decoded
                            except json.JSONDecodeError:
                                return {}, decoded
                        else:
                            return {}, decoded


def query_general_hf_model(model, message, task, encoding, model_type, temp=0.1):
    if "FragmentListPrediction" in task:
        schema = get_fragment_list_prediction_schema()
    elif "IntensityFragPrediction" in task:
        schema = get_intensity_frag_prediction_schema()
    elif "SubformulaPrediction" in task:
        schema = get_subformula_prediction_schema()
    parser = JsonSchemaParser(schema)
    prefix_fn = build_transformers_prefix_allowed_tokens_fn(encoding, parser)

    inputs = encoding(message, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1256,
        do_sample=True,
        temperature=temp,
        prefix_allowed_tokens_fn=prefix_fn
    )

    assert len(outputs) == 1
    decoded = encoding.decode(outputs[0], skip_special_tokens=True)
    if model_type == "chemdfm":
        json_candidate = decoded.split("Assistant:")[-1].strip()
    try:
        return json.loads(json_candidate), decoded
    except json.JSONDecodeError:
        try:
            json_candidate = json_candidate.split('###')[0].strip()
            return json.loads(json_candidate), decoded
        except json.JSONDecodeError:
            try:
                return json.loads(json_candidate + '}'), decoded
            except json.JSONDecodeError:
                try:
                    return json.loads(json_candidate + '"}'), decoded
                except json.JSONDecodeError:
                    try:
                        return json.loads(json_candidate + '"}]}'), decoded
                    except json.JSONDecodeError:
                        if "###" in json_candidate:
                            try:
                                return json.load(json_candidate.split('###')[0].strip()), decoded
                            except json.JSONDecodeError:
                                return {}, decoded
                        else:
                            return {}, decoded

def iterative_run(previous_task, new_task, type_of_task, dataset, save_path, data_path, client, model_name, temperature, model_type, encoding):
    queries = []
    for data in dataset:
        if os.path.isfile(
                save_path + 'raw/json/' + previous_task + '/' + data[
                    'identifier'] + '_' + previous_task + '_model_response.json'):

            with open(save_path + 'raw/json/' + previous_task + '/' + data[
                'identifier'] + '_' + previous_task + '_model_response.json',
                      'r') as file:
                response = json.load(file)

            # process response
            gen_frags = []
            if 'fragments' in response.keys():
                for frag in response['fragments']:
                    gen_frags.append(frag['frag'])

            true_substructure = []
            if model_type == "chemdfm":
                for frag in gen_frags:
                    if is_substructure_smi(frag, data['smiles']):
                        true_substructure.append(frag)
            else:
                for frag in gen_frags:
                    if is_substructure(frag, data['smiles']):
                        true_substructure.append(frag)

            not_true_substructure = len(gen_frags) - len(true_substructure)

            new_frags_str = []
            for frag in true_substructure:
                new_frags_str.append({"frag": frag})

            # create new query
            temp = {
                "system": "You are a chemistry model specialized in mass spectrometry. In this task, predict the most likely mass spectrometry fragments based on molecular structure and fragmentation patterns.",
                "user": "<<TASK=IterativeFragmentListPrediction>> You predicted all major fragments (ordered by descending intensity) in SELFIES format based on this information:\n<<MOL>> [[SELFIES]]\n<<EXP_SETTINGS>> [[EXP_SETTINGS]]\n\n [[INVALID_SUBSTRUCTURES]] of the predicted fragments were invalid substructures, and the remaining are possible: <<FRAGMENTS>> [[PREVIOUS FRAGMENTS]]. Respond with the final list of fragments of the molecule (ordered by descending intensity) in SELFIES format produced by these experiment settings.",
                "assistant": "[[FRAGMENTS]]"
            }
            if model_type == "llama":
                full_prompt = "### System:\n[[SYSTEM_PROMPT]]\n\n### Instruction:\n[[USER_PROMPT]]\n\n### Response:"
                schema = get_fragment_list_prediction_desired_format()
                json_instruction = " Respond only in the following JSON format (Do not include any commentary):\n" + json.dumps(
                    schema, indent=2) + "\n"
                user_prompt = temp["user"].replace("[[INVALID_SUBSTRUCTURES]]", str(not_true_substructure)).replace(
                    "[[PREVIOUS FRAGMENTS]]", str(new_frags_str)).replace("[[SELFIES]]", data["selfies"]).replace(
                    "[[EXP_SETTINGS]]",
                    json.dumps(data["exp_settings"],
                               indent=None).replace(
                        '"', ''))
                user_prompt = user_prompt + json_instruction
                temp_prompt = full_prompt.replace("[[SYSTEM_PROMPT]]", temp["system"]).replace("[[USER_PROMPT]]",
                                                                                               user_prompt)
            elif model_type == "chemdfm":
                full_prompt = "[Round 0]\nHuman: [[USER_PROMPT]]\nAssistant:"
                json_instruction = " Respond only in the JSON schema format above (Do not include any commentary)."
                user_prompt = temp["user"].replace("[[INVALID_SUBSTRUCTURES]]", str(not_true_substructure)).replace(
                    "[[PREVIOUS FRAGMENTS]]", str(new_frags_str)).replace("[[SELFIES]]", data["smiles"]).replace(
                    "[[EXP_SETTINGS]]",
                    json.dumps(data["exp_settings"],
                               indent=None).replace(
                        '"', ''))
                user_prompt = user_prompt.replace("SELFIES", "SMILES")
                user_prompt = user_prompt + json_instruction
                user_prompt = temp["system"] + " " + user_prompt
                temp_prompt = full_prompt.replace("[[USER_PROMPT]]", user_prompt)

            data_point = {"message": temp_prompt,
                          "identifier": data['identifier'],
                          "frags": data['sorted_frags'],
                          "answer": json.dumps(data["frags_str"], indent=None)}
            queries.append(data_point)

            if not os.path.exists(save_path + 'raw/json/' + new_task + '/' + data[
                'identifier'] + '_' + new_task + '_model_response.json') and not os.path.exists(save_path + 'raw/json/' + new_task + '/' + data[
                'identifier'] + '_' + type_of_task + '_model_response.json'):

                # submit and save query
                if model_type == "llama":
                    response, full_response = query_llama(base_model, data_point['message'], temperature)
                elif model_type == "chemdfm":
                    response, full_response = query_general_hf_model(base_model, data_point['message'], type_of_task, encoding, model_type, temperature)
                else:
                    sys.exit("ERROR no model")
                with open(save_path + 'raw/pkl/' + new_task + '/' + data['identifier'] + '_' + new_task + '_model_response.pkl',
                          'wb') as file:
                    pickle.dump(response, file)
                with open(
                        save_path + 'raw/json/' + new_task + '/' + data['identifier'] + '_' + new_task + '_model_response.json',
                        "w") as file:
                    json.dump(response, file, indent=2)
                with open(save_path + 'raw/txt/' + new_task + '/' + data['identifier'] + '_' + new_task + '_model_response.txt',
                          "w") as f:
                    f.writelines(full_response)

    if not os.path.isdir(save_path+model_type+'_prompts/'):
        os.mkdir(save_path+model_type+'_prompts/')
    if not os.path.isdir(save_path+model_type+'_prompts/iterative_prompts/'):
        os.mkdir(save_path+model_type+'_prompts/iterative_prompts/')
    with open(save_path+model_type+'_prompts/iterative_prompts/test_queries_' + new_task + '.jsonl', "w") as f:
        for pt in queries:
            f.write(json.dumps(pt) + "\n")

    return

def iterate_fragmentation(dataset, save_path, data_path, base_model, model_name, model_type, encoding, n=4):
    type_of_task = 'IterativeFragmentListPrediction'

    temperature = 0.9
    for i in range(1, n + 1):
        if i == 4:
            temperature = 0.1
        if i == 1:
            previous_task = 'FragmentListPrediction'
        else:
            previous_task = 'IterativeFragmentListPrediction' + str(i - 1)
        new_task = 'IterativeFragmentListPrediction' + str(i)
        if not os.path.isdir(save_path + 'raw/json/' + new_task):
            os.mkdir(save_path + 'raw/json/' + new_task)
        if not os.path.isdir(save_path + 'raw/pkl/' + new_task):
            os.mkdir(save_path + 'raw/pkl/' + new_task)
        if not os.path.isdir(save_path + 'raw/txt/' + new_task):
            os.mkdir(save_path + 'raw/txt/' + new_task)
        iterative_run(previous_task, new_task, type_of_task, dataset, save_path, data_path, base_model, model_name,
                      temperature, model_type, encoding)

    return

def iterative_run_intensity(generation_task, previous_task, new_task, type_of_task, dataset, save_path, data_path, base_model, model_name, model_type, temperature):
    queries = []
    for data in dataset:
        if os.path.isfile(save_path + 'raw/json/' + generation_task + '/' + data[
            'identifier'] + '_' + generation_task + '_model_response.json') and os.path.isfile(
                save_path + 'raw/json/' + previous_task + '/' + data[
                    'identifier'] + '_' + previous_task + '_model_response.json'):

            with open(save_path + 'raw/json/' + generation_task + '/' + data[
                'identifier'] + '_' + generation_task + '_model_response.json') as file:
                gen_frag_response = json.load(file)


            if model_type == "chemdfm":
                gen_frags = get_generated_fragments_smi(gen_frag_response, data)
            else:
                gen_frags = get_generated_fragments(gen_frag_response, data)
            gen_frags_dict = {}
            for frag in gen_frags:
                gen_frags_dict[frag['mz']] = [frag['frag']]

            with open(save_path + 'raw/json/' + previous_task + '/' + data[
                'identifier'] + '_' + previous_task + '_model_response.json') as file:
                int_response = json.load(file)

            # process response
            mzs = []
            if 'intensities' in int_response.keys():
                for frag in int_response['intensities']:
                    if frag['mz'] in gen_frags_dict.keys() and frag['mz'] not in mzs:
                        if 'int' in frag.keys():
                            if isinstance(frag['int'], int) and frag['int'] >= 0 and frag['int'] <= 10:
                                mzs.append(frag['mz'])
                                gen_frags_dict[frag['mz']].append(frag['int'])

            new_str = []
            count = 0
            for k in gen_frags_dict.keys():
                if len(gen_frags_dict[k]) == 2:
                    new_str.append({'frag': gen_frags_dict[k][0], 'mz': k, 'int': gen_frags_dict[k][1]})
                else:
                    new_str.append({'frag': gen_frags_dict[k][0], 'mz': k})
                    count +=1

            # create new query
            if model_type == "llama":
                full_prompt = "### System:\n[[SYSTEM_PROMPT]]\n\n### Instruction:\n[[USER_PROMPT]]\n\n### Response:"
                schema = get_intensity_frag_prediction_desired_format()
                json_instruction = " Respond only in the following JSON format (Do not include any commentary):\n" + json.dumps(
                    schema, indent=2) + "\n"
            elif model_type == "chemdfm":
                full_prompt = "[Round 0]\nHuman: [[USER_PROMPT]]\nAssistant:"
                json_instruction = " Respond only in the JSON schema format above (Do not include any commentary)."  # + json.dumps(schema, indent=2) + "\n"
            temp = {
                "system": "You are a chemistry model specialized in mass spectrometry. In this task, estimate the intensity of fragments based on their mass, structure, and likely ionization behavior.",
                "user": "<<TASK=IterativeIntensityFragPrediction>> You predicted the intensity scores (1–10) for provided fragments (ordered by descending intensity). [[INVALID_RESPONSES]] were either incorrect or not included. Provide the intensity of each fragment of the molecules under the specified experiment settings and return the exact corresponding m/z values listed here too. The included intensity values were provided by you in the previous response:\n<<MOL>> [[SELFIES]]\n<<EXP_SETTINGS>> [[EXP_SETTINGS]]\n[[FRAG_LIST_MZ]]",
                "assistant": "[[LIST_MZ_INT]]"
                }
            if model_type == "llama":
                user_prompt = temp["user"].replace("[[INVALID_RESPONSES]]", str(count)).replace(
                    "[[FRAG_LIST_MZ]]", str(new_str)).replace("[[SELFIES]]", data["selfies"]).replace(
                    "[[EXP_SETTINGS]]",
                    json.dumps(data["exp_settings"],
                               indent=None).replace(
                        '"', ''))
                user_prompt = user_prompt + json_instruction
                temp_prompt = full_prompt.replace("[[SYSTEM_PROMPT]]", temp["system"]).replace("[[USER_PROMPT]]",
                                                                                               user_prompt)
            elif model_type == "chemdfm":
                user_prompt = temp["user"].replace("[[INVALID_RESPONSES]]", str(count)).replace(
                    "[[FRAG_LIST_MZ]]", str(new_str)).replace("[[SELFIES]]", data["smiles"]).replace(
                    "[[EXP_SETTINGS]]",
                    json.dumps(data["exp_settings"],
                               indent=None).replace(
                        '"', ''))

                user_prompt = user_prompt + json_instruction
                user_prompt = temp["system"] + " " + user_prompt
                temp_prompt = full_prompt.replace("[[USER_PROMPT]]", user_prompt)

            assistant_prompt = temp["assistant"].replace("[[LIST_MZ_INT]]",
                                                         json.dumps(data["str_mz_int"], indent=None).replace('"', ''))

            data_point = {"message": temp_prompt,
                          "identifier": data['identifier'],
                          "frags": data['sorted_frags'],
                          "answer": assistant_prompt}
            queries.append(data_point)

            if not os.path.exists(save_path + 'raw/json/' + new_task + '/' + data[
                'identifier'] + '_' + new_task + '_model_response.json') and not os.path.exists(save_path + 'raw/json/' + new_task + '/' + data[
                'identifier'] + '_' + type_of_task + '_model_response.json'):

                # submit and save query
                if model_type == "llama":
                    response, full_response = query_llama(base_model, data_point['message'], temperature)
                elif model_type == "chemdfm":
                    response, full_response = query_general_hf_model(base_model, data_point['message'], type_of_task, encoding,
                                                                     model_type, temperature)
                else:
                    sys.exit("ERROR NO MODEL")
                with open(save_path + 'raw/pkl/' + new_task + '/' + data['identifier'] + '_' + new_task + '_model_response.pkl',
                          'wb') as file:
                    pickle.dump(response, file)
                with open(
                        save_path + 'raw/json/' + new_task + '/' + data['identifier'] + '_' + new_task + '_model_response.json',
                        "w") as file:
                    json.dump(response, file, indent=2)
                with open(save_path + 'raw/txt/' + new_task + '/' + data['identifier'] + '_' + new_task + '_model_response.txt',
                          "w") as f:
                    f.writelines(full_response)

    if not os.path.isdir(save_path+model_type+'_prompts/'):
        os.mkdir(save_path+model_type+'_prompts/')
    if not os.path.isdir(save_path+model_type+'_prompts/iterative_prompts/'):
        os.mkdir(save_path+model_type+'_prompts/iterative_prompts/')
    with open(save_path+model_type+'_prompts/iterative_prompts/test_queries_' + new_task + '.jsonl', "w") as f:
        for pt in queries:
            f.write(json.dumps(pt) + "\n")

    return

def iterate_intensity(generation_task, dataset, save_path, data_path, base_model, model_name, model_type, n=1):
    type_of_task = 'IterativeIntensityFragPrediction'

    temperature = 0.1
    for i in range(1,n+1):

        if i==1:
            previous_task = 'IntensityFragPrediction'

        else:
            previous_task = 'IterativeIntensityFragPrediction' + str(i - 1)
        new_task = 'IterativeIntensityFragPrediction' + str(i)
        if not os.path.isdir(save_path + 'raw/json/' + new_task):
            os.mkdir(save_path + 'raw/json/' + new_task)
        if not os.path.isdir(save_path + 'raw/pkl/' + new_task):
            os.mkdir(save_path + 'raw/pkl/' + new_task)
        if not os.path.isdir(save_path + 'raw/txt/' + new_task):
            os.mkdir(save_path + 'raw/txt/' + new_task)
        iterative_run_intensity(generation_task, previous_task, new_task, type_of_task, dataset, save_path, data_path, base_model, model_name, model_type, temperature)

    return

def llama_fine_tune(model_path, dataset_path, val_dataset_path, suffix):
    data_info_json = {
      "dataset_training"+suffix: {
        "file_name": dataset_path.split('/')[-1],#dataset_path,
        "file_name_eval": val_dataset_path.split('/')[-1],#val_dataset_path,
        "columns": {
          "prompt": "instruction",
          "response": "output"
        }
      }
    }
    with open(
            dataset_path.rpartition("/")[0] + '/dataset_info.json',
            "w") as file:
        json.dump(data_info_json, file, indent=2)

    config = {
        "stage": "sft",
        "do_train": True,
        "model_name_or_path": model_path,
        "dataset": "dataset_training"+suffix,
        "dataset_dir": dataset_path.rpartition("/")[0],
        "template": "llama3",
        "finetuning_type": "lora",
        "lora_target": "q_proj,v_proj",
        "output_dir": "finetuned_models/llama3-lora-"+suffix,
        "per_device_train_batch_size": 4,
        "num_train_epochs": 3.0,
        "learning_rate": 1e-4,
        "val_size": 0.1,
        "logging_steps": 10,
        "save_steps": 500,
        "overwrite_output_dir": True,
        "bf16": True,
    }

    output_path = Path("train_config.yaml")
    output_path.write_text(yaml.dump(config, sort_keys=False))

    print(f"YAML config written to: {output_path}")

    print("Starting fine-tuning using LLaMA-Factory...")

    train_command = [
        "conda", "run", "-n", "llama-factory-env",
        "python", "LLaMA-Factory-main/src/train.py", "train_config.yaml"
    ]

    try:
        subprocess.run(train_command, check=True)
    except subprocess.CalledProcessError as e:
        print("Fine-tuning failed:", e)
    else:
        print("Fine-tuning completed successfully.")
    return

def chemdfm_fine_tune(model_path, dataset_path, val_dataset_path, suffix):
    data_info_json = {
      "dataset_training"+suffix: {
        "file_name": dataset_path.split('/')[-1],#dataset_path,
        "file_name_eval": val_dataset_path.split('/')[-1],#val_dataset_path,
        "formatting": "alpaca",
        "columns": {
          "prompt": "instruction",
          "response": "output"
        }
      }
    }
    with open(
            dataset_path.rpartition("/")[0] + '/dataset_info.json',
            "w") as file:
        json.dump(data_info_json, file, indent=2)


    config = {
        "stage": "sft",
        "do_train": True,
        "model_name_or_path": model_path,
        "dataset": "dataset_training" + suffix,  # must exist in your dataset_info.json
        "dataset_dir": dataset_path.rpartition("/")[0],

        "template": "chemdfm",

        "finetuning_type": "lora",
        "lora_target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",


        "output_dir": "finetuned_models/chemdfm-qlora-" + suffix,

        # Training knobs (safe starters for a 13B with QLoRA)
        "per_device_train_batch_size": 1,  # start small; scale up if VRAM allows
        "gradient_accumulation_steps": 16,  # effective batch size = 16
        "num_train_epochs": 3.0,
        "learning_rate": 1e-4,
        "cutoff_len": 2048,
        "packing": True,

        "val_size": 0,  # <- use eval file; no random split
        "logging_steps": 10,
        "save_steps": 500,
        "overwrite_output_dir": True,
        "bf16": False, #for mac
        "gradient_checkpointing": True,
    }

    output_path = Path("train_config_chemdfm.yaml")
    output_path.write_text(yaml.dump(config, sort_keys=False))

    print(f"YAML config written to: {output_path}")

    print("Starting fine-tuning using LLaMA-Factory...")

    cfg_path = Path.cwd() / "train_config_chemdfm.yaml"
    train_command = [
        "conda", "run", "-n", "biochemLLM2",
        "llamafactory-cli", "train",
        str(cfg_path.resolve())
    ]

    try:
        subprocess.run(train_command, check=True)
    except subprocess.CalledProcessError as e:
        print("Fine-tuning failed:", e)
    else:
        print("Fine-tuning completed successfully.")
    return




if training:
    if os.path.exists(data_path + "processed/train_data_with_method_features.pkl"):
        with open(data_path + "processed/train_data_with_method_features.pkl", "rb") as f:
            train_data = pickle.load(f)
    else:
        train_data = load_train_data(data_path)
        train_data = add_method_features(train_data)
        with open(data_path + "processed/train_data_with_method_features.pkl", "wb") as f:
            pickle.dump(train_data, f)

    # validation files
    if os.path.exists(data_path + "processed/val_data_with_method_features.pkl"):
        with open(data_path + "processed/val_data_with_method_features.pkl", "rb") as f:
            val_data = pickle.load(f)
    else:
        val_data = load_val_data(data_path)
        val_data = add_method_features(val_data)
        with open(data_path + "processed/val_data_with_method_features.pkl", "wb") as f:
            pickle.dump(val_data, f)

    #create data file
    # suffix = '_allpts_auto'
    if model_type == "llama":
        if not os.path.exists(
            data_path + 'processed/llama_prompts/llama_train_queries_' + 'IntensityFragPrediction_subset' + suffix + '.json'):
            if gpt_train_file == None:
                train_queries = create_train_prompts("IntensityFragPrediction", train_data, round(len_limits/10*9))
            else:
                train_queries = create_train_prompts_from_gpt("IntensityFragPrediction", train_data, gpt_train_file)
            with open(data_path + 'processed/llama_prompts/llama_train_queries_' + 'IntensityFragPrediction_subset' + suffix + '.json',
                      "w") as file:
                json.dump(train_queries, file, indent=2)
        if not os.path.exists(
            data_path + 'processed/llama_prompts/llama_val_queries_' + 'IntensityFragPrediction_subset' + suffix + '.json'):
            if gpt_val_file == None:
                val_queries = create_train_prompts("IntensityFragPrediction", val_data, round(len_limits/10))
            else:
                val_queries = create_train_prompts_from_gpt("IntensityFragPrediction", val_data, gpt_val_file)
            with open(data_path + 'processed/llama_prompts/llama_val_queries_' + 'IntensityFragPrediction_subset' + suffix + '.json',
                      "w") as file:
                json.dump(val_queries, file, indent=2)

    #start finetuning
        llama_fine_tune(model_name, data_path + 'processed/llama_prompts/llama_train_queries_' + 'IntensityFragPrediction_subset' + suffix + '.json', data_path + 'processed/llama_prompts/llama_val_queries_' + 'IntensityFragPrediction_subset' + suffix + '.json', suffix)
        intensity_model = "finetuned_models/llama3-lora-"+suffix
    elif model_type == "chemdfm":
        if not os.path.exists(
            data_path + 'processed/chemdfm_prompts/chemdfm_train_queries_' + 'IntensityFragPrediction_subset' + suffix + '.json'):
            train_queries = create_train_prompts_from_gpt("IntensityFragPrediction", train_data, gpt_train_file, model_type)
            with open(data_path + 'processed/chemdfm_prompts/chemdfm_train_queries_' + 'IntensityFragPrediction_subset' + suffix + '.json',
                      "w") as file:
                json.dump(train_queries, file, indent=2)
        if not os.path.exists(
            data_path + 'processed/chemdfm_prompts/chemdfm_val_queries_' + 'IntensityFragPrediction_subset' + suffix + '.json'):
            if gpt_val_file == None:
                val_queries = create_train_prompts("IntensityFragPrediction", val_data, round(len_limits/10))
            else:
                val_queries = create_train_prompts_from_gpt("IntensityFragPrediction", val_data, gpt_val_file, model_type)
            with open(data_path + 'processed/chemdfm_prompts/chemdfm_val_queries_' + 'IntensityFragPrediction_subset' + suffix + '.json',
                      "w") as file:
                json.dump(val_queries, file, indent=2)

        chemdfm_fine_tune(model_name, data_path + 'processed/chemdfm_prompts/chemdfm_train_queries_' + 'IntensityFragPrediction_subset' + suffix + '.json', data_path + 'processed/chemdfm_prompts/chemdfm_val_queries_' + 'IntensityFragPrediction_subset' + suffix + '.json', suffix)
        intensity_model = "finetuned_models/chemdfm-lora-"+suffix


#Load testing data

if os.path.exists(data_path + "processed/test_data_with_method_features.pkl"):
    with open(data_path + "processed/test_data_with_method_features.pkl", "rb") as f:
        test_data = pickle.load(f)
else:
    test_data = load_test_data(data_path)
    test_data = add_method_features(test_data)
    with open(data_path + "processed/test_data_with_method_features.pkl", "wb") as f:
        pickle.dump(test_data, f)
test_queries_dict = {}
for task in tasks:
    if os.path.exists(data_path + 'processed/' + model_type + '_prompts/test_queries_' + task + '.jsonl'):
        with open(data_path + 'processed/' + model_type + '_prompts/test_queries_' + task + '.jsonl', 'r') as f:
            test_queries = [json.loads(line) for line in f]
    else:
        if model_type == "llama" or model_type == "chemdfm":
            test_queries = create_test_prompts(task, test_data, model_type)
        else:
            sys.exit("ERROR no model")
        with open(data_path + 'processed/' + model_type + '_prompts/test_queries_' + task + '.jsonl', "w") as f:
            for pt in test_queries:
                f.write(json.dumps(pt) + "\n")
    test_queries_dict[task] = test_queries

for task, test_queries in test_queries_dict.items():
    count = 0
    for data in test_queries:
        if not os.path.exists(save_path + 'raw/json/' + task + '/' + data['identifier'] + '_' + task + '_model_response.json'):

            if model_type == "llama":
                response, full_response = query_llama(base_model, data['message'], temp)
            elif model_type == "chemdfm":
                response, full_response = query_general_hf_model(base_model, data['message'], task, encoding, model_type, temp)
            else:
                sys.exit("ERROR no model")
            with open(save_path + 'raw/pkl/' + task + '/' + data['identifier'] + '_' + task + '_model_response.pkl', 'wb') as file:
                pickle.dump(response, file)
            with open(save_path + 'raw/json/' + task + '/' + data['identifier'] + '_' + task + '_model_response.json', "w") as file:
                json.dump(response, file, indent=2)
            with open(save_path + 'raw/txt/' + task + '/' + data['identifier'] + '_' + task + '_model_response.txt', "w") as f:
                f.writelines(full_response)
        count +=1
        if count > 299:
            break


if iterate and model_type != "chemdfm":
    assert 'FragmentListPrediction' in tasks
    n=5
    iterate_fragmentation(test_data, save_path, data_path, base_model, model_name, model_type, encoding, n)
    generation_task = "IterativeFragmentListPrediction" +str(n)
else:
    generation_task = "FragmentListPrediction"

if inference:
    if intensity_model != "base":
        inference_model = PeftModel.from_pretrained(base_model, intensity_model)#Path(intensity_model))
    else:
        inference_model = base_model

    task = "IntensityFragPrediction"
    tasks.append(task)
    queries = []
    for data in test_data:
        if os.path.isfile(
                save_path + 'raw/json/' + generation_task + '/' + data[
                    'identifier'] + '_' + generation_task + '_model_response.json'):
            with open(save_path + 'raw/json/' + generation_task + '/' + data[
                'identifier'] + '_' + generation_task + '_model_response.json',
                      'r') as file:
                response = json.load(file)

            if model_type == "llama" or model_type == "chemdfm":
                single_query = create_llama_test_prompts_from_gen_frags(response, data, model_type)
            else:
                sys.exit("ERROR no model")
            queries.append(single_query)

    if not os.path.isdir(save_path + model_type + '_prompts/inference_intensity_prompts/'):
        os.mkdir(save_path + model_type + '_prompts/inference_intensity_prompts/')
    with open(save_path + model_type + '_prompts/inference_intensity_prompts/test_queries_' + task + '.jsonl', "w") as f:
        for pt in queries:
            f.write(json.dumps(pt) + "\n")
    test_queries_dict[task] = queries

# run queries
    if not os.path.isdir(save_path + 'raw/json/' + task):
        os.mkdir(save_path + 'raw/json/' + task)
    if not os.path.isdir(save_path + 'raw/pkl/' + task):
        os.mkdir(save_path + 'raw/pkl/' + task)
    if not os.path.isdir(save_path + 'raw/txt/' + task):
        os.mkdir(save_path + 'raw/txt/' + task)
    for data in queries:
        if not os.path.exists(
                save_path + 'raw/json/' + task + '/' + data['identifier'] + '_' + task + '_model_response.json'):
            if model_type == "llama":
                response, full_response = query_llama(inference_model, data['message'])
            elif model_type == "chemdfm":
                response, full_response = query_general_hf_model(inference_model, data['message'], task, encoding, model_type)
            else:
                sys.exit("ERROR no model")
            with open(save_path + 'raw/pkl/' + task + '/' + data[
                'identifier'] + '_' + task + '_model_response.pkl',
                      'wb') as file:
                pickle.dump(response, file)
            with open(
                    save_path + 'raw/json/' + task + '/' + data[
                        'identifier'] + '_' + task + '_model_response.json',
                    "w") as file:
                json.dump(response, file, indent=2)
            with open(save_path + 'raw/txt/' + task + '/' + data[
                'identifier'] + '_' + task + '_model_response.txt',
                      "w") as f:
                f.writelines(full_response)

    if iterate:
        int_n = 1
        iterate_intensity(generation_task, test_data, save_path, data_path, inference_model, model_name, model_type, n=int_n)
        iterative_intensity_task = 'IterativeIntensityFragPrediction' + str(int_n)

metrics = {}
for task in tasks:
    if task == "FragmentListPrediction":
        metrics[task] = {"sequence_level_accuracy":[], "overall_frag_accuracy":[], "formula_accuracy":[], "mass_accuracy":[], "validity":[], "true_substructure":[]}
    elif task == "IntensityFragPrediction":
        metrics[task] = {"sequence_level_accuracy":[], "gen_input_accuracy":[], "int_accuracy":[], "int_mse":[], "int_mae":[]}
    elif task == "SubformulaPrediction":
        metrics[task] = {"sequence_level_accuracy":[], "overall_sf_accuracy":[], "unordered_sf_accuracy":[], "unordered_sf_equivalence":[], "mass_accuracy":[], "actual_subformula":[], "validity":[]}

if evaluation:#run evaluation metrics with ground truth
    for data in test_data:
        for task in tasks:
            if os.path.isfile(save_path + 'raw/json/'+ task + '/' + data['identifier'] + '_' + task + '_model_response.json'):
                with open(save_path + 'raw/json/'+ task + '/' + data['identifier'] + '_' + task + '_model_response.json', 'r') as file:
                    response = json.load(file)
                if task == "FragmentListPrediction":
                    if model_type == "chemdfm":
                        sequence_level_accuracy, overall_frag_accuracy, formula_accuracy, mass_accuracy, validity, true_substructure = smi_fragment_list_prediction_evaluation(
                            response, data)
                    else:
                        sequence_level_accuracy, overall_frag_accuracy, formula_accuracy, mass_accuracy, validity, true_substructure = fragment_list_prediction_evaluation(response, data)
                    metrics[task]["sequence_level_accuracy"].append(sequence_level_accuracy)
                    metrics[task]["overall_frag_accuracy"].append(overall_frag_accuracy)
                    metrics[task]["formula_accuracy"].append(formula_accuracy)
                    metrics[task]["mass_accuracy"].append(mass_accuracy)
                    metrics[task]["validity"].extend(validity)
                    metrics[task]["true_substructure"].extend(true_substructure)
                elif task == "SubformulaPrediction":
                    sequence_level_accuracy, overall_sf_accuracy, unordered_sf_accuracy, unordered_sf_equivalence, mass_accuracy, actual_subformula, validity = subformula_prediction_evaluation(response, data)
                    metrics[task]["sequence_level_accuracy"].append(sequence_level_accuracy)
                    metrics[task]["overall_sf_accuracy"].append(overall_sf_accuracy)
                    metrics[task]["unordered_sf_accuracy"].append(unordered_sf_accuracy)
                    metrics[task]["unordered_sf_equivalence"].append(unordered_sf_equivalence)
                    metrics[task]["mass_accuracy"].append(mass_accuracy)
                    metrics[task]["actual_subformula"].append(actual_subformula)
                    metrics[task]["validity"].append(validity)
                elif task == "IntensityFragPrediction":
                    sequence_level_accuracy, gen_input_accuracy, int_accuracy, int_mse, int_mae = intensity_frag_prediction_evaluation(response, data)
                    metrics[task]["sequence_level_accuracy"].append(sequence_level_accuracy)
                    metrics[task]["gen_input_accuracy"].append(gen_input_accuracy)
                    metrics[task]["int_accuracy"].extend(int_accuracy)
                    metrics[task]["int_mse"].extend([int_mse])
                    metrics[task]["int_mae"].extend([int_mae])


    for task in tasks:
        if task == "FragmentListPrediction":
            print("FragmentListPrediction results over " + str(len(metrics[task]["validity"])) + " data points:")
            print("Sequence level accuracy: " + str(statistics.mean(metrics[task]["sequence_level_accuracy"])))
            print("Overall fragment accuracy: " + str(statistics.mean(metrics[task]["overall_frag_accuracy"])))
            print("Formula accuracy: " + str(
                statistics.mean(metrics[task]["formula_accuracy"])))
            print("Mass accuracy: " + str(
                statistics.mean(metrics[task]["mass_accuracy"])))
            print("Validity: " + str(statistics.mean(metrics[task]["validity"])))
            print("Proportion true substructure: " + str(statistics.mean(metrics[task]["true_substructure"])))
        elif task == "IntensityFragPrediction":
            print("IntensityFragPrediction results over " + str(len(metrics[task]["sequence_level_accuracy"])) + " data points:")
            print("Sequence level accuracy: " + str(statistics.mean(metrics[task]["sequence_level_accuracy"])))
            print("Regenerated input sequence accuracy: " + str(statistics.mean(metrics[task]["gen_input_accuracy"])))
            print("Intensity accuracy: " + str(statistics.mean(metrics[task]["int_accuracy"])))
            print("Intensity MSE: " + str(statistics.mean(metrics[task]["int_mse"])))
            print("Intensity MAE: " + str(statistics.mean(metrics[task]["int_mae"])))

if iterate and evaluation and model_type != "chemdfm":
    # n=1
    for i in range(1,n+1):

        task = 'IterativeFragmentListPrediction' + str(i)
        metrics[task] = {"sequence_level_accuracy": [], "overall_frag_accuracy": [], "formula_accuracy": [],
                         "mass_accuracy": [], "validity": [], "true_substructure": []}
        metrics["Cumulative" + task] = {"sequence_level_accuracy": [], "overall_frag_accuracy": [], "formula_accuracy": [],
                         "mass_accuracy": [], "validity": [], "true_substructure": []}
        for data in test_data:
            if os.path.isfile(
                    save_path + 'raw/json/' + task + '/' + data['identifier'] + '_' + task + '_model_response.json'):
                with open(save_path + 'raw/json/' + task + '/' + data['identifier'] + '_' + task + '_model_response.json',
                          'r') as file:
                    response = json.load(file)
                sequence_level_accuracy, overall_frag_accuracy, formula_accuracy, mass_accuracy, validity, true_substructure = fragment_list_prediction_evaluation(
                    response, data)
                metrics[task]["sequence_level_accuracy"].append(sequence_level_accuracy)
                metrics[task]["overall_frag_accuracy"].append(overall_frag_accuracy)
                metrics[task]["formula_accuracy"].append(formula_accuracy)
                metrics[task]["mass_accuracy"].append(mass_accuracy)
                metrics[task]["validity"].extend(validity)
                metrics[task]["true_substructure"].extend(true_substructure)


                sequence_level_accuracy, overall_frag_accuracy, formula_accuracy, mass_accuracy, validity, true_substructure = cumulative_fragment_list_prediction_evaluation(
                    task, data, save_path)
                metrics["Cumulative" + task]["sequence_level_accuracy"].append(sequence_level_accuracy)
                metrics["Cumulative" + task]["overall_frag_accuracy"].append(overall_frag_accuracy)
                metrics["Cumulative" + task]["formula_accuracy"].append(formula_accuracy)
                metrics["Cumulative" + task]["mass_accuracy"].append(mass_accuracy)
                metrics["Cumulative" + task]["validity"].extend(validity)
                metrics["Cumulative" + task]["true_substructure"].extend(true_substructure)

        print(task + " results over " + str(len(metrics[task]["sequence_level_accuracy"])) + " data points:")
        print("Sequence level accuracy: " + str(statistics.mean(metrics[task]["sequence_level_accuracy"])))
        print("Overall fragment accuracy: " + str(statistics.mean(metrics[task]["overall_frag_accuracy"])))
        print("Formula accuracy: " + str(
            statistics.mean(metrics[task]["formula_accuracy"])))
        print("Mass accuracy: " + str(
            statistics.mean(metrics[task]["mass_accuracy"])))
        print("Validity: " + str(statistics.mean(metrics[task]["validity"])))
        print("Proportion true substructure: " + str(statistics.mean(metrics[task]["true_substructure"])))

        print("Cumulative" + task + " results over " + str(len(metrics["Cumulative" + task]["sequence_level_accuracy"])) + " data points:")
        print("Sequence level accuracy: " + str(statistics.mean(metrics["Cumulative" + task]["sequence_level_accuracy"])))
        print("Overall fragment accuracy: " + str(statistics.mean(metrics["Cumulative" + task]["overall_frag_accuracy"])))
        print("Formula accuracy: " + str(
            statistics.mean(metrics["Cumulative" + task]["formula_accuracy"])))
        print("Mass accuracy: " + str(
            statistics.mean(metrics["Cumulative" + task]["mass_accuracy"])))
        print("Validity: " + str(statistics.mean(metrics["Cumulative" + task]["validity"])))
        print("Proportion true substructure: " + str(statistics.mean(metrics["Cumulative" + task]["true_substructure"])))


if inference:
    task = "IntensityFragPrediction"

    intensity_prompts = {}
    for prompt in test_queries_dict[task]:
        intensity_prompts[prompt['identifier']] = prompt


    cos_sims = []
    js_sims = []
    regen_mz_accuracy = []
    for data in test_data:
        if os.path.isfile(
                save_path + 'raw/json/' + task + '/' + data['identifier'] + '_' + task + '_model_response.json'):# and os.path.isfile(
            with open(save_path + 'raw/json/' + task + '/' + data['identifier'] + '_' + task + '_model_response.json',
                      'r') as file:
                int_response = json.load(file)

            cos_sim, js_sim = baseline_metrics(int_response, data)
            cos_sims.append(cos_sim)
            js_sims.append(js_sim)

            assert os.path.isfile(save_path + 'raw/json/' + generation_task + '/' + data['identifier'] + '_' + generation_task + '_model_response.json')
            with open(
                    save_path + 'raw/json/' + generation_task + '/' + data['identifier'] + '_' + generation_task + '_model_response.json',
                    'r') as file:
                frag_response = json.load(file)
            regen_mz_accuracy.append(calculate_regen_mz(frag_response, int_response, data))


    print("Average overall cosine similarity for " + str(len(cos_sims)) + " points: " + str(statistics.mean(cos_sims)))
    filtered_js_sims = [x for x in js_sims if x is not None]  # NEW
    print("Average overall Jensen-Shannon similarity for " + str(len(js_sims)) + " points: " + str(statistics.mean(filtered_js_sims)))
    print("Average overall regenerated m/z accuracy for " + str(len(regen_mz_accuracy)) + " points: " + str(statistics.mean(regen_mz_accuracy)))

    # plot_distribution(cos_sims, task, "Cosine similarity", save_path)

    if iterate:
        for i in range(1, int_n + 1):

            task = 'IterativeIntensityFragPrediction' + str(i)

            cos_sims = []
            js_sims = []
            regen_mz_accuracy = []
            for data in test_data:
                if os.path.isfile(
                        save_path + 'raw/json/' + task + '/' + data[
                            'identifier'] + '_' + task + '_model_response.json'):  # and os.path.isfile(
                    with open(
                            save_path + 'raw/json/' + task + '/' + data['identifier'] + '_' + task + '_model_response.json',
                            'r') as file:
                        int_response = json.load(file)

                    cos_sim, js_sim = baseline_metrics(int_response, data)
                    cos_sims.append(cos_sim)
                    js_sims.append(js_sim)

                    assert os.path.isfile(save_path + 'raw/json/' + generation_task + '/' + data[
                        'identifier'] + '_' + generation_task + '_model_response.json')
                    with open(
                            save_path + 'raw/json/' + generation_task + '/' + data[
                                'identifier'] + '_' + generation_task + '_model_response.json',
                            'r') as file:
                        frag_response = json.load(file)
                    regen_mz_accuracy.append(calculate_regen_mz(frag_response, int_response, data))


            print("Iterative result: Average overall cosine similarity for " + str(len(cos_sims)) + " points: " + str(
                statistics.mean(cos_sims)))
            filtered_js_sims = [x for x in js_sims if x is not None]
            print("Iterative result: Average overall Jensen-Shannon similarity for " + str(len(js_sims)) + " points: " + str(
                statistics.mean(filtered_js_sims)))
            print("Iterative result: Average overall regenerated m/z accuracy for " + str(len(regen_mz_accuracy)) + " points: " + str(
                statistics.mean(regen_mz_accuracy)))



