import os
import pickle
import csv
import json
import statistics
import tiktoken
import random
import sys
from openai import OpenAI
from collections import Counter
import argparse



from preprocess_data import load_test_data, load_train_data, load_val_data, add_method_features, load_cands_dict, order_by_complexity
from prompts import create_train_prompts, create_test_prompts, create_test_prompts_from_gen_frags
from finetuning import fine_tune_pipeline
from query import query_model, get_intensity_frag_prediction_schema, get_fragment_list_prediction_schema
from iterate_fragmentation import iterate_fragmentation
from iterate_intensity import iterate_intensity
from metrics import intensity_frag_prediction_evaluation, fragment_list_prediction_evaluation, baseline_metrics, calculate_regen_mz



# parser
parser = argparse.ArgumentParser()
parser.add_argument("--api_key", default=None)
parser.add_argument("--model_type", default="gpt-4o-mini-2024-07-18")
# "gpt-4o-2024-08-06","gpt-5"
parser.add_argument("--data_path", default='./data/')
parser.add_argument("--save_path", default='./output/')
parser.add_argument("--temperature", default=0.9)
parser.add_argument("--disable_training", action="store_true", default=False)
parser.add_argument("--disable_inference", action="store_true", default=False)
parser.add_argument("--disable_evaluation", action="store_true", default=False)
parser.add_argument("--disable_iterate", action="store_true", default=False)
parser.add_argument("--intensity_model", default="base_model")
parser.add_argument("--test_data_filename", default=None)
parser.add_argument("--test_subset", action="store_true", default=False)
parser.add_argument("--test_subset_size", default=300)
parser.add_argument("--training_data_size", default=1000)

args = vars(parser.parse_args())

api_key = args["api_key"]
if api_key == None:
    sys.exit("Exiting because no api key was provided for OpenAI")
model_name = args["model_type"]
data_path = args["data_path"]
save_path = args["save_path"]
temp = float(args["temperature"])
training = not args["disable_training"]
inference = not args["disable_inference"]
evaluation = not args["disable_evaluation"]
iterate = not args["disable_iterate"]
intensity_model = args["intensity_model"]
test_data_filename = args["test_data_filename"]
test_subset = not args["test_subset"]
if test_subset:
    test_subset_size = float(args["test_subset_size"])
if not evaluation:
    assert test_data_filename != None
if training:
    encoding = tiktoken.encoding_for_model(model_name)
    training_pts = float(args["training_data_size"])
    suffix = "_"+str(training_pts)

client = OpenAI(api_key=api_key)
tasks = ["FragmentListPrediction"]

if not os.path.isdir(save_path):
    os.mkdir(save_path)
if not os.path.isdir(save_path+'raw/'):
    os.mkdir(save_path+'raw/')
if not os.path.isdir(save_path+'raw/json/'):
    os.mkdir(save_path+'raw/json/')
if not os.path.isdir(save_path+'raw/pkl/'):
    os.mkdir(save_path+'raw/pkl/')
for task in tasks:
    if not os.path.isdir(save_path + 'raw/json/'+task):
        os.mkdir(save_path + 'raw/json/'+task)
for task in tasks:
    if not os.path.isdir(save_path + 'raw/pkl/'+task):
        os.mkdir(save_path + 'raw/pkl/'+task)
if not os.path.isdir(save_path+'postprocessed/'):
    os.mkdir(save_path+'postprocessed/')
if not os.path.isdir(save_path+'figures/'):
    os.mkdir(save_path+'figures/')

if not os.path.isdir(data_path+'processed/'):
    os.mkdir(data_path+'processed/')
if not os.path.isdir(data_path+'processed/prompts/'):
    os.mkdir(data_path+'processed/prompts/')
if not os.path.isdir(data_path+'processed/prompts/prompts_with_metadata/'):
    os.mkdir(data_path+'processed/prompts/prompts_with_metadata/')


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


    # don't consider cases without labels
    temp = []
    for pt in train_data:
        if len(pt['sorted_subforms']) != 0:
            temp.append(pt)
    print(len(temp))
    print(len(train_data))
    train_data = temp
    temp = []
    for pt in val_data:
        if len(pt['sorted_subforms']) != 0:
            temp.append(pt)
    print(len(temp))
    print(len(val_data))
    val_data = temp




    if not os.path.exists(data_path + 'processed/prompts/train_queries_' + 'IntensityFragPrediction_subset' + suffix + '.jsonl'):

        l_mol = []
        l_adduct = []
        l_instrument = []

        # make prompts
        train_queries_dict = {}
        train_tasks = ['FragmentListPrediction', 'IntensityFragPrediction']
        for task in train_tasks:
            train_queries = create_train_prompts(task, train_data)
            train_queries_dict[task] = train_queries

        final_prompts = {"MaskedFragment": [], "FragmentListPrediction": [], "MaskedIntensity" : [], "IntensityFragPrediction": []}


        count_tokens_full1 = 0
        l_adduct_full = []
        l_instrument_full = []
        for i in range(0,len(train_queries_dict["FragmentListPrediction"])):
            if len(final_prompts["FragmentListPrediction"]) == training_pts*0.9: #1800:# 450: #900:#4500:#9000:
                break
            num_frags = len(train_queries_dict["FragmentListPrediction"][i]['frags'])
            num_tokens = len(encoding.encode(train_queries_dict["FragmentListPrediction"][i]['messages'][1]['content']))
            num_tokens_int = len(encoding.encode(train_queries_dict["IntensityFragPrediction"][i]['messages'][1]['content']))
            spectra_data = [d for d in train_data if d.get('identifier') == train_queries_dict["FragmentListPrediction"][i]['identifier']]
            assert len(spectra_data) == 1
            spectra_data = spectra_data[0]
            if spectra_data['inchikey'] not in l_mol:
                final_prompts["FragmentListPrediction"].append(train_queries_dict["FragmentListPrediction"][i])
                l_mol.append(spectra_data['inchikey'])
                l_adduct_full.append(spectra_data['adduct'])
                l_instrument_full.append(spectra_data['instrument_type'])
                count_tokens_full1 += num_tokens
                count_tokens_full1 += len(encoding.encode(train_queries_dict["FragmentListPrediction"][i]['messages'][0]['content']))
                count_tokens_full1 += len(encoding.encode(train_queries_dict["FragmentListPrediction"][i]['messages'][2]['content']))
                final_prompts["IntensityFragPrediction"].append(train_queries_dict["IntensityFragPrediction"][i])
                count_tokens_full1 += len(encoding.encode(train_queries_dict["IntensityFragPrediction"][i]['messages'][1]['content']))
                count_tokens_full1 += len(encoding.encode(train_queries_dict["IntensityFragPrediction"][i]['messages'][0]['content']))
                count_tokens_full1 += len(encoding.encode(train_queries_dict["IntensityFragPrediction"][i]['messages'][2]['content']))

        # print('Training')
        # print(len(l_mol))
        # print(dict(Counter(l_adduct_full)))
        # print(dict(Counter(l_instrument_full)))

        # save
        ordered_queries = order_by_complexity(final_prompts["FragmentListPrediction"], train_data, encoding)
        with open(data_path + 'processed/prompts/prompts_with_metadata/train_queries_' + 'FragmentListPrediction_subset' + suffix + '.jsonl', "w") as f:
            for pt in ordered_queries:
                f.write(json.dumps(pt) + "\n")
        ordered_queries = order_by_complexity(final_prompts["IntensityFragPrediction"], train_data, encoding)
        with open(data_path + 'processed/prompts/prompts_with_metadata/train_queries_' + 'IntensityFragPrediction_subset' + suffix + '.jsonl', "w") as f:
            for pt in ordered_queries:
                f.write(json.dumps(pt) + "\n")

        ############################################
        # validation files
        # make prompts
        val_queries_dict = {}
        for task in train_tasks:
            val_queries = create_train_prompts(task, val_data)
            val_queries_dict[task] = val_queries
        final_val_prompts = {"MaskedFragment": [], "FragmentListPrediction": [], "MaskedIntensity" : [], "IntensityFragPrediction": []}

        count_tokens_val = 0
        l_adduct_val = []
        l_instrument_val = []
        for i in range(0, len(val_queries_dict["FragmentListPrediction"])):
            if len(final_val_prompts["FragmentListPrediction"]) == training_pts*0.1: #200: #50: #100:#:500:#1000:#556:#12:
                break
            num_frags = len(val_queries_dict["FragmentListPrediction"][i]['frags'])
            num_tokens = len(encoding.encode(val_queries_dict["FragmentListPrediction"][i]['messages'][1]['content']))
            num_tokens_int = len(encoding.encode(val_queries_dict["IntensityFragPrediction"][i]['messages'][1]['content']))
            spectra_data = [d for d in val_data if
                            d.get('identifier') == val_queries_dict["FragmentListPrediction"][i]['identifier']]
            assert len(spectra_data) == 1
            spectra_data = spectra_data[0]
            if spectra_data['inchikey'] not in l_mol:
                final_val_prompts["FragmentListPrediction"].append(val_queries_dict["FragmentListPrediction"][i])
                l_mol.append(spectra_data['inchikey'])
                l_adduct_val.append(spectra_data['adduct'])
                l_instrument_val.append(spectra_data['instrument_type'])
                count_tokens_val += num_tokens
                count_tokens_val += len(
                    encoding.encode(val_queries_dict["FragmentListPrediction"][i]['messages'][0]['content']))
                count_tokens_val += len(
                    encoding.encode(val_queries_dict["FragmentListPrediction"][i]['messages'][2]['content']))
                final_val_prompts["IntensityFragPrediction"].append(val_queries_dict["IntensityFragPrediction"][i])
                count_tokens_val += len(
                    encoding.encode(val_queries_dict["IntensityFragPrediction"][i]['messages'][1]['content']))
                count_tokens_val += len(
                    encoding.encode(val_queries_dict["IntensityFragPrediction"][i]['messages'][0]['content']))
                count_tokens_val += len(
                    encoding.encode(val_queries_dict["IntensityFragPrediction"][i]['messages'][2]['content']))

        print('Validation')
        print(len(l_mol))
        print(dict(Counter(l_adduct_val)))
        print(dict(Counter(l_instrument_val)))

        # save
        ordered_queries = order_by_complexity(final_val_prompts["FragmentListPrediction"], val_data, encoding)
        with open(data_path + 'processed/prompts/prompts_with_metadata/val_queries_' + 'FragmentListPrediction_subset' + suffix + '.jsonl', "w") as f:
            for pt in ordered_queries:
                f.write(json.dumps(pt) + "\n")
        ordered_queries = order_by_complexity(final_val_prompts["IntensityFragPrediction"], val_data, encoding)
        with open(data_path + 'processed/prompts/prompts_with_metadata/val_queries_' + 'IntensityFragPrediction_subset' + suffix + '.jsonl', "w") as f:
            for pt in ordered_queries:
                f.write(json.dumps(pt) + "\n")
        # ############################################
        # remove metadata
        with open(
                data_path + 'processed/prompts/prompts_with_metadata/train_queries_' + 'FragmentListPrediction_subset' + suffix + '.jsonl',
                "r") as infile, open(
                data_path + 'processed/prompts/train_queries_' + 'FragmentListPrediction_subset' + suffix + '.jsonl',
                "w") as outfile:
            for line in infile:
                data = json.loads(line)
                # Keep only the 'messages' field
                if "messages" in data:
                    json.dump({"messages": data["messages"]}, outfile)
                    outfile.write("\n")

        with open(
                data_path + 'processed/prompts/prompts_with_metadata/train_queries_' + 'IntensityFragPrediction_subset' + suffix + '.jsonl',
                "r") as infile, open(
                data_path + 'processed/prompts/train_queries_' + 'IntensityFragPrediction_subset' + suffix + '.jsonl',
                "w") as outfile:
            for line in infile:
                data = json.loads(line)
                # Keep only the 'messages' field
                if "messages" in data:
                    json.dump({"messages": data["messages"]}, outfile)
                    outfile.write("\n")


        with open(data_path + 'processed/prompts/prompts_with_metadata/val_queries_' + 'FragmentListPrediction_subset' + suffix + '.jsonl', "r") as infile, open(data_path + 'processed/prompts/val_queries_' + 'FragmentListPrediction_subset' + suffix + '.jsonl', "w") as outfile:
            for line in infile:
                data = json.loads(line)
                # Keep only the 'messages' field
                if "messages" in data:
                    json.dump({"messages": data["messages"]}, outfile)
                    outfile.write("\n")

        with open(data_path + 'processed/prompts/prompts_with_metadata/val_queries_' + 'IntensityFragPrediction_subset' + suffix + '.jsonl', "r") as infile, open(data_path + 'processed/prompts/val_queries_' + 'IntensityFragPrediction_subset' + suffix + '.jsonl', "w") as outfile:
            for line in infile:
                data = json.loads(line)
                # Keep only the 'messages' field
                if "messages" in data:
                    json.dump({"messages": data["messages"]}, outfile)
                    outfile.write("\n")


    task_files = []
    for task in ['IntensityFragPrediction']:
        task_schemas = {
            "IntensityFragPrediction": get_intensity_frag_prediction_schema(),
            "FragmentListPrediction": get_fragment_list_prediction_schema(),
        }
        task_files.append({'file': data_path + 'processed/prompts/train_queries_' + task + '_subset' + suffix +'.jsonl',
                           'val_file': data_path + 'processed/prompts/val_queries_' + task + '_subset' + suffix + '.jsonl',
                           'suffix': task + suffix, 'task_schema': task_schemas[task], 'task_name': task})



    #fine-tune
    intensity_model = fine_tune_pipeline(task_files, model_name, api_key)

if evaluation:
    if os.path.exists(data_path + "processed/test_data_with_method_features.pkl"):
        with open(data_path + "processed/test_data_with_method_features.pkl", "rb") as f:
            test_data = pickle.load(f)
    else:
        test_data = load_test_data(data_path)
        test_data = add_method_features(test_data)
        with open(data_path + "processed/test_data_with_method_features.pkl", "wb") as f:
            pickle.dump(test_data, f)
else:
    #new data
    with open(data_path + test_data_filename, "rb") as f:
        test_data = pickle.load(f)
    # test_data = add_method_features(test_data)

test_queries_dict = {}
for task in tasks:
    if os.path.exists(data_path + 'processed/prompts/test_queries_' + task + '.jsonl'):
        with open(data_path + 'processed/prompts/test_queries_' + task + '.jsonl', 'r') as f:
            test_queries = [json.loads(line) for line in f]
    else:
        test_queries = create_test_prompts(task, test_data)
        with open(data_path + 'processed/prompts/test_queries_' + task + '.jsonl', "w") as f:
            for pt in test_queries:
                f.write(json.dumps(pt) + "\n")
    test_queries_dict[task] = test_queries


for task, test_queries in test_queries_dict.items():
    count = 0
    for data in test_queries:
        if not os.path.exists(save_path + 'raw/json/' + task + '/' + data['identifier'] + '_' + task + '_model_response.json'):
            response = query_model(client, model_name, task, data)
            if response == {}: #if nothing was returned
                response_dict = {}
            else:
                response_dict = response.model_dump()
            with open(save_path + 'raw/pkl/' + task + '/' + data['identifier'] + '_' + task + '_model_response.pkl', 'wb') as file:
                pickle.dump(response_dict, file)
            try:
                answer = json.loads(response.choices[0].message.content)
            except:
                try:
                    answer = json.loads(response.choices[0].message.content + '}')
                except:
                    try:
                        answer = json.loads(response.choices[0].message.content + '"}')
                    except:
                        try:
                            answer = json.loads(response.choices[0].message.content + '"}]}')
                        except:
                            answer = {}
            with open(save_path + 'raw/json/' + task + '/' + data['identifier'] + '_' + task + '_model_response.json', "w") as file:
                json.dump(answer, file, indent=2)
        count +=1
        if test_subset:
            if count >= test_subset_size:
                break


if iterate:
    assert 'FragmentListPrediction' in tasks
    n=5
    if not test_subset:
        iterate_fragmentation(test_data, save_path, data_path, client, model_name, n)
    generation_task = "IterativeFragmentListPrediction" +str(n)
else:
    generation_task = "FragmentListPrediction"


if inference:
    task = "IntensityFragPrediction"
    tasks.append(task)
    queries = []
    for data in test_data:
        if os.path.isfile(
                save_path + 'raw/json/' + generation_task + '/' + data['identifier'] + '_' + generation_task + '_model_response.json'):
            with open(save_path + 'raw/json/' + generation_task + '/' + data['identifier'] + '_' + generation_task + '_model_response.json',
                      'r') as file:
                response = json.load(file)

            single_query = create_test_prompts_from_gen_frags(response, data)
            queries.append(single_query)
            if test_subset:
                if len(queries) >= test_subset_size:
                    break

    if not os.path.isdir(save_path + 'inference_intensity_prompts/'):
        os.mkdir(save_path + 'inference_intensity_prompts/')
    with open(save_path + 'inference_intensity_prompts/test_queries_' + task + '.jsonl', "w") as f:
        for pt in queries:
            f.write(json.dumps(pt) + "\n")
    test_queries_dict[task] = queries

    #run queries
    if not os.path.isdir(save_path + 'raw/json/' + task):
        os.mkdir(save_path + 'raw/json/' + task)
    if not os.path.isdir(save_path + 'raw/pkl/' + task):
        os.mkdir(save_path + 'raw/pkl/' + task)
    for data in queries:
        if not os.path.exists(save_path + 'raw/json/' + task + '/' + data['identifier'] + '_' + task + '_model_response.json'):
            response = query_model(client, intensity_model, task, data)
            if response == {}: #if nothing was returned
                response_dict = {}
            else:
                response_dict = response.model_dump()
            with open(save_path + 'raw/pkl/' + task + '/' + data['identifier'] + '_' + task + '_model_response.pkl', 'wb') as file:
                pickle.dump(response_dict, file)
            try:
                answer = json.loads(response.choices[0].message.content)
            except:
                try:
                    answer = json.loads(response.choices[0].message.content + '}')
                except:
                    try:
                        answer = json.loads(response.choices[0].message.content + '"}')
                    except:
                        try:
                            answer = json.loads(response.choices[0].message.content + '"}]}')
                        except:
                            answer = {}
            with open(save_path + 'raw/json/' + task + '/' + data['identifier'] + '_' + task + '_model_response.json', "w") as file:
                json.dump(answer, file, indent=2)

    if iterate:
        iterate_intensity(generation_task, test_data, save_path, data_path, client, intensity_model, n=1)
        iterative_intensity_task = 'IterativeIntensityFragPrediction1'


metrics = {}
for task in tasks:
    if task == "MaskedSELFIES":
        metrics[task] = {"validity":[], "accuracy":[], "sim":[], "gen_unmasked_accuracy":[], "masked_token_accuracy":[]}
    elif task == "MaskedFragment":
        metrics[task] = {"sequence_level_accuracy":[], "overall_frag_accuracy":[], "gen_unmasked_accuracy":[], "masked_frag_accuracy":[], "masked_frag_structurally_equivalent":[], "masked_frag_sim":[], "validity":[], "true_substructure":[]}
    elif task == "FragmentListPrediction":
        metrics[task] = {"sequence_level_accuracy":[], "overall_frag_accuracy":[], "formula_accuracy":[], "mass_accuracy":[], "validity":[], "true_substructure":[]}
    elif task == "MaskedIntensity":
        metrics[task] = {}
    elif task == "IntensityFragPrediction":
        metrics[task] = {"sequence_level_accuracy":[], "gen_input_accuracy":[], "int_accuracy":[], "int_mse":[], "int_mae":[]}
    elif task == "IntensityPrediction":
        metrics[task] = {"sequence_level_accuracy":[], "gen_input_accuracy":[], "int_accuracy":[], "int_mse":[], "int_mae":[]}
    elif task == "SubformulaPrediction":
        metrics[task] = {"sequence_level_accuracy":[], "overall_sf_accuracy":[], "unordered_sf_accuracy":[], "unordered_sf_equivalence":[], "mass_accuracy":[], "actual_subformula":[], "validity":[]}

if evaluation:#run evaluation metrics with ground truth
    count_data = 0
    for data in test_data:
        for task in tasks:
            if os.path.isfile(save_path + 'raw/json/'+ task + '/' + data['identifier'] + '_' + task + '_model_response.json'):
                with open(save_path + 'raw/json/'+ task + '/' + data['identifier'] + '_' + task + '_model_response.json', 'r') as file:
                    response = json.load(file)
                if task == "FragmentListPrediction":
                    sequence_level_accuracy, overall_frag_accuracy, formula_accuracy, mass_accuracy, validity, true_substructure = fragment_list_prediction_evaluation(response, data)
                    metrics[task]["sequence_level_accuracy"].append(sequence_level_accuracy)
                    metrics[task]["overall_frag_accuracy"].append(overall_frag_accuracy)
                    metrics[task]["formula_accuracy"].append(formula_accuracy)
                    metrics[task]["mass_accuracy"].append(mass_accuracy)
                    metrics[task]["validity"].extend(validity)
                    metrics[task]["true_substructure"].extend(true_substructure)
                elif task == "IntensityFragPrediction":
                    sequence_level_accuracy, gen_input_accuracy, int_accuracy, int_mse, int_mae = intensity_frag_prediction_evaluation(response, data)
                    metrics[task]["sequence_level_accuracy"].append(sequence_level_accuracy)
                    metrics[task]["gen_input_accuracy"].append(gen_input_accuracy)
                    metrics[task]["int_accuracy"].extend(int_accuracy)
                    metrics[task]["int_mse"].extend([int_mse])
                    metrics[task]["int_mae"].extend([int_mae])

        if test_subset:
            count_data+=1
            if count_data>=test_subset_size:
                break

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


    if test_subset:
        for task in tasks:
            if task == "FragmentListPrediction":
                print("FragmentListPrediction results over " + str(test_subset_size) + " data points:")
                print("Sequence level accuracy: " + str(
                    statistics.mean(metrics[task]["sequence_level_accuracy"][:test_subset_size])))
                print("Overall fragment accuracy: " + str(
                    statistics.mean(metrics[task]["overall_frag_accuracy"][:test_subset_size])))
                print("Formula accuracy: " + str(
                    statistics.mean(metrics[task]["formula_accuracy"][:test_subset_size])))
                print("Mass accuracy: " + str(
                    statistics.mean(metrics[task]["mass_accuracy"][:test_subset_size])))
                print("Validity: " + str(statistics.mean(metrics[task]["validity"][:test_subset_size])))
                print("Proportion true substructure: " + str(
                    statistics.mean(metrics[task]["true_substructure"][:test_subset_size])))
            elif task == "IntensityFragPrediction":
                print("IntensityFragPrediction results over " + str(test_subset_size) + " data points:")
                print("Sequence level accuracy: " + str(
                    statistics.mean(metrics[task]["sequence_level_accuracy"][:test_subset_size])))
                print("Regenerated input sequence accuracy: " + str(
                    statistics.mean(metrics[task]["gen_input_accuracy"][:test_subset_size])))
                print("Intensity accuracy: " + str(statistics.mean(metrics[task]["int_accuracy"][:test_subset_size])))
                print("Intensity MSE: " + str(statistics.mean(metrics[task]["int_mse"][:test_subset_size])))
                print("Intensity MAE: " + str(statistics.mean(metrics[task]["int_mae"][:test_subset_size])))

if iterate and evaluation:
    n=5
    for i in range(1,n+1):
        task = 'IterativeFragmentListPrediction' + str(i)
        metrics[task] = {"sequence_level_accuracy": [], "overall_frag_accuracy": [], "formula_accuracy": [],
                         "mass_accuracy": [], "validity": [], "true_substructure": []}
        metrics["Cumulative" + task] = {"sequence_level_accuracy": [], "overall_frag_accuracy": [], "formula_accuracy": [],
                         "mass_accuracy": [], "validity": [], "true_substructure": []}
        count_data = 0
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

                if test_subset:
                    count_data += 1
                    if count_data >= test_subset_size:
                        break


        print(task + " results over " + str(len(metrics[task]["sequence_level_accuracy"])) + " data points:")
        print("Sequence level accuracy: " + str(statistics.mean(metrics[task]["sequence_level_accuracy"])))
        print("Overall fragment accuracy: " + str(statistics.mean(metrics[task]["overall_frag_accuracy"])))
        print("Formula accuracy: " + str(
            statistics.mean(metrics[task]["formula_accuracy"])))
        print("Mass accuracy: " + str(
            statistics.mean(metrics[task]["mass_accuracy"])))
        print("Validity: " + str(statistics.mean(metrics[task]["validity"])))
        print("Proportion true substructure: " + str(statistics.mean(metrics[task]["true_substructure"])))


        if test_subset:
            print(task + " results over " + str(test_subset_size) + " data points:")
            print("Sequence level accuracy: " + str(statistics.mean(metrics[task]["sequence_level_accuracy"][:test_subset_size])))
            print("Overall fragment accuracy: " + str(statistics.mean(metrics[task]["overall_frag_accuracy"][:test_subset_size])))
            print("Formula accuracy: " + str(
                statistics.mean(metrics[task]["formula_accuracy"][:test_subset_size])))
            print("Mass accuracy: " + str(
                statistics.mean(metrics[task]["mass_accuracy"][:test_subset_size])))
            print("Validity: " + str(statistics.mean(metrics[task]["validity"][:test_subset_size])))
            print("Proportion true substructure: " + str(statistics.mean(metrics[task]["true_substructure"][:test_subset_size])))


if inference:
    task = "IntensityFragPrediction"

    intensity_prompts = {}
    for prompt in test_queries_dict[task]:
        intensity_prompts[prompt['identifier']] = prompt


    cos_sims = []
    js_sims = []
    regen_mz_accuracy = []
    count_data = 0
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


            if test_subset:
                count_data += 1
                if count_data >= test_subset_size:
                    break

    print("Average overall cosine similarity for " + str(len(cos_sims)) + " points: " + str(statistics.mean(cos_sims)))
    filtered_js_sims = [x for x in js_sims if x is not None]  # NEW
    print("Average overall Jensen-Shannon similarity for " + str(len(js_sims)) + " points: " + str(statistics.mean(filtered_js_sims)))
    print("Average overall regenerated m/z accuracy for " + str(len(regen_mz_accuracy)) + " points: " + str(statistics.mean(regen_mz_accuracy)))


    if test_subset:
        print("Average overall cosine similarity for " + str(test_subset_size) + " points: " + str(
            statistics.mean(cos_sims[:test_subset_size])))
        filtered_js_sims = [x for x in js_sims[:test_subset_size] if x is not None] #NEW
        print("Average overall Jensen-Shannon similarity for " + str(test_subset_size) + " points: " + str(
            statistics.mean(filtered_js_sims)))
        print("Average overall regenerated m/z accuracy for " + str(test_subset_size) + " points: " + str(
            statistics.mean(regen_mz_accuracy[:test_subset_size])))

    if iterate:
        n=1
        for i in range(1, n + 1):
            task = 'IterativeIntensityFragPrediction' + str(i)

            cos_sims = []
            js_sims = []
            regen_mz_accuracy = []
            count_data = 0
            for data in test_data:
                if os.path.isfile(
                        save_path + 'raw/json/' + task + '/' + data[
                            'identifier'] + '_' + task + '_model_response.json'):
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


                    if test_subset:
                        count_data += 1
                        if count_data >= test_subset_size:
                            break

            print("Iterative result: Average overall cosine similarity for " + str(len(cos_sims)) + " points: " + str(
                statistics.mean(cos_sims)))
            filtered_js_sims = [x for x in js_sims if x is not None]  # NEW
            print("Iterative result: Average overall Jensen-Shannon similarity for " + str(len(js_sims)) + " points: " + str(
                statistics.mean(filtered_js_sims)))
            print("Iterative result: Average overall regenerated m/z accuracy for " + str(len(regen_mz_accuracy)) + " points: " + str(
                statistics.mean(regen_mz_accuracy)))


            if test_subset:
                print(
                    "Iterative result: Average overall cosine similarity for " + str(test_subset_size) + " points: " + str(
                        statistics.mean(cos_sims[:test_subset_size])))
                filtered_js_sims = [x for x in js_sims[:test_subset_size] if x is not None]  # NEW
                print("Iterative result: Average overall Jensen-Shannon similarity for " + str(
                    test_subset_size) + " points: " + str(
                    statistics.mean(filtered_js_sims)))
                print("Iterative result: Average overall regenerated m/z accuracy for " + str(
                    test_subset_size) + " points: " + str(
                    statistics.mean(regen_mz_accuracy[:test_subset_size])))

