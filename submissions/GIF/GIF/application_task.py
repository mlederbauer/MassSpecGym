import copy
import sys
import os
import json
import pickle
import selfies
from rdkit import Chem
from openai import OpenAI
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--api_key", default=None)

args = vars(parser.parse_args())

api_key = args["api_key"]
if api_key == None:
    sys.exit("Exiting because no api key was provided for OpenAI")

client = OpenAI(api_key=api_key)
model_name = "gpt-4o-2024-08-06"

mol1_results_path = "./output/inference_gpt4o_07192025/raw/json/"
data_path = './data/'

mol2_results_path = "./application_task/GIF_output/raw/json/"



def query(client, model_name, test_data, temperature = 0.1):
    response = client.chat.completions.create(
        model=model_name,
        messages=test_data['messages'],
        temperature=temperature
    )


    return response

###########################
#molecule1
with open(data_path + "processed/test_data_with_method_features.pkl", "rb") as f:
    test_data = pickle.load(f)
data = test_data[10]
assert data['identifier'] == "MassSpecGymID0088536"

with open(mol1_results_path + 'IterativeFragmentListPrediction5/' + data['identifier'] + '_IterativeFragmentListPrediction5_model_response.json', 'r') as f:
    response = json.load(f)
gen_frags1 = []
if 'fragments' in response.keys():
    for frag in response['fragments']:
        gen_frags1.append(frag['frag'])
with open(mol1_results_path + 'IterativeIntensityFragPrediction1/' + data['identifier'] + '_IterativeIntensityFragPrediction1_model_response.json', 'r') as f:
    response = json.load(f)
if 'intensities' in response:
    list_gen_spec = response['intensities']
else:
    list_gen_spec = []

mol1_gen_spec = []
for peak in list_gen_spec:
    if 'mz' in peak and 'int' in peak:
        mol1_gen_spec.append({"mz": float(peak['mz']), "int": float(peak['int']) / 10})

###########################
#spectrum representation
spectra_str = []
for j in range(0,len(data['mzs'])):
    spectra_str.append({"mz": data['mzs'][j], "int": data['intensities'][j]})


###########################
#molecule2
candidate = '[C][O][C][=Branch1][C][=O][C][=C][C][=C][Branch2][Ring2][#Branch1][N][C][=Branch1][C][=O][C][C][Branch2][Ring1][Branch2][N][Branch1][N][C][=Branch1][C][=O][C][=C][C][=C][O][Ring1][Branch1][C][C][C][C][C][C][C][Ring1][#Branch1][C][Ring2][Ring1][Branch1][=O][C][=C][Ring2][Ring1][N]'

with open(mol2_results_path + 'IterativeFragmentListPrediction5/' + data['identifier'] + '_cand1_IterativeFragmentListPrediction5_model_response.json', 'r') as f:
    response = json.load(f)
gen_frags2 = []
if 'fragments' in response.keys():
    for frag in response['fragments']:
        gen_frags2.append(frag['frag'])
with open(mol2_results_path + 'IterativeIntensityFragPrediction1/' + data['identifier'] + '_cand1_IterativeIntensityFragPrediction1_model_response.json', 'r') as f:
    response = json.load(f)
if 'intensities' in response:
    list_gen_spec = response['intensities']
else:
    list_gen_spec = []

mol2_gen_spec = []
for peak in list_gen_spec:
    if 'mz' in peak and 'int' in peak:
        mol2_gen_spec.append({"mz": float(peak['mz']), "int": float(peak['int']) / 10})




#question
temp = {
    "system": "You are a chemistry model specialized in mass spectrometry. In this task, provide scientific reasoning to the question, considering your previous prediction of fragments, based on molecular structure and fragmentation patterns.",
    "user": "When provided with the following molecular structure in SELFIES format and experiment settings:"
            "\n<<MOL1>> [[SELFIES1]]\n<<EXP_SETTINGS>> [[EXP_SETTINGS]],\nYou predicted all of the following fragments (ordered by descending "
            "intensity) and simulated the following spectra: <<FRAGMENTS>> [[PREVIOUS_FRAGMENTS1]]\n <<SPECTRUM>> [[SIMULATED_SPEC1]]\n"
            "And when provided with the following differing molecular structure in SELFIES format and the same experiment settings:"
            "\n<<MOL2>> [[SELFIES2]]\n<<EXP_SETTINGS>> [[EXP_SETTINGS]],\nYou predicted all of the following fragments (ordered by descending "
            "intensity) and simulated the following spectra: <<FRAGMENTS>> [[PREVIOUS_FRAGMENTS2]]\n <<SPECTRUM>> [[SIMULATED_SPEC2]]\n"
            "Based on all of this information, which molecule is more likely to represent the following experimental spectrum under the specified experiment settings and why?\n"
            "spectrum: <<EXP_SPECTRUM>> [[SPECTRUM]]"

}


temp["user"] = (temp["user"].replace("[[SELFIES1]]", data["selfies"])
                .replace("[[SELFIES2]]", candidate).replace("[[SIMULATED_SPEC1]]", json.dumps(mol1_gen_spec,indent=None).replace('"', ''))
                .replace("[[SIMULATED_SPEC2]]", json.dumps(mol2_gen_spec,indent=None).replace('"', ''))
               .replace("[[EXP_SETTINGS]]", json.dumps(data["exp_settings"],indent=None).replace('"', ''))
                .replace("[[PREVIOUS_FRAGMENTS1]]", json.dumps(gen_frags1, indent=None).replace('"', ''))
                .replace("[[PREVIOUS_FRAGMENTS2]]", json.dumps(gen_frags2, indent=None).replace('"', ''))
                .replace("[[SPECTRUM]]", json.dumps(spectra_str, indent=None).replace('"', ''))
               )

message = {"messages": [{"role": "system", "content": temp["system"]},
                                               {"role": "user", "content": temp["user"]}]}

response = query(client, model_name, message, temperature = 0.1)

with open("./application_task/example_question.pkl", 'wb') as file:
    pickle.dump(response, file)
with open("./application_task/example_question.txt","w") as f:
    f.writelines(response.choices[0].message.content)
with open('./application_task/example_prompt.jsonl',"w") as f:
    for pt in message["messages"]:
        f.write(json.dumps(pt) + "\n")









