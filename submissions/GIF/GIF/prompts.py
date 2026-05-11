import copy
import json
from rdkit import Chem
from rdkit.Chem import Descriptors

from metrics import is_substructure, selfies_to_smiles, molar_mass, calculate_mz, is_substructure_smi


def get_prompt_templates():
    prompts = {
        "FragmentListPrediction":
            {"system":"You are a chemistry model specialized in mass spectrometry. In this task, predict the most likely mass spectrometry fragments based on molecular structure and fragmentation patterns.",
             "user":"<<TASK=FragmentListPrediction>> Predict all major fragments (ordered by descending intensity) in SELFIES format:\n<<MOL>> [[SELFIES]]\n<<EXP_SETTINGS>> [[EXP_SETTINGS]]",
             "assistant": "[[FRAGMENTS]]"
            },
        "IntensityFragPrediction":
            {"system":"You are a chemistry model specialized in mass spectrometry. In this task, estimate the intensity of fragments based on their mass, structure, and likely ionization behavior.",
             "user": "<<TASK=IntensityFragPrediction>> Predict the intensity scores (1–10) for all fragments provided based on the molecular structure and experiment settings. Return the predicted intensity and the exact corresponding m/z values for each fragment listed here:\n<<MOL>> [[SELFIES]]\n<<EXP_SETTINGS>> [[EXP_SETTINGS]]\n[[FRAG_LIST_MZ]]",
             "assistant": "[[LIST_MZ_INT]]"
             },
    }
    return prompts

def create_train_prompts(task, data):
    prompts = get_prompt_templates()
    queries = []
    if "FragmentListPrediction" == task:
        for pt in data:
            temp = copy.deepcopy(prompts[task])
            user_prompt = temp["user"].replace("[[SELFIES]]",pt["selfies"]).replace("[[EXP_SETTINGS]]",json.dumps(pt["exp_settings"], indent=None).replace('"',''))
            assistant_prompt = temp["assistant"].replace("[[FRAGMENTS]]",json.dumps(pt["frags_str"], indent=None).replace('"',''))
            data_point = {"messages": [{"role": "system", "content": temp["system"]},
                                       {"role": "user", "content": user_prompt},
                                       {"role": "assistant", "content": assistant_prompt}],
                          "identifier": pt['identifier'],
                          "frags": pt['sorted_frags']}
            queries.append(data_point)
    if "IntensityFragPrediction" == task:
        for pt in data:
            temp = copy.deepcopy(prompts[task])
            user_prompt = temp["user"].replace("[[SELFIES]]",pt["selfies"]).replace("[[EXP_SETTINGS]]",json.dumps(pt["exp_settings"], indent=None).replace('"','')).replace("[[FRAG_LIST_MZ]]", json.dumps(pt["frags_mz"], indent=None).replace('"',''))
            assistant_prompt = temp["assistant"].replace("[[LIST_MZ_INT]]",json.dumps(pt["str_mz_int"], indent=None).replace('"',''))
            data_point = {"messages": [{"role": "system", "content": temp["system"]},
                                       {"role": "user", "content": user_prompt},
                                       {"role": "assistant", "content": assistant_prompt}],
                          "identifier": pt['identifier'],
                          "frags": pt['sorted_frags']}
            queries.append(data_point)

    return queries

def get_generated_fragments(response,data):
    gen_frags = []
    if 'fragments' in response.keys():
        for frag in response['fragments']:
            gen_frags.append(frag['frag'])
    true_substructure = {}
    for frag in gen_frags:
        if is_substructure(frag, data['smiles']) and frag not in true_substructure.keys():
            m = Chem.MolFromSmiles(selfies_to_smiles(frag))
            mass = Descriptors.ExactMolWt(m)
            mz = calculate_mz(mass, data['adduct'])

            true_substructure[frag] = mz

    frags_mz_str = []
    for frag in true_substructure.keys():
        frags_mz_str.append({"frag": frag, "mz": round(true_substructure[frag], 2)})
    return frags_mz_str

def get_generated_fragments_smi(response,data):
    gen_frags = []
    if 'fragments' in response.keys():
        for frag in response['fragments']:
            gen_frags.append(frag['frag'])
    true_substructure = {}
    for frag in gen_frags:
        if is_substructure_smi(frag, data['smiles']) and frag not in true_substructure.keys():
            m = Chem.MolFromSmiles(frag)
            mass = Descriptors.ExactMolWt(m)
            mz = calculate_mz(mass, data['adduct'])
            true_substructure[frag] = mz

    frags_mz_str = []
    for frag in true_substructure.keys():
        frags_mz_str.append({"frag": frag, "mz": round(true_substructure[frag], 2)})
    return frags_mz_str

def create_test_prompts_from_gen_frags(response, data):
    task = "IntensityFragPrediction"
    prompts = get_prompt_templates()
    frags_mz_str = get_generated_fragments(response,data)

    temp = copy.deepcopy(prompts[task])
    user_prompt = temp["user"].replace("[[SELFIES]]", data["selfies"]).replace("[[EXP_SETTINGS]]",
                                                                             json.dumps(data["exp_settings"],
                                                                                        indent=None).replace('"',
                                                                                                             '')).replace(
        "[[FRAG_LIST_MZ]]", json.dumps(frags_mz_str, indent=None).replace('"', ''))
    assistant_prompt = temp["assistant"].replace("[[LIST_MZ_INT]]",json.dumps(data["str_mz_int"], indent=None).replace('"',''))
    data_point = {"messages": [{"role": "system", "content": temp["system"]},
                               {"role": "user", "content": user_prompt}],
                  "expected_output": assistant_prompt,
                  "identifier": data['identifier'],
                  "frags": data['sorted_frags']}

    return data_point

def create_test_prompts_few_shot(task, data, few_shot_queries):
    assert task == 'FragmentListPrediction'
    l_few_shots = []
    for pt in few_shot_queries:
        if pt['messages'][2]['content'].count(',') >= 3 and pt['messages'][2]['content'].count(',') <= 6:
            l_few_shots.append(pt)
            if len(l_few_shots) >= 3:
                break
    str_few_shot = ""
    for pt in l_few_shots:
        str_few_shot = str_few_shot+"Question: "+ pt['messages'][1]['content'].split('\n')[1] + '\n' + pt['messages'][1]['content'].split('\n')[2] + '\nAnswer: ' + l_few_shots[0]['messages'][2]['content'] + '\n'

    prompts = get_prompt_templates()
    queries = []
    for pt in data:
        temp = copy.deepcopy(prompts[task])
        assert temp["user"][0:31] == '<<TASK=FragmentListPrediction>>'
        user_prompt = temp["user"][0:31] + " Predict all major fragments (ordered by descending intensity) in SELFIES format based on molecular structure and experiment settings. When answering, follow these examples:\n" +str_few_shot + '\n' + temp["user"][32:]
        user_prompt = user_prompt.replace("[[SELFIES]]", pt["selfies"]).replace("[[EXP_SETTINGS]]",
                                                                                 json.dumps(pt["exp_settings"],
                                                                                            indent=None).replace('"',
                                                                                                                 ''))

        assistant_prompt = temp["assistant"].replace("[[FRAGMENTS]]",
                                                     json.dumps(pt["frags_str"], indent=None).replace('"', ''))
        data_point = {"messages": [{"role": "system", "content": temp["system"]},
                                   {"role": "user", "content": user_prompt}],
                      "expected_output": assistant_prompt,
                      "identifier": pt['identifier'],
                      "frags": pt['sorted_frags']}
        queries.append(data_point)

    return queries

def create_test_prompts(task, data):
    prompts = get_prompt_templates()
    queries = []
    if "FragmentListPrediction" == task:
        for pt in data:
            temp = copy.deepcopy(prompts[task])
            user_prompt = temp["user"].replace("[[SELFIES]]",pt["selfies"]).replace("[[EXP_SETTINGS]]",json.dumps(pt["exp_settings"], indent=None).replace('"',''))
            assistant_prompt = temp["assistant"].replace("[[FRAGMENTS]]",json.dumps(pt["frags_str"], indent=None).replace('"',''))
            data_point = {"messages": [{"role": "system", "content": temp["system"]},
                                       {"role": "user", "content": user_prompt}],
                          "expected_output": assistant_prompt,
                          "identifier": pt['identifier'],
                          "frags": pt['sorted_frags']}
            queries.append(data_point)
    if "IntensityFragPrediction" == task:
        for pt in data:
            temp = copy.deepcopy(prompts[task])
            user_prompt = temp["user"].replace("[[SELFIES]]",pt["selfies"]).replace("[[EXP_SETTINGS]]",json.dumps(pt["exp_settings"], indent=None).replace('"','')).replace("[[FRAG_LIST_MZ]]", json.dumps(pt["frags_mz"], indent=None).replace('"',''))
            assistant_prompt = temp["assistant"].replace("[[LIST_MZ_INT]]",json.dumps(pt["str_mz_int"], indent=None).replace('"',''))
            data_point = {"messages": [{"role": "system", "content": temp["system"]},
                                       {"role": "user", "content": user_prompt}],
                          "expected_output": assistant_prompt,
                          "identifier": pt['identifier'],
                          "frags": pt['sorted_frags']}
            queries.append(data_point)
    return queries

