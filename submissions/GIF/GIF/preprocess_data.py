import os
import pickle
import json
from datasets import load_dataset
import selfies
import tiktoken
import csv
import math



import random
seed_value = 0
random.seed(seed_value)

def load_data(data_path):
    data = load_dataset("roman-bushuiev/MassSpecGym")
    return data

def filter_unannotated_peaks_by_subformula(spec_data, data_path):
    filtered_data = []
    count_none = 0
    count_changed = 0
    for spec in spec_data:
        # print(spec)
        #load peak annotation...
        with open(data_path + 'subformulae_default/' + spec['identifier'] + '.json') as f:
            peak_annotations = json.load(f)
        peaks = peak_annotations['output_tbl']
        unfiltered_mzs = spec['mzs'].split(',')
        unfiltered_ints = spec['intensities'].split(',')
        unfiltered_mzs=[float(i) for i in unfiltered_mzs]
        unfiltered_ints=[float(i) for i in unfiltered_ints]
        new_mzs = []
        new_ints = []
        if peaks != None:
            for i in range(0,len(unfiltered_mzs)):
                if unfiltered_mzs[i] in peaks['mz']:
                    new_mzs.append(unfiltered_mzs[i])
                    new_ints.append(unfiltered_ints[i])
        spec['mzs'] = new_mzs
        spec['intensities'] = new_ints
        filtered_data.append(spec)
        if len(spec['mzs']) == 0:
            count_none += 1
        if len(spec['mzs']) != len(unfiltered_mzs):
            count_changed += 1
    print(str(count_none) + ' spectra without labels')
    print(str(count_changed) + ' spectra are filtered')

    return filtered_data

def filter_unannotated_peaks(spec_data, data_path):
    filtered_data = []
    count_none = 0
    count_changed = 0
    for spec in spec_data:
        # print(spec)
        #load peak annotation...
        with open(data_path + 'subformulae_default/' + spec['identifier'] + '.json') as f:
            peak_annotations = json.load(f)
        peaks = peak_annotations['output_tbl']
        unfiltered_mzs = spec['mzs'].split(',')
        unfiltered_ints = spec['intensities'].split(',')
        unfiltered_mzs=[float(i) for i in unfiltered_mzs]
        unfiltered_ints=[float(i) for i in unfiltered_ints]
        new_mzs = []
        new_ints = []
        if peaks != None:
            for i in range(0,len(unfiltered_mzs)):
                if unfiltered_mzs[i] in peaks['mz']:
                    new_mzs.append(unfiltered_mzs[i])
                    new_ints.append(unfiltered_ints[i])
        spec['mzs'] = new_mzs
        spec['intensities'] = new_ints
        filtered_data.append(spec)
        if len(spec['mzs']) == 0:
            count_none += 1
        if len(spec['mzs']) != len(unfiltered_mzs):
            count_changed += 1
    print(str(count_none) + ' spectra without labels')
    print(str(count_changed) + ' spectra are filtered')
    return filtered_data

def add_selfies_to_spec_data(spec_data):
    data_with_selfies = []

    for spec in spec_data:
        smi = spec['smiles']
        # mol = Chem.MolFromSmiles(smi)
        target_selfies = selfies.encoder(smi)
        spec['selfies'] = target_selfies
        data_with_selfies.append(spec)
    return data_with_selfies


def mask_selfies(selfies_str, masking_ratio=0.1, mask_token='[MASK]'):
    tokens = [f"[{t}]" for t in selfies_str[1:-1].split("][")]

    n_to_mask = max(1, int(len(tokens) * masking_ratio))
    mask_indices = random.sample(range(len(tokens)), n_to_mask)

    masked_tokens = [
        mask_token if i in mask_indices else token
        for i, token in enumerate(tokens)
    ]

    masked_selfies = ''.join(masked_tokens)
    return masked_selfies

def load_fragments(curr_id, data_path='./data/'):
    if os.path.exists(data_path + 'peak_annotations/' + curr_id + '.magma'):
        with open(data_path + 'peak_annotations/' + curr_id + '.magma', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            frag_int = {}
            frag_mz = {}
            header_row = 0
            for row in reader:
                if header_row == 0:
                    header_row += 1
                    if row[-1] != 'smiles':
                        break
                if row[-1] != 'smiles' and row[-1] != '':
                    # peak_annotations.append(row[-1].split("'")[1])
                    try:
                        assert row[-1][:2] == "['"
                        temp_frag = row[-1][2:-2]
                        temp_frag = temp_frag.split("', '")
                        for frag in temp_frag:
                            try:
                                selfies_frag = selfies.encoder(frag)
                                if selfies_frag not in frag_int:
                                    frag_int[selfies_frag] = float(row[2])
                                    frag_mz[selfies_frag] = float(row[1])
                                    break
                                else:
                                    if frag_int[selfies_frag] < float(row[2]):
                                        frag_int[selfies_frag] = float(row[2])
                                        frag_mz[selfies_frag] = float(row[1])
                                        break
                                # frag_int[selfies_frag] = float(row[2])
                                # frag_mz[selfies_frag] = float(row[1])
                                # break
                            except selfies.exceptions.EncoderError:
                                pass

                        # else:


                    except ValueError:
                        print(curr_id + ': not a float')

        sorted_frags = sorted(frag_int, key=frag_int.get, reverse=True)

        frags_str = []
        frags_mz_str = []
        frags_int_str = []
        frags_mz_int_str = []
        frag_spec_list = []
        str_mz_int = []
        for frag in sorted_frags:
            frags_str.append({"frag":frag})
            frags_mz_str.append({"frag": frag, "mz":round(frag_mz[frag], 2)})
            frags_int_str.append({"frag": frag, "int": math.ceil(frag_int[frag] * 10)})
            frags_mz_int_str.append({"frag": frag, "mz": round(frag_mz[frag], 2), "int":
                math.ceil(frag_int[frag] * 10)})
            frag_spec_list.append([frag, round(frag_mz[frag], 2),
                math.ceil(frag_int[frag] * 10)])
            str_mz_int.append({"mz": round(frag_mz[frag], 2), "int":
                math.ceil(frag_int[frag] * 10)})

    else:
        sorted_frags = []
        frags_str = []
        frags_mz_str = []
        frags_int_str = []
        frags_mz_int_str = []
        frag_spec_list = []
        str_mz_int = []
    return sorted_frags, frags_str, frags_mz_str, frags_int_str, frag_spec_list, frags_mz_int_str, str_mz_int


def load_subformulae(curr_id, data_path='./data/'):
    sorted_subforms = None
    if os.path.exists(data_path + 'subformulae_default/' + curr_id + '.json'):
        with open(data_path + 'subformulae_default/' + curr_id + '.json') as f:
            peak_annotations = json.load(f)

        sf_int = {}
        sf_mz = {}

        if peak_annotations['output_tbl'] != None:
            assert len(peak_annotations['output_tbl']['mz'])==len(peak_annotations['output_tbl']['formula'])
            for i in range(0, len(peak_annotations['output_tbl']['mz'])):
                sf_int[peak_annotations['output_tbl']['formula'][i]] = float(peak_annotations['output_tbl']['ms2_inten'][i])
                sf_mz[peak_annotations['output_tbl']['formula'][i]] = float(peak_annotations['output_tbl']['mz'][i])
            # print(len(frag_int))
            sorted_subforms = sorted(sf_int, key=sf_int.get, reverse=True)
            subforms_str = []
            subforms_mz_str = []
            subforms_int_str = []
            subforms_mz_int_str = []
            subform_spec_list = []
            for subform in sorted_subforms:
                subforms_str.append({"subformula":subform})
                subforms_mz_str.append({"subformula": subform, "mz":round(sf_mz[subform], 2)})
                subforms_int_str.append({"subformula": subform, "int": math.ceil(sf_int[subform] * 10)})
                subforms_mz_int_str.append({"subformula": subform, "mz": round(sf_mz[subform], 2), "int":
                    math.ceil(sf_int[subform] * 10)})
                subform_spec_list.append([subform, round(sf_mz[subform], 2),
                    math.ceil(sf_int[subform] * 10)])

    if sorted_subforms == None:
        sorted_subforms = []
        subforms_str = []
        subforms_mz_str = []
        subforms_int_str = []
        subforms_mz_int_str = []
        subform_spec_list = []
    return sorted_subforms, subforms_str, subforms_mz_str, subforms_int_str, subform_spec_list, subforms_mz_int_str


def mask_fragments(fragment_list, mask_ratio=0.1):
    num_to_mask = max(1, int(len(fragment_list) * mask_ratio))
    if len(fragment_list) > 0:
        masked_indices = random.sample(range(len(fragment_list)), num_to_mask)
        masked_frags = []
        for i, frag in enumerate(fragment_list):
            if i in masked_indices:
                masked_frags.append({"frag":"[MASK]"})
            else:
                masked_frags.append({"frag":frag})
        # masked_frags = ["<<FRAG>> [MASK]" if i in masked_indices else f"<<FRAG>> {frag}" for i, frag in enumerate(fragment_list)]
        # masked_frags = "\n".join(masked_frags)
    else:
        masked_frags = []

    return masked_frags

def mask_intensities(fragment_list, mask_ratio=0.1):
    num_to_mask = max(1, int(len(fragment_list) * mask_ratio))
    if len(fragment_list) > 0:
        masked_indices = random.sample(range(len(fragment_list)), num_to_mask)
        masked_frags = []
        for i, frag in enumerate(fragment_list):
            if i in masked_indices:
                masked_frags.append({"frag": frag[0], "mz": frag[1], "int":"[MASK]"})
                # masked_frags.append("<<FRAG>> " + frag[0] + " <m/z=" + frag[1] + "> <int=[MASK]>")
            else:
                masked_frags.append({"frag": frag[0], "mz": frag[1], "int":frag[2]})
                # masked_frags.append("<<FRAG>> " + frag[0] + " <m/z=" + frag[1] + "> <int=" + frag[2] + ">")
        # masked_frags = ["<<FRAG>> {frag[0]} <m/z={frag[1]}> <int={frag[2]}>" if i in masked_indices else f"<<FRAG>> {frag[0]} <m/z={frag[1]}> <int={frag[2]}>" for i, frag in enumerate(fragment_list)]
        # masked_frags = "\n".join(masked_frags)
    else:
        masked_frags = []

    return masked_frags


def add_method_features(spec_data,masking_ratio=0.15):
    for spec in spec_data:
        spec['selfies_masked'] = mask_selfies(spec['selfies'], masking_ratio)

        # spec['sorted_frags'], spec['frags_str'], spec['frags_mz'], spec['frags_int'], spec['frag_spec_list'], spec['frags_mz_int'] = load_fragments(spec['identifier'])
        spec['sorted_frags'], spec['frags_str'], spec['frags_mz'], spec['frags_int'], spec['frag_spec_list'], spec['frags_mz_int'], spec['str_mz_int'] = load_fragments(spec['identifier'])
        spec['sorted_subforms'], spec['subforms_str'], spec['subforms_mz'], spec['subforms_int'], spec['subform_spec_list'], spec['subforms_mz_int'] = load_subformulae(spec['identifier'])
        spec['frags_masked'] = mask_fragments(spec['sorted_frags'])
        spec['frags_int_masked'] = mask_intensities(spec['frag_spec_list'])
        if spec['adduct'] == None:
            spec['adduct'] = 'n/a'
        if spec['instrument_type'] == None:
            spec['instrument_type'] = 'n/a'
        if spec['collision_energy'] == None:
            spec['collision_energy'] = 'n/a'
        spec['exp_settings'] = {"adduct": spec['adduct'], "instrument": spec['instrument_type'], "collision_energy": spec['collision_energy']}

    return spec_data

def load_test_data(data_path):
    if os.path.exists(data_path + "test_data.pkl"):
        with open(data_path + "test_data.pkl", 'rb') as f:
            test_data = pickle.load(f)
    else:
        unfiltered_data = load_data(data_path)
        unfiltered_test_data = unfiltered_data['val'].filter(lambda row: row['fold'] == 'test')
        with open(data_path + "test_data_unfiltered.pkl", "wb") as f:
            pickle.dump(unfiltered_test_data, f)

        # filter for only peaks that have peak annotations and save
        test_data_no_seflies = filter_unannotated_peaks(unfiltered_test_data, data_path)
        random.shuffle(test_data_no_seflies)
        with open(data_path + "test_data_no_selfies.pkl", "wb") as f:
            pickle.dump(test_data_no_seflies, f)
        test_data = add_selfies_to_spec_data(test_data_no_seflies)
        with open(data_path + "test_data.pkl", "wb") as f:
            pickle.dump(test_data, f)
    return test_data


def load_train_data(data_path):
    if os.path.exists(data_path + "train_data.pkl"):
        with open(data_path + "train_data.pkl", 'rb') as f:
            train_data = pickle.load(f)
    else:
        unfiltered_data = load_data(data_path)
        unfiltered_train_data = unfiltered_data['val'].filter(lambda row: row['fold'] == 'train')
        with open(data_path + "train_data_unfiltered.pkl", "wb") as f:
            pickle.dump(unfiltered_train_data, f)
        # filter for only peaks that have peak annotations and save
        train_data_no_seflies = filter_unannotated_peaks(unfiltered_train_data, data_path)
        random.shuffle(train_data_no_seflies)
        with open(data_path + "train_data_no_selfies.pkl", "wb") as f:
            pickle.dump(train_data_no_seflies, f)
        train_data = add_selfies_to_spec_data(train_data_no_seflies)
        with open(data_path + "train_data.pkl", "wb") as f:
            pickle.dump(train_data, f)
    return train_data

def load_val_data(data_path):
    if os.path.exists(data_path + "val_data.pkl"):
        with open(data_path + "val_data.pkl", 'rb') as f:
            val_data = pickle.load(f)
    else:
        unfiltered_data = load_data(data_path)
        unfiltered_val_data = unfiltered_data['val'].filter(lambda row: row['fold'] == 'val')
        with open(data_path + "val_data_unfiltered.pkl", "wb") as f:
            pickle.dump(unfiltered_val_data, f)
        val_data_no_seflies = filter_unannotated_peaks(unfiltered_val_data, data_path)
        random.shuffle(val_data_no_seflies)
        with open(data_path + "val_data_no_selfies.pkl", "wb") as f:
            pickle.dump(val_data_no_seflies, f)
        val_data = add_selfies_to_spec_data(val_data_no_seflies)
        with open(data_path + "val_data.pkl", "wb") as f:
            pickle.dump(val_data, f)
    return val_data

def load_cands_dict(data_path):
    # candidates
    if os.path.exists(data_path + "cands_dict.pkl"):
        with open(data_path + "cands_dict.pkl", "rb") as f:
            cands_dict = pickle.load(f)
    else:
        with open(data_path + 'cand_dict_large_form.pkl', 'rb') as f:
            cands_dict = pickle.load(f)
        for k, v in cands_dict.items():
            random.shuffle(cands_dict[k])
        with open(data_path + "cands_dict.pkl", "wb") as f:
            pickle.dump(cands_dict, f)
    return cands_dict

def order_by_complexity(data_pts, total_data, model_encoding):

    max_length = 0
    max_frags = 0
    for pt in data_pts:
        num_frags = len(pt['frags'])
        num_tokens = len(model_encoding.encode(pt['messages'][1]['content']))
        if num_frags > max_frags:
            max_frags = num_frags
        if num_tokens > max_length:
            max_length = num_tokens
        pt['scores'] = [num_frags, num_tokens]

    for pt in data_pts:
        frag_score = pt['scores'][0] / max_frags
        length_score = pt['scores'][1] / max_length
        pt['complexity'] = frag_score+length_score

    ordered_data = sorted(data_pts, key=lambda item: item['complexity'])

    return ordered_data

def avg_num_tokens(data_pts, model_encoding):
    max_length = 0
    max_frags = 0
    l_num_tokens = []
    for pt in data_pts:
        num_frags = len(pt['frags'])
        num_tokens = len(model_encoding.encode(pt['messages'][1]['content']))
        if num_frags > max_frags:
            max_frags = num_frags
        if num_tokens > max_length:
            max_length = num_tokens
        pt['complexity'] = [num_frags, num_tokens]
        l_num_tokens.append(num_tokens)
    print(sum(l_num_tokens) / len(l_num_tokens))
    return

def split_for_masking_training(data_pts, masking_ratio=0.2):

    #clean
    cleaned_data = []
    for pt in data_pts:
        if len(pt['sorted_frags']) > 0:
            cleaned_data.append(pt)


    #split into two objects
    list_len = len(cleaned_data)
    index = int(list_len * masking_ratio)
    return cleaned_data[:index], cleaned_data[index:]
