import os
import json
import pickle
import multiprocessing as mp
from metrics import selfies_to_smiles, is_substructure
from query import query_model, mp_query
from prompts import get_generated_fragments

def iterative_run_intensity(generation_task, previous_task, new_task, type_of_task, dataset, save_path, data_path, client, model_name, temperature):
    queries = []
    for data in dataset:
        if os.path.isfile(save_path + 'raw/json/' + generation_task + '/' + data['identifier'] + '_' + generation_task + '_model_response.json') and os.path.isfile(save_path + 'raw/json/' + previous_task + '/' + data['identifier'] + '_' + previous_task + '_model_response.json'):

            with open(save_path + 'raw/json/' + generation_task + '/' + data[
                    'identifier'] + '_' + generation_task + '_model_response.json') as file:
                gen_frag_response = json.load(file)

            gen_frags = get_generated_fragments(gen_frag_response, data)
            gen_frags_dict = {}
            for frag in gen_frags:
                gen_frags_dict[frag['mz']] = [frag['frag']]

            with open(save_path + 'raw/json/' + previous_task + '/' + data[
                'identifier'] + '_' + previous_task + '_model_response.json') as file:
                int_response = json.load(file)

            # process response
            #todo!!! keep order
            # gen_int_mz = []
            mzs = []
            if 'intensities' in int_response.keys():
                for frag in int_response['intensities']:
                    if frag['mz'] in gen_frags_dict.keys() and frag['mz'] not in mzs:
                        if 'int' in frag.keys():
                            if isinstance(frag['int'], int) and frag['int'] >= 0 and frag['int'] <= 10:
                                mzs.append(frag['mz'])
                                gen_frags_dict[frag['mz']].append(frag['int'])
                            # gen_int_mz.append({'frag': gen_frags_dict[frag['mz']], 'mz': frag['mz'], 'int': frag['int']})

            new_str = []
            count = 0
            for k in gen_frags_dict.keys():
                if len(gen_frags_dict[k]) == 2:
                    new_str.append({'frag': gen_frags_dict[k][0], 'mz': k, 'int': gen_frags_dict[k][1]})
                else:
                    new_str.append({'frag': gen_frags_dict[k][0], 'mz': k})
                    count +=1

            # create new query
            temp = {"system":"You are a chemistry model specialized in mass spectrometry. In this task, estimate the intensity of fragments based on their mass, structure, and likely ionization behavior.",
                 "user": "<<TASK=IterativeIntensityFragPrediction>> You predicted the intensity scores (1–10) for provided fragments (ordered by descending intensity). [[INVALID_RESPONSES]] were either incorrect or not included. Provide the intensity of each fragment of the molecules under the specified experiment settings and return the exact corresponding m/z values listed here too. The included intensity values were provided by you in the previous response:\n<<MOL>> [[SELFIES]]\n<<EXP_SETTINGS>> [[EXP_SETTINGS]]\n[[FRAG_LIST_MZ]]",
                 # "assistant": "[[FRAG_LIST_MZ_INT]]"
                 "assistant": "[[LIST_MZ_INT]]"
            }

            user_prompt = temp["user"].replace("[[INVALID_RESPONSES]]", str(count)).replace(
                "[[FRAG_LIST_MZ]]", str(new_str)).replace("[[SELFIES]]", data["selfies"]).replace(
                "[[EXP_SETTINGS]]",
                json.dumps(data["exp_settings"],
                           indent=None).replace(
                    '"', ''))
            assistant_prompt = temp["assistant"].replace("[[LIST_MZ_INT]]",
                                                         json.dumps(data["str_mz_int"], indent=None).replace('"', ''))
            data_point = {"messages": [{"role": "system", "content": temp["system"]},
                                       {"role": "user", "content": user_prompt},
                                       {"role": "assistant", "content": assistant_prompt}],
                          "identifier": data['identifier'],
                          "frags": data['sorted_frags']}
            queries.append(data_point)

            if not os.path.exists(save_path + 'raw/json/' + new_task + '/' + data[
                'identifier'] + '_' + new_task + '_model_response.json') and not os.path.exists(save_path + 'raw/json/' + new_task + '/' + data[
                'identifier'] + '_' + type_of_task + '_model_response.json'):

                # submit and save query
                response = query_model(client, model_name, type_of_task, data_point, temperature)
                if response == {}:  # if nothing was returned
                    response_dict = {}
                else:
                    response_dict = response.model_dump()
                with open(save_path + 'raw/pkl/' + new_task + '/' + data[
                    'identifier'] + '_' + new_task + '_model_response.pkl',
                          'wb') as file:
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
                with open(save_path + 'raw/json/' + new_task + '/' + data[
                    'identifier'] + '_' + new_task + '_model_response.json',
                          "w") as file:
                    json.dump(answer, file, indent=2)

    if not os.path.isdir(save_path+'iterative_prompts/'):
        os.mkdir(save_path+'iterative_prompts/')
    with open(save_path+'iterative_prompts/test_queries_' + new_task + '.jsonl', "w") as f:
        for pt in queries:
            f.write(json.dumps(pt) + "\n")

    return

def mp_iterative_run_intensity(generation_task, previous_task, new_task, type_of_task, dataset, save_path, data_path, client, model_name, temperature, max_processes, api_key):
    queries = []
    # for data in dataset:
    i_count = 0
    processes = []
    while i_count < len(dataset):
        while len(processes) < max_processes:
            if i_count < len(dataset):
                data = dataset[i_count]

                if os.path.isfile(save_path + 'raw/json/' + generation_task + '/' + data['identifier'] + '_' + generation_task + '_model_response.json') and os.path.isfile(save_path + 'raw/json/' + previous_task + '/' + data['identifier'] + '_' + previous_task + '_model_response.json'):
                        # save_path + 'raw/json/' + previous_task + '/' + data[
                        #     'identifier'] + '_' + previous_task + '_model_response.json'):

                    with open(save_path + 'raw/json/' + generation_task + '/' + data[
                            'identifier'] + '_' + generation_task + '_model_response.json') as file:
                        gen_frag_response = json.load(file)

                    gen_frags = get_generated_fragments(gen_frag_response, data)
                    gen_frags_dict = {}
                    for frag in gen_frags:
                        gen_frags_dict[frag['mz']] = [frag['frag']]

                    with open(save_path + 'raw/json/' + previous_task + '/' + data[
                        'identifier'] + '_' + previous_task + '_model_response.json') as file:
                        int_response = json.load(file)

                    # process response
                    #todo!!! keep order
                    # gen_int_mz = []
                    mzs = []
                    if 'intensities' in int_response.keys():
                        for frag in int_response['intensities']:
                            if frag['mz'] in gen_frags_dict.keys() and frag['mz'] not in mzs:
                                if 'int' in frag.keys():
                                    if isinstance(frag['int'], int) and frag['int'] >= 0 and frag['int'] <= 10:
                                        mzs.append(frag['mz'])
                                        gen_frags_dict[frag['mz']].append(frag['int'])
                                    # gen_int_mz.append({'frag': gen_frags_dict[frag['mz']], 'mz': frag['mz'], 'int': frag['int']})

                    new_str = []
                    count = 0
                    for k in gen_frags_dict.keys():
                        if len(gen_frags_dict[k]) == 2:
                            new_str.append({'frag': gen_frags_dict[k][0], 'mz': k, 'int': gen_frags_dict[k][1]})
                        else:
                            new_str.append({'frag': gen_frags_dict[k][0], 'mz': k})
                            count +=1

                    # create new query
                    temp = {"system":"You are a chemistry model specialized in mass spectrometry. In this task, estimate the intensity of fragments based on their mass, structure, and likely ionization behavior.",
                         "user": "<<TASK=IterativeIntensityFragPrediction>> You predicted the intensity scores (1–10) for provided fragments (ordered by descending intensity). [[INVALID_RESPONSES]] were either incorrect or not included. Provide the intensity of each fragment of the molecules under the specified experiment settings and return the exact corresponding m/z values listed here too. The included intensity values were provided by you in the previous response:\n<<MOL>> [[SELFIES]]\n<<EXP_SETTINGS>> [[EXP_SETTINGS]]\n[[FRAG_LIST_MZ]]",
                         # "assistant": "[[FRAG_LIST_MZ_INT]]"
                         "assistant": "[[LIST_MZ_INT]]"
                    }

                    user_prompt = temp["user"].replace("[[INVALID_RESPONSES]]", str(count)).replace(
                        "[[FRAG_LIST_MZ]]", str(new_str)).replace("[[SELFIES]]", data["selfies"]).replace(
                        "[[EXP_SETTINGS]]",
                        json.dumps(data["exp_settings"],
                                   indent=None).replace(
                            '"', ''))
                    assistant_prompt = temp["assistant"].replace("[[LIST_MZ_INT]]",
                                                                 json.dumps(data["str_mz_int"], indent=None).replace('"', ''))
                    data_point = {"messages": [{"role": "system", "content": temp["system"]},
                                               {"role": "user", "content": user_prompt},
                                               {"role": "assistant", "content": assistant_prompt}],
                                  "identifier": data['identifier'],
                                  "frags": data['sorted_frags']}
                    queries.append(data_point)

                    if not os.path.exists(save_path + 'raw/json/' + new_task + '/' + data[
                        'identifier'] + '_' + new_task + '_model_response.json') and not os.path.exists(save_path + 'raw/json/' + new_task + '/' + data[
                        'identifier'] + '_' + type_of_task + '_model_response.json'):

                        p = mp.Process(target=mp_query,
                                       args=[api_key, model_name, type_of_task, data_point, save_path, temperature, new_task])
                        p.start()
                        processes.append(p)

                i_count += 1
            else:
                break
        for p in processes:
            p.join()
        for p in processes:
            p.terminate()
        processes = []


    if not os.path.isdir(save_path+'iterative_prompts/'):
        os.mkdir(save_path+'iterative_prompts/')
    with open(save_path+'iterative_prompts/test_queries_' + new_task + '.jsonl', "w") as f:
        for pt in queries:
            f.write(json.dumps(pt) + "\n")

    return

def iterate_intensity(generation_task, dataset, save_path, data_path, client, model_name, n=1, max_processes=None, api_key=None):
    type_of_task = 'IterativeIntensityFragPrediction'

    # iterative_run(previous_task, new_task, type_of_task, dataset, save_path, data_path, client, model_name)
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
        if max_processes != None:
            mp_iterative_run_intensity(generation_task, previous_task, new_task, type_of_task, dataset, save_path,
                                    data_path, client, model_name, temperature, max_processes, api_key)
        else:
            iterative_run_intensity(generation_task, previous_task, new_task, type_of_task, dataset, save_path, data_path, client, model_name, temperature)

    return

