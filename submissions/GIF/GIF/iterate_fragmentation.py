import os
import json
import pickle
import multiprocessing as mp

from metrics import selfies_to_smiles, is_substructure
from query import query_model, query_model_feedback, mp_query

def iterative_feedback_run(previous_task, new_task, type_of_task, dataset, save_path, data_path, client, model_name, temperature):
    queries = []
    feedback_queries = []
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
            for frag in gen_frags:
                if is_substructure(frag, data['smiles']):
                    true_substructure.append(frag)

            not_true_substructure = len(gen_frags) - len(true_substructure)

            new_frags_str = []
            for frag in true_substructure:
                new_frags_str.append({"frag": frag})

            ##########################
            # feedback query
            temp = {
                "system": "You are a chemistry model specialized in mass spectrometry. In this task, give feedback on your previous prediction of the most likely mass spectrometry fragments based on molecular structure and fragmentation patterns.",
                "user": "<<TASK=IterativeFragmentListPrediction>> You predicted all major fragments (ordered by descending intensity) in SELFIES format based on this information:\n<<MOL>> [[SELFIES]]\n<<EXP_SETTINGS>> [[EXP_SETTINGS]]\n\nYour response was: [[GENERATED_FRAGMENTS]]\n\nPlease give feedback on this response.",
                # "assistant": "[[FRAGMENTS]]"
            }
            user_prompt = temp["user"].replace("[[GENERATED_FRAGMENTS]]", str(response)).replace("[[SELFIES]]", data["selfies"]).replace(
                "[[EXP_SETTINGS]]",
                json.dumps(data["exp_settings"],
                           indent=None).replace(
                    '"', ''))
            data_point = {"messages": [{"role": "system", "content": temp["system"]},
                                       {"role": "user", "content": user_prompt}],
                          "identifier": data['identifier'],
                          "frags": data['sorted_frags']}
            feedback_queries.append(data_point)
            if not os.path.exists(save_path + 'raw/json/' + new_task + '/' + data[
                'identifier'] + '_' + new_task + '_feedback_model_response.txt') and not os.path.exists(save_path + 'raw/json/' + new_task + '/' + data[
                'identifier'] + '_' + type_of_task + '_feedback_model_response.txt'):

                # submit and save query
                feedback_response = query_model_feedback(client, model_name, data_point, temperature)
                if feedback_response == {}:  # if nothing was returned
                    response_dict = {}
                else:
                    response_dict = feedback_response.model_dump()
                with open(save_path + 'raw/pkl/' + new_task + '/' + data[
                    'identifier'] + '_' + new_task + 'feedback_model_response.pkl',
                          'wb') as file:
                    pickle.dump(response_dict, file)
                with open(save_path + 'raw/json/' + new_task + '/' + data[
                    'identifier'] + '_' + new_task + '_feedback_model_response.txt', "w", encoding="utf-8") as file:
                    file.write(feedback_response.choices[0].message.content)
                feedback_response_str = feedback_response.choices[0].message.content

            else:
                with open(save_path + 'raw/json/' + new_task + '/' + data[
                    'identifier'] + '_' + new_task + '_feedback_model_response.txt', "r", encoding="utf-8") as file:
                    feedback_response_str = file.read()

            ##########################
            # create new query
            temp = {
                "system": "You are a chemistry model specialized in mass spectrometry. In this task, predict the most likely mass spectrometry fragments based on molecular structure and fragmentation patterns.",
                "user": "<<TASK=IterativeFragmentListPrediction>> You predicted all major fragments (ordered by descending intensity) in SELFIES format based on this information:\n<<MOL>> [[SELFIES]]\n<<EXP_SETTINGS>> [[EXP_SETTINGS]]\n\nYour response was: [[GENERATED_FRAGMENTS]]\n\nYour feedback was: [[FEEDBACK]]\n\n[[INVALID_SUBSTRUCTURES]] of the predicted fragments were invalid substructures, and the remaining are possible: <<FRAGMENTS>> [[PREVIOUS FRAGMENTS]]. Respond with the final list of fragments of the molecule (ordered by descending intensity) in SELFIES format produced by these experiment settings.",
                "assistant": "[[FRAGMENTS]]"
            }
            user_prompt = temp["user"].replace("[[GENERATED_FRAGMENTS]]", str(response)).replace("[[FEEDBACK]]",str(feedback_response_str)).replace("[[INVALID_SUBSTRUCTURES]]", str(not_true_substructure)).replace(
                "[[PREVIOUS FRAGMENTS]]", str(new_frags_str)).replace("[[SELFIES]]", data["selfies"]).replace(
                "[[EXP_SETTINGS]]",
                json.dumps(data["exp_settings"],
                           indent=None).replace(
                    '"', ''))
            assistant_prompt = temp["assistant"].replace("[[FRAGMENTS]]",
                                                         json.dumps(data["frags_str"], indent=None).replace('"', ''))
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
    with open(save_path+'iterative_prompts/test_queries_' + new_task + '_feedback.jsonl', "w") as f:
        for pt in feedback_queries:
            f.write(json.dumps(pt) + "\n")

    return

def iterative_run(previous_task, new_task, type_of_task, dataset, save_path, data_path, client, model_name, temperature):
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
            # # convert to smiles
            # smi_gen_frags = []
            # for frag in gen_frags:
            #     smi_gen_frags.append(selfies_to_smiles(frag))
            true_substructure = []
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
            user_prompt = temp["user"].replace("[[INVALID_SUBSTRUCTURES]]", str(not_true_substructure)).replace(
                "[[PREVIOUS FRAGMENTS]]", str(new_frags_str)).replace("[[SELFIES]]", data["selfies"]).replace(
                "[[EXP_SETTINGS]]",
                json.dumps(data["exp_settings"],
                           indent=None).replace(
                    '"', ''))
            assistant_prompt = temp["assistant"].replace("[[FRAGMENTS]]",
                                                         json.dumps(data["frags_str"], indent=None).replace('"', ''))
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

def mp_iterative_run(previous_task, new_task, type_of_task, dataset, save_path, data_path, client, model_name, temperature, max_processes, api_key):
    queries = []

    count = 0
    processes = []
    while count < len(dataset):
        # for data in dataset:
        while len(processes) < max_processes:
            if count < len(dataset):
                data = dataset[count]
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
                    user_prompt = temp["user"].replace("[[INVALID_SUBSTRUCTURES]]", str(not_true_substructure)).replace(
                        "[[PREVIOUS FRAGMENTS]]", str(new_frags_str)).replace("[[SELFIES]]", data["selfies"]).replace(
                        "[[EXP_SETTINGS]]",
                        json.dumps(data["exp_settings"],
                                   indent=None).replace(
                            '"', ''))
                    assistant_prompt = temp["assistant"].replace("[[FRAGMENTS]]",
                                                                 json.dumps(data["frags_str"], indent=None).replace('"',
                                                                                                                    ''))
                    data_point = {"messages": [{"role": "system", "content": temp["system"]},
                                               {"role": "user", "content": user_prompt},
                                               {"role": "assistant", "content": assistant_prompt}],
                                  "identifier": data['identifier'],
                                  "frags": data['sorted_frags']}
                    queries.append(data_point)

                    if not os.path.exists(save_path + 'raw/json/' + new_task + '/' + data[
                        'identifier'] + '_' + new_task + '_model_response.json') and not os.path.exists(
                        save_path + 'raw/json/' + new_task + '/' + data[
                            'identifier'] + '_' + type_of_task + '_model_response.json'):

                        # response = query_model(client, model_name, type_of_task, data_point, temperature)
                        p = mp.Process(target=mp_query, args=[api_key, model_name, type_of_task, data_point, save_path, temperature, new_task])
                        p.start()
                        processes.append(p)

                count += 1
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

def iterate_fragmentation(dataset, save_path, data_path, client, model_name, n=4, max_processes=None, api_key=None):
    # previous_task = 'FragmentListPrediction'
    # new_task = 'IterativeFragmentListPrediction'
    type_of_task = 'IterativeFragmentListPrediction'

    # iterative_run(previous_task, new_task, type_of_task, dataset, save_path, data_path, client, model_name)
    temperature = 0.9
    for i in range(1,n+1):
        if i==4:
            temperature = 0.1
        if i==1:
            previous_task = 'FragmentListPrediction'
        else:
            previous_task = 'IterativeFragmentListPrediction' + str(i - 1)
            # new_task = 'IterativeFragmentListPrediction' + str(i)
        new_task = 'IterativeFragmentListPrediction' + str(i)
        if not os.path.isdir(save_path + 'raw/json/' + new_task):
            os.mkdir(save_path + 'raw/json/' + new_task)
        if not os.path.isdir(save_path + 'raw/pkl/' + new_task):
            os.mkdir(save_path + 'raw/pkl/' + new_task)
        if max_processes!=None:
            mp_iterative_run(previous_task, new_task, type_of_task, dataset, save_path, data_path, client, model_name,
                          temperature, max_processes, api_key)
        else:
            iterative_run(previous_task, new_task, type_of_task, dataset, save_path, data_path, client, model_name, temperature)
        # iterative_feedback_run(previous_task, new_task, type_of_task, dataset, save_path, data_path, client, model_name, temperature)

    return

