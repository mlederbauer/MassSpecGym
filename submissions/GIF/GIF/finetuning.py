import os
import openai
import time
import requests
import hashlib
import json


def get_grading_function(task):
    if task == 'SubformulaPrediction':
        grading_function = """def grade(sample, item) -> float:
    output_text = sample["output_text"]
    reference_answer = item["reference_answer"]
    true_sfs = []
    for temp in reference_answer:
        true_sfs.append(temp['subformula'])

    gen_sfs = []
    if 'subformulae' in output_text.keys():
        for pt in response['subformulae']:
            assert list(pt.keys()) == ['subformula']
            gen_sfs.append(pt['subformula'])
    
    pattern = r'([A-Z][a-z]?)(\d*)'

    true_sf_dicts = []
    for sf in true_sfs:
        elements = re.findall(pattern, sf)
        counts = Counter()
        for element, count in elements:
            counts[element] += int(count) if count else 1
        true_sf_dicts.append(counts)
    gen_sf_dicts = []
    for sf in gen_sfs:
        elements = re.findall(pattern, sf)
        counts = Counter()
        for element, count in elements:
            counts[element] += int(count) if count else 1
        gen_sf_dicts.append(counts)
    c1 = Counter(frozenset(d.items()) for d in true_sf_dicts)
    c2 = Counter(frozenset(d.items()) for d in gen_sf_dicts)
    unordered_sf_accuracy = len(c1 & c2)/ len(c1)
    
    return unordered_sf_accuracy
"""
    else:
        grading_function = ""
    return grading_function


def get_file_id(filename,client):
    """Retrieves the file ID of a file with the given filename."""
    try:
        for file in client.files.list():
            if file.filename == filename:
                return file.id
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def upload_file(file_path, client):
    existing_file_id = get_file_id(file_path.split('/')[-1], client)
    if existing_file_id:
        print(f"File already uploaded: {existing_file_id}")
        return existing_file_id

    print(f"Uploading {file_path}...")
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    print(f"Uploaded as {response.id}")
    return response.id



def fine_tune_model(training_file_id, validation_file_id, client, base_model='gpt-4o-mini', suffix='fine-tuned'):
    print(f"Starting fine-tuning for {training_file_id}...")
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=base_model,
        suffix=suffix,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "learning_rate_multiplier": "auto",
                    "n_epochs": "auto", #2,
                    "batch_size": "auto"
                },
            },
        }
    )
    print("Fine-tuning job started. Job ID:", response.id)
    return response.id


def wait_for_completion(job_id, client):
    while True:
        job_status = client.fine_tuning.jobs.retrieve(job_id)
        print(f"Job Status: {job_status.status}")
        if job_status.status in ['succeeded', 'failed', 'cancelled']:
            print("Job finished with status:", job_status.status)
            return job_status
        time.sleep(60)




def fine_tune_pipeline(task_files, base_model, api_key):

    client = openai.OpenAI(api_key=api_key)

    for i, task in enumerate(task_files):
        print(f"\n=== Fine-tuning on {task['file']} ===")
        file_id = upload_file(task['file'], client)
        val_file_id = upload_file(task['val_file'], client)
        job_id = fine_tune_model(file_id, val_file_id, client, base_model=base_model, suffix=task['suffix'])
        status = wait_for_completion(job_id, client)

        if status.status == 'succeeded':
            fine_tuned_model = status
            print(f"Fine-tuned model available: {fine_tuned_model}")
            # run_evaluation(fine_tuned_model, eval_file)
            base_model = fine_tuned_model.fine_tuned_model  # Continue from the fine-tuned model
        else:
            print(f"Fine-tuning failed at step {i + 1}. Exiting.")
            break

    return fine_tuned_model.fine_tuned_model

def reinforcement_fine_tune_model(training_file_id, validation_file_id, client, task_schema, grading_function, base_model='o4-mini', suffix='fine-tuned'):
    print(f"Starting fine-tuning for {training_file_id}...")



    grader = {
        "type": "python",
        "source": grading_function
    }

    method_dict = {
        "type": "reinforcement",
        "reinforcement": {
            "grader": grader,
            # "response_format": {
            #     "type": "json_schema",
            #     "json_schema": task_schema
            # },
            "hyperparameters": {
                "learning_rate_multiplier": 0.1,
                "batch_size": 16
            }
        }
    }

    # Full serialization to ensure exact correctness
    method_json = json.loads(json.dumps(method_dict))

    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=base_model,
        suffix=suffix,
        method=method_json,
        seed=0
    )

    print("Fine-tuning job started. Job ID:", response.id)
    return response.id

def reinforcement_fine_tune_pipeline(task_files, base_model, api_key):

    client = openai.OpenAI(api_key=api_key)


    for i, task in enumerate(task_files):
        print(f"\n=== Reinforcment Fine-tuning on {task['file']} ===")
        file_id = upload_file(task['file'], client)
        val_file_id = upload_file(task['val_file'], client)
        grading_function = get_grading_function(task['task_name'])

        job_id = reinforcement_fine_tune_model(file_id, val_file_id, client, task['task_schema'], grading_function, base_model=base_model, suffix=task['suffix'])
        status = wait_for_completion(job_id, client)

        if status.status == 'succeeded':
            fine_tuned_model = status
            print(f"Fine-tuned model available: {fine_tuned_model}")
            base_model = fine_tuned_model.fine_tuned_model  # Continue from the fine-tuned model
        else:
            print(f"Fine-tuning failed at step {i + 1}. Exiting.")
            break


if __name__ == "__main__":
    fine_tune_pipeline()
