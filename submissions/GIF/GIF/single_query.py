import argparse
import json
import pickle
import sys
from openai import OpenAI

from query import query_model

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model_name", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--task", default="FragmentListPrediction")
    parser.add_argument("--data", help="Optional JSON string of the data dict")
    parser.add_argument("--save_path", default="./output/")
    parser.add_argument("--temperature", default=None)
    parser.add_argument("--save_task", default="None")
    parser.add_argument("--from-stdin", action="store_true",
                        help="Read JSON payload from stdin instead of --data")
    args = vars(parser.parse_args())
    # api_key, model_name, task, data, save_path

    client = OpenAI(api_key=args["api_key"], timeout=360)
    model_name = args["model_name"]
    task = args["task"]
    save_path = args["save_path"]
    temperature = args["temperature"]
    save_task = args["save_task"]
    if save_task == "None":
        save_task = task

    # payload = args["data"]
    # data = json.loads(payload)
    # print(data)
    if args["from_stdin"] or not args["data"]:
        buf = sys.stdin.read()
        if not buf.strip():
            print("ERROR: no JSON on stdin and --data not provided", file=sys.stderr)
            sys.exit()
        payload = buf
    else:
        payload = args["data"]

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        print(f"ERROR: bad JSON payload: {e}", file=sys.stderr)
        sys.exit()

    print(data)
    if temperature==None:
        response = query_model(client, model_name, task, data)
    else:
        temperature = float(temperature)
        response = query_model(client, model_name, task, data, temperature)
    if response == {}:  # if nothing was returned
        response_dict = {}
    else:
        response_dict = response.model_dump()
    with open(save_path + 'raw/pkl/' + save_task + '/' + data['identifier'] + '_' + save_task + '_model_response.pkl',
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
    with open(save_path + 'raw/json/' + save_task + '/' + data['identifier'] + '_' + save_task + '_model_response.json',
              "w") as file:
        json.dump(answer, file, indent=2)


