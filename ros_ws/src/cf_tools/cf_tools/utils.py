import os

import json
from typing import Any
import numpy as np
#import openai
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

#openai.api_key = os.getenv("OPENAI_API_KEY")


def read_json_file(filepath):
    # Read the uploaded JSON file
    content = None
    with open(filepath, "r") as f:
        content = f.read()
    data = json.loads(content)
    return data

def find_objects_by_ids(object_list, target_ids):
    return [obj for obj in object_list if obj['id'] in target_ids]


def query_llm(query, system_prompt, scene_desc):
    CHUNK_SIZE = 80  # Adjust this size as needed

    scene_desc_chunks = [scene_desc[i:i + CHUNK_SIZE] for i in range(0, len(scene_desc), CHUNK_SIZE)]
    aggregate_relevant_objects = []

    # for idx, chunk in enumerate(scene_desc_chunks):

    for idx in range(len(scene_desc_chunks) + 1):

        if idx < len(scene_desc_chunks):
            chunk = scene_desc_chunks[idx]
        else:  # On last iteration pass the aggregate_relevant_objects
            print(f"final query")
            chunk = aggregate_relevant_objects
            print(f"chunk : {chunk}")

        scene_desc_str = json.dumps(chunk)
        num_tokens = len(scene_desc_str.split())

        chat_completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(chunk)},
                {"role": "assistant", "content": "I'm ready."},
                {"role": "user", "content": query},
            ],
            temperature=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        response = json.loads(chat_completion["choices"][0]["message"]["content"])
        print(f"Line 183, response : {response}")

        curr_relevant_objects = find_objects_by_ids(
            chunk, response['final_relevant_objects'])

        aggregate_relevant_objects.extend(curr_relevant_objects)
    try:
        # Try parsing the response as JSON
        response = json.loads(chat_completion["choices"][0]["message"]["content"])
        # response = json.dumps(response, indent=4)
    except:
        # Otherwise, just print the response
        response = chat_completion["choices"][0]["message"]["content"]
        print("NOTE: The language model did not produce a valid JSON")

    return response


def query_clip(query, objects, clip_tokenizer, clip_model):
    print(f"Querying CLIP with '{query}'")
    text_queries_tokenized = clip_tokenizer(query) #.to("cuda")
    text_query_ft = clip_model.encode_text(text_queries_tokenized)
    text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
    text_query_ft = text_query_ft.squeeze()
    
    # similarities = objects.compute_similarities(text_query_ft)
    objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
    objects_clip_fts = objects_clip_fts #.to("cuda")
    similarities = F.cosine_similarity(
        text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
    )
    probs = F.softmax(similarities, dim=0)
    max_prob_idx = torch.argmax(probs).item()

    max_prob_object = objects[max_prob_idx]
    print(f"Most probable object is at index {max_prob_idx} with class name '{max_prob_object['class_name']}'")
    print(f"location xyz: {max_prob_object['bbox'].center}")
    return max_prob_idx, max_prob_object['class_name'], max_prob_object['bbox'].center


