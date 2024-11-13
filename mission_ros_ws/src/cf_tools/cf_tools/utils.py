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

def find_objects_and_clusters_by_ids(scene_desc, target_ids):
    clusters = []

    for id_ in target_ids:
        for cluster in scene_desc:
            if cluster['cluster_id'] == id_:
                existing_cluster = next((c for c in clusters if c['cluster_id'] == cluster['cluster_id']), None)
                if existing_cluster:
                    existing_cluster['objects'].extend(cluster['objects'])
                else:
                    clusters.append({
                        "cluster_id": cluster['cluster_id'],
                        "label": cluster['label'],
                        "description": cluster['description'],
                        "objects": cluster['objects'][:]  # Copy objects
                    })
                break

            for obj in cluster['objects']:
                if obj['object_id'] == id_:
                    existing_cluster = next((c for c in clusters if c['cluster_id'] == cluster['cluster_id']), None)
                    if existing_cluster:
                        existing_cluster['objects'].append(obj)
                    else:
                        clusters.append({
                            "cluster_id": cluster['cluster_id'],
                            "label": cluster['label'],
                            "description": cluster['description'],
                            "objects": [obj]
                        })
                    break

    return clusters



def query_groq(query, system_prompt, scene_desc, global_context, client):
    aggregate_relevant = []

    for idx in range(len(scene_desc) + 1):
        if idx < len(scene_desc):
            print(f"Querying LLM with cluster {idx}")
            chunk = scene_desc[idx]
        else:  # On last iteration pass the aggregate_relevant_objects
            print(f"Final query")
            chunk = aggregate_relevant

        message= {
            "global_context":global_context,
            "clusters": chunk
        }

        chat_completion = client.chat.completions.create(
            messages=[
                {"role":    "system",       "content":  system_prompt},
                {"role":    "user",         "content":  json.dumps(chunk)},
                {"role":    "assistant",    "content":  "I'm ready."},
                {"role":    "user",         "content":  query},
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            #max_tokens=1024,
            top_p=1,
            #stream=False,
            response_format={"type": "json_object"},
        )
        response = json.loads(chat_completion.choices[0].message.content)
        print(f"response : {response}")

        aggregate_relevant.extend(
            find_objects_and_clusters_by_ids(scene_desc, response['final_relevant'])
        )

    try:
        # Try parsing the response as JSON
        response = json.loads(chat_completion.choices[0].message.content)
        # response = json.dumps(response, indent=4)
    except:
        # Otherwise, just print the response
        response = chat_completion.choices[0].message.content
        print("NOTE: The language model did not produce a valid JSON")

    return response


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


def clip_similarities(query, objects, clip_tokenizer, clip_model):
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
    return similarities





