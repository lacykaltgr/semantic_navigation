from groq import Groq
import base64
import json


def parse_json(json_str, exception_output=None):
    try:
        # Try parsing the response as JSON
        response = json.loads(json_str)
        print(f"response : {response}")
        # response = json.dumps(response, indent=4)
    except:
        # Otherwise, just print the response
        print("NOTE: The language model did not produce a valid JSON")
        print("Response: ",  json_str)
        return exception_output

    return response


def query_groq_describe_image(prompt, image_path):
    client = Groq()

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    chat_completion = client.chat.completions.create(
        messages=[
            {"role":    "user",    "content":  [
                {
                    "type": "text", 
                    "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}",},
                },
            ]},
        ],
        model="llama-3.2-11b-vision-preview",
        temperature=0.5,
        #max_tokens=1024,
        top_p=1,
        #stream=False,
        response_format={"type": "json_object"},
    )

    response_string = chat_completion.choices[0].message.content
    exception_output = {"label": "No label available.", "description": "No description available."}
    response = parse_json(response_string, exception_output=exception_output)

    return response


def query_groq_label_cluster(scene_desc, system_prompt, global_context):
    client = Groq()
    clusters = []
    for cluster in scene_desc:
        cluster_desc = {
            "cluster_id": cluster["cluster_id"],
            "objects": [obj["object_tag"] for obj in cluster["objects"]],
            "image_descriptions": cluster["image_descriptions"]
        }
        clusters.append(cluster_desc)

    message= {
        "global_context": global_context,
        "clusters": clusters
    }

    chat_completion = client.chat.completions.create(
        messages=[
            {"role":    "system",       "content":  system_prompt},
            {"role":    "user",         "content":  json.dumps(message)},
        ],
        model="llama3-8b-8192",
        temperature=0.5,
        #max_tokens=1024,
        top_p=1,
        #stream=False,
        response_format={"type": "json_object"},
    )

    response_string = chat_completion.choices[0].message.content
    exception_output = [{
        "cluster_id": f"{cluster['cluster_id']}", 
        "label": "No label available.", 
        "description": "No explanation available."} 
            for cluster in scene_desc]
    response = parse_json(response_string, exception_output=exception_output)

    return response

