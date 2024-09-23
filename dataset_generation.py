# Import necessary libraries
from dotenv import load_dotenv
from video_processing.ingest_video import Video
from langfuse import Langfuse

# Load environment variables
load_dotenv()

from openai import OpenAI
import json

client = OpenAI()

"""
Process following vidios: 
python3 video_rag/ingest_video.py --youtube_url "https://www.youtube.com/watch?v=SYRunzR9fbk"
python3 video_processing/ingest_video.py --youtube_url "https://www.youtube.com/watch\?v=RqubKSF3wig&ab_channel=GoNorthwest"
"""


# Function to generate questions and answers
def generate_qa(prompt, text, temperature=0.2):    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}],
        temperature=temperature,
    )
    
    print(response.choices[0].message.content)

    # Strip extraneous symbols from the response content
    content = response.choices[0].message.content.strip()
    
    # Remove potential JSON code block markers
    content = content.strip()
    if content.startswith('```'):
        content = content.split('\n', 1)[-1]
    if content.endswith('```'):
        content = content.rsplit('\n', 1)[0]
    content = content.strip()
    
    # Attempt to parse the cleaned content as JSON
    try:
        parsed_content = json.loads(content.strip())
        return parsed_content
    except json.JSONDecodeError:
        print("Error: Unable to parse JSON. Raw content:")
        print(content)
        return []

factual_prompt = """
You are an expert educational content creator tasked with generating factual questions and answers based on the text transcript from a video. 
These questions should focus on retrieving specific details, figures, definitions, and key facts from the transcript.

Instructions:

- Generate **10** factual questions, each with a corresponding **expected_output**.
- Ensure all questions are directly related to the text transcript from the video.
- Present the output in the following structured JSON format:

[
    {
      "question": "What is the purpose of the 'Go Generate' command in code generation?",
      "expected_output": "The 'Go Generate' command is used to combine templates, data, and logic to produce an output file, typically Go code."
    },
    {
      "question": "What two DigitalOcean open search projects were mentioned in Katrina's talk?",
      "expected_output": "The two open search projects mentioned are Go QMU and Go LibVirt."
    }
]
"""

# Generate dataset
import os
import json


def get_dataset():
    transcript_file = 'video_transcript_1.txt'
    dataset_file = 'video_dataset_1.json'
    dataset = []
    if os.path.exists(dataset_file):
        # Load dataset from local file if it exists
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
    else:
        # Generate dataset if local file doesn't exist
        with open(transcript_file, 'r') as f:
            dataset = generate_qa(factual_prompt, f.read(), temperature=0.2)
        
        # Write dataset to local file
        with open(dataset_file, 'w') as f:
            json.dump(dataset, f)
    
    return dataset
        
# Note: we're choosing to create the dataset in Langfuse below, but it's equally easy to create it in another platform.
def create_dataset_in_langfuse(dataset_name, dataset):
    langfuse = Langfuse()

    langfuse.create_dataset(name=dataset_name);

    for item in dataset:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=item["question"],
            expected_output=item["expected_output"]
        )

create_dataset_in_langfuse(dataset_name="video_1_transcript", dataset=get_dataset())
    