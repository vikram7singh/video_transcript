from langfuse import Langfuse
import openai
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

load_dotenv()

# Load documents from a directory (you can change this path as needed)
documents = SimpleDirectoryReader(input_files=["video_transcript_1.txt",]).load_data()

# Create an index from the documents
index = VectorStoreIndex.from_documents(documents)

# Create a retriever to fetch relevant documents
retriever = index.as_retriever(retrieval_mode='similarity', k=3)

langfuse = Langfuse()


# def print_retrieved_docs():
#     query = "What years does the strategic plan cover?"

#     # Retrieve relevant documents
#     relevant_docs = retriever.retrieve(query)

#     print(f"Number of relevant documents: {len(relevant_docs)}")
#     print("\n" + "="*50 + "\n")

#     for i, doc in enumerate(relevant_docs):
#         print(f"Document {i+1}:")
#         print(f"Text sample: {doc.node.get_content()[:200]}...")  # Print first 200 characters
#         print(f"Metadata: {doc.node.metadata}")
#         print(f"Score: {doc.score}")
#         print("\n" + "="*50 + "\n")

# print_retrieved_docs()



# we use a very simple eval here, you can use any eval library
# see https://langfuse.com/docs/scores/model-based-evals for details
def llm_evaluation(output, expected_output):
    client = openai.OpenAI()
    
    prompt = f"""
    Compare the following output with the expected output and evaluate its accuracy:
    
    Output: {output}
    Expected Output: {expected_output}
    
    Provide a score (0 for incorrect, 1 for correct) and a brief reason for your evaluation.
    For correct answers, output and expected output shouldn't necessarily be exactly the same, but the overall meaning should be the same.
    Return your response in the following JSON format:
    {{
        "score": 0 or 1,
        "reason": "Your explanation here"
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with evaluating the accuracy of responses."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    evaluation = response.choices[0].message.content
    result = eval(evaluation)  # Convert the JSON string to a Python dictionary
    
    # Debug printout
    print(f"Output: {output}")
    print(f"Expected Output: {expected_output}")
    print(f"Evaluation Result: {result}")
    
    return result["score"], result["reason"]

from datetime import datetime
 
def rag_query(input):
  
    generationStartTime = datetime.now()

    relevant_docs = retriever.retrieve(input)
    print("::::::::len relevant_docs::::::::", len(relevant_docs))
    relevant_text = "\n".join([doc.node.get_content() for doc in relevant_docs])
    print("::::::::relevant_text::::::::", relevant_text)

    prompt = f"""
    You are a helpful assistant responsible for answering questions based on the provided context.
    You are given the following context from a video transcript and you need to answer the question based on the context. 
    Do not use any other information and stick to the context.
    Context: {relevant_text}
    Return your answer in the following JSON format:
    {{
    "answer": "Provide your answer here"
    }}
    """
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input}
        ],
        temperature=0.2
    )

    evaluation = response.choices[0].message.content
    result = eval(evaluation)  # Convert the JSON string to a Python dictionary
    print("::::::::result::::::::", result)

    output = result["answer"]
    langfuse_generation = langfuse.generation(
    name="strategic-plan-qa",
    input=input,
    output=output,
    model="gpt-3.5-turbo",
    start_time=generationStartTime,
    end_time=datetime.now()
    )

    return output, langfuse_generation

def run_experiment(experiment_name):
    dataset = langfuse.get_dataset("video_1_transcript")

    for item in dataset.items:
        print("::::::::input::::::::", item.input)
        completion, langfuse_generation = rag_query(item.input)

        item.link(langfuse_generation, experiment_name) # pass the observation/generation object or the id

        score, reason = llm_evaluation(completion, item.expected_output)
        langfuse_generation.score(
            name="accuracy",
            value=score,
            comment=reason
        )

run_experiment("Video RAG pipeline 2")