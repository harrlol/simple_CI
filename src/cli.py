import os
import argparse
import pandas as pd
import json
import time
from json_repair import repair_json
import glob
from sklearn.model_selection import train_test_split
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, FinishReason
import vertexai.generative_models as generative_models

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
GENERATIVE_MODEL = "gemini-1.5-flash-001"
OUTPUT_FOLDER = "data"
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]

# Configuration settings for the content generation
generation_config = {
    "max_output_tokens": 8192,  # Maximum number of tokens for output
    "temperature": 0.25,  # Control randomness in output, changed from 1 to 0.25 10/18
    "top_p": 0.95,  # Use nucleus sampling
}

# Safety settings to filter out harmful content
safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
    )
]

# System Prompt
SYSTEM_INSTRUCTION = """
    Question Independence:
    - Ensure each question-answer pair is completely independent and self-contained
    - Do not reference other questions or answers within the set
    - Each Q&A pair should be understandable without any additional context

    Question Types:
    - Limit the questions to the format "What kinds of tests should a doctor order if a patient presents with...", "What might the diagnosis be if a patient has...", "What medicine might be used to treat a patient with..."
    - Ensure questions cover most if not all of the information you can extrapolate from the note

    Accuracy and Relevance:
      - Ensure all question, especially technical data, is contained within this note, however, answers can be drawn from your knowledge base where necessary, make sure they are correct and up-to-date. 
      - Focus on widely accepted information in the field of medicine expertise and biological science

    Output Format:
    Provide the Q&A pairs in JSON format, with each pair as an object containing 'question' and 'answer' fields, within a JSONL array.
    Follow these strict guidelines:
    1. Use double quotes for JSON keys and string values.
    2. For any quotation marks within the text content, use single quotes (') instead of double quotes. Avoid quotation marks.
    3. If a single quote (apostrophe) appears in the text, escape it with a backslash (\').
    4. Ensure there are no unescaped special characters that could break the JSON structure.
    5. Avoid any Invalid control characters that JSON decode will not be able to decode.
    6. Keep the total length of your text within 8000 tokens, including the prompt. If that means less question-answer pairs than 100, do so.

    Here's an example of the expected format:
    [{"question": "What kinds of tests should a doctor order if a patient presents with shortness of breath, chest pain, and dilated pupil?","answer": "Electrocardiogram, Chest X-ray, CT Scan, Echocardiogram "},{"question": "What might the diagnosis be if a patient has enlarged lymph nodes, acidosis, and anemia?","answer": "Lymphoma, Renal Failure"},{"question": "What medicine might be used to treat a patient with fever, dry cough, and crackles","answer": "Ceftriaxone, Tamiflu, Tylenol"}]
    Note: output ONLY the JSON string, as a simple string of text, do not format it as a code block!!!!!
    """


# Return a blobs objuct from GCP that we can cycle through
def read_gcs_file(bucket_name, file_path):
    # Initialize the client
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # List all the blobs (files) in the folder
    blobs = bucket.list_blobs(prefix=folder_path)
    
    return blobs


def generate():
    print("generate()")

    # Make dataset folders
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Initialize Vertex AI project and location
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    
    # Initialize the GenerativeModel with specific system instructions
    model = GenerativeModel(
        GENERATIVE_MODEL,
        system_instruction=[SYSTEM_INSTRUCTION]
    )

    INPUT_PROMPT = """Generate a set of 100 question-answer pairs about this patient note in English. Adhere to these guidelines. Ensure each pair is independent and self-contained, and stay scientifically accurate and true to the information presented to you. """
    # Loop to generate and save the content
    try:
        bucket_name = 'cliniq-dataset'  # Replace with your bucket name
        folder_path = 'test/'        # Replace with your file path inside the bucket
        # Loop through each file in the folder
        blobs = read_gcs_file(bucket_name, folder_path)
        for blob in blobs:
        # Check if the file ends with '.txt'
            if blob.name.endswith('.txt'):
                file_number = os.path.splitext(os.path.basename(blob))[0]
                print(f"Reading file: {blob.name}")
                doctor_note = blob.download_as_text()
                responses = model.generate_content(
                    [INPUT_PROMPT + doctor_note],  # Input prompt
                    generation_config=generation_config,  # Configuration settings
                    safety_settings=safety_settings,  # Safety settings
                    stream=False,  # Enable streaming for responses
                )
                # pre-processing json
                response_json = responses.text
                print(response_json)
                print("")
                try:
                    json_out = json.loads(response_json)
                    with open('out_' + file_number + '.json', 'w') as f:
                        json.dump(json_out, f, indent=4)
                    print(response_json)
                    print(file_number + ": finished successfully")
                except:
                    try:
                        print(file_number + ": json string is broken, attempting to fix...")
                        response_json = repair_json(response_json)
                        json_out = json.loads(response_json)
                        print(file_number + ": successfully fixed")
                        file_name = f"{OUTPUT_FOLDER}/qa_{file_number}.json"
                        with open(file_name, "w") as file:
                            json.dump(json_out, file, indent=4)
                    except:
                        print(file_number + ": cannot be fixed, skipping...")
                        pass
    except Exception as e:
        print(f"Error occurred while generating content: {e}")


def prepare():
    print("prepare()")

    # Get the generated files
    output_files = glob.glob(os.path.join(OUTPUT_FOLDER, "out_*.json"))
    output_files.sort()

    # Consolidate the data
    output_pairs = []
    errors = []
    for output_file in output_files:
        print("Processing file:", output_file)
        try:
            with open(output_file, "r") as read_file:
                json_responses = json.load(read_file)  # Directly load JSON data
            # Ensure each item in json_responses is a dictionary and has the 'question' key
            for item in json_responses:
                if isinstance(item, dict) and 'question' in item:
                    output_pairs.append(item)
                else:
                    errors.append({"file": output_file, "error": "Missing 'question' key or invalid format"})
        except Exception as e:
            errors.append({"file": output_file, "error": str(e)})

        print("Number of errors:", len(errors))
        print(errors[:5])

    # Save the dataset
    output_pairs_df = pd.DataFrame(output_pairs)
    output_pairs_df = output_pairs_df.iloc[:,:2]
    output_pairs_df.drop_duplicates(subset=['question'], inplace=True)
    output_pairs_df = output_pairs_df.dropna()
    print("Shape:", output_pairs_df.shape)
    print(output_pairs_df.head())
    filename = os.path.join(OUTPUT_FOLDER, "instruct-dataset.csv")
    output_pairs_df.to_csv(filename, index=False)

    # Build training formats
    output_pairs_df['text'] = "human: " + output_pairs_df['question'] + "\n" + "bot: " + output_pairs_df['answer']
    
    # Gemini Data prep: https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning-prepare
    # {"contents":[{"role":"user","parts":[{"text":"..."}]},{"role":"model","parts":[{"text":"..."}]}]}
    output_pairs_df["contents"] = output_pairs_df.apply(lambda row: [{"role":"user","parts":[{"text": row["question"]}]},{"role":"model","parts":[{"text": row["answer"]}]}], axis=1)


    # Test train split
    df_train, df_test = train_test_split(output_pairs_df, test_size=0.1, random_state=42)
    df_train[["text"]].to_csv(os.path.join(OUTPUT_FOLDER, "train.csv"), index = False)
    df_test[["text"]].to_csv(os.path.join(OUTPUT_FOLDER, "test.csv"), index = False)

    # Gemini : Max numbers of examples in validation dataset: 256
    df_test = df_test[:256]

    # JSONL
    with open(os.path.join(OUTPUT_FOLDER, "train.jsonl"), "w") as json_file:
        json_file.write(df_train[["contents"]].to_json(orient='records', lines=True))
    with open(os.path.join(OUTPUT_FOLDER, "test.jsonl"), "w") as json_file:
        json_file.write(df_test[["contents"]].to_json(orient='records', lines=True))


def upload():
    print("upload()")

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    timeout = 300

    data_files = glob.glob(os.path.join(OUTPUT_FOLDER, "*.jsonl")) + glob.glob(os.path.join(OUTPUT_FOLDER, "*.csv"))
    data_files.sort()
    
    # Upload
    for index, data_file in enumerate(data_files):
        filename = os.path.basename(data_file)
        destination_blob_name = os.path.join("llm-finetune-dataset-small", filename)
        blob = bucket.blob(destination_blob_name)
        print("Uploading file:", data_file, destination_blob_name)
        blob.upload_from_filename(data_file, timeout=timeout)
    

def main(args=None):
    print("CLI Arguments:", args)

    if args.generate:
        generate()

    if args.prepare:
        prepare()
      
    if args.upload:
        upload()


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal '--help', it will provide the description
    parser = argparse.ArgumentParser(description="CLI")

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate data",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare data",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload data to bucket",
    )

    args = parser.parse_args()

    main(args)