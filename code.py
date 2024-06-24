import whisper
from openai import OpenAI


# Function to query OpenAI GPT-4 for validation
def validate_text_with_gpt4(transcribed_text):
    api_key = ""
    client = OpenAI(api_key=api_key)

    input = transcribed_text + ". Answer with references."

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user", "content": f"{input}"
            },
        ]
    )

    return completion.choices[0].message.content


def main():
    # Load the Whisper model
    model = whisper.load_model("base")

    # Path to the input audio file
    input_audio_path = "tiktok.wav"

    # Transcribe the audio file
    result = model.transcribe(input_audio_path)

    # Print the transcribed text
    transcribed_text = result["text"]

    # Validate the transcribed text using GPT-4
    validation_result = validate_text_with_gpt4(transcribed_text)
    print("Response:")
    print(validation_result)


if __name__ == "__main__":
    main()

    # Extract information
    # Query
    # Find Sources
    # Check if Sources are Reliable
    # Use APIs to check Domain Authority
    # Decide with Confidence
