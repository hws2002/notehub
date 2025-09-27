import json
import re

from llm import llm_client
from preprocess_input import stream_conversations

# --- Constants ---
INPUT_PATH = "input_data/conversations.json"
OUTPUT_PATH = "graph_data/graph_data.json"

CONNECTION_THRESHOLD = 0.5


def extract_json_from_response(response: str) -> dict | None:
    """Extracts a JSON object from a string, even if it's embedded in markdown."""
    # Use a more robust regex to find the JSON block, allowing for optional 'json' and whitespace.
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # If no markdown block, find the first '{' or '[' and the last '}' or ']'
        # This is a bit more robust than assuming the whole string is JSON.
        start_brace = response.find("{")
        start_bracket = response.find("[")

        if start_brace == -1 and start_bracket == -1:
            print(f"❌ ERROR: No JSON object or array found in the response.")
            return None

        start = min(s for s in [start_brace, start_bracket] if s != -1)
        end = max(response.rfind("}"), response.rfind("]")) + 1
        json_str = response[start:end]

    try:
        return json.loads(json_str.strip())
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Could not decode the extracted JSON string: {json_str}")
        return None


def _get_conversation_text(mapping: dict) -> str:
    """
    Extracts and concatenates the text from a conversation mapping,
    creating a simple, linear transcript.
    """
    if not mapping:
        return ""

    # Find the root message (the one with no parent)
    root_id = None
    for msg_id, msg_data in mapping.items():
        if msg_data.get("parent") is None:
            root_id = msg_id
            break

    transcript = []
    current_id = root_id
    while current_id in mapping:
        message_content = (
            mapping[current_id]
            .get("message", {})
            .get("content", {})
            .get("parts", [""])[0]
        )
        transcript.append(message_content)
        current_id = mapping[current_id].get("children", [None])[0]

    return "\n".join(transcript)


def create_graph_from_data(connection_threshold: float = 0.5):
    """
    Reads conversation data, generates a graph using an LLM, and saves it to a file.
    """
    # Construct absolute path for input data
    input_file_path = INPUT_PATH
    output_file_path = OUTPUT_PATH

    # Create a graph from the data
    graph = {"nodes": [], "links": []}
    conversations_data = []

    # --- Step 1: Stream conversations and create nodes ---
    # This step does not require an LLM. We create nodes directly from the input data.
    print("--- Streaming conversations and creating nodes ---")
    try:
        for i, conversation in enumerate(stream_conversations(input_file_path)):
            if "title" not in conversation:
                print(
                    f"⚠️ WARNING: Skipping conversation {i} due to missing 'title' key."
                )
                continue

            # Create a node for each conversation
            graph["nodes"].append(
                {
                    "id": f"conv_{i}",
                    "label": conversation["title"],
                    "type": "conversation",
                }
            )
            # Store the title and a lean version of the content for the next step
            conversations_data.append(
                {
                    "id": i,
                    "title": conversation["title"],
                    "content": _get_conversation_text(conversation.get("mapping", {})),
                }
            )
            print(f"    Processed node for conversation {i+1}", end="\r")

    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found at {input_file_path}")
        return
    print("\n--- Finished extracting topics ---")

    # Then, generate links between conversations
    print("\n--- Generating links between conversations using LLM ---")
    # Create a list of topic indices to iterate over
    total_pairs = len(conversations_data) * (len(conversations_data) - 1) // 2
    processed_pairs = 0

    for idx1 in range(len(conversations_data)):
        for idx2 in range(idx1 + 1, len(conversations_data)):
            processed_pairs += 1
            conv1 = conversations_data[idx1]
            conv2 = conversations_data[idx2]

            prompt = f"""
            You will be given two conversations, each with a title and its content.
            Your task is to determine the strength of the connection between them on a scale of 0.0 to 1.0.
            A score of 1.0 means they are about the same topic. 0.0 means they are completely unrelated.
            Base your judgment primarily on the titles. Only refer to the content if the titles are too generic or ambiguous.

            Conversation 1 Title: "{conv1['title']}"
            Conversation 1 Content: "{conv1['content'][:500]}..."

            Conversation 2 Title: "{conv2['title']}"
            Conversation 2 Content: "{conv2['content'][:500]}..."

            Return ONLY a single floating-point number and nothing else.
            """

            response = llm_client.request_llm_output(
                system_prompt="You are an AI assistant that determines the connection strength between conversation topics.",
                user_prompt=prompt,
                temperature=0.0,  # Set to 0 for deterministic, numerical output
                max_tokens=10,  # The response should only be a number
                presence_penalty=0.0,  # Not needed for single number output
            )

            if response:
                try:
                    # Clean up response to ensure it's just a number
                    cleaned_response = re.sub(r"[^0-9.]", "", response)
                    if cleaned_response:
                        connection_strength = float(cleaned_response)
                        if connection_strength > connection_threshold:
                            graph["links"].append(
                                {
                                    "source": f"conv_{conv1['id']}",
                                    "target": f"conv_{conv2['id']}",
                                    "strength": connection_strength,
                                }
                            )
                except (ValueError, TypeError):
                    print(
                        f"❌ ERROR: Could not parse connection strength as a float from response: '{response}'"
                    )
            print(f"    Processed pair {processed_pairs}/{total_pairs}", end="\r")

    print("\n--- Finished generating links ---")

    # Save the graph to a file
    with open(output_file_path, "w") as f:
        json.dump(graph, f, indent=2)

    print("\n--- Graph Generation Finished ---")
    print(f"Graph data saved to {output_file_path}")


if __name__ == "__main__":
    print("--- Starting Graph Generation Process ---")
    # Using a threshold of 0.4 to capture moderately strong connections
    create_graph_from_data(CONNECTION_THRESHOLD)
