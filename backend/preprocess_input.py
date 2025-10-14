import json
from llm import llm_client

MAX_PROMPT_TOKENS = 30000  # Max tokens for the prompt to avoid API errors

def _summarize_large_conversation(conversation_content: str) -> str:
    """
    Summarizes a large conversation by splitting it into chunks and summarizing each chunk.
    """
    chunks = [
        conversation_content[j : j + MAX_PROMPT_TOKENS]
        for j in range(0, len(conversation_content), MAX_PROMPT_TOKENS)
    ]
    chunk_summaries = []

    print(f"\n    Conversation is too long, splitting into {len(chunks)} chunks.")

    for chunk_idx, chunk in enumerate(chunks):
        prompt = f'''
        This is part of a larger conversation.
        Conversation Part:
        {chunk}

        Summarize this part of the conversation.
        '''
        summary_part = llm_client.request_llm_output(
            system_prompt="You are an AI assistant that summarizes parts of conversations.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=1024,
        )
        if summary_part:
            chunk_summaries.append(summary_part)
        print(f"        - Summarized chunk {chunk_idx + 1}/{len(chunks)}", end="\r")
    
    print("\n        - Combining summaries...")
    combined_summary = "\n\n".join(chunk_summaries)

    # Final summary of the combined summaries
    prompt = f'''
    The following are summaries of different parts of a long conversation.
    Combine them into a single, concise summary of the entire conversation's main subject.

    Summaries:
    {combined_summary}
    '''
    final_summary = llm_client.request_llm_output(
        system_prompt="You are an AI assistant that creates a final summary from partial summaries.",
        user_prompt=prompt,
        temperature=0.3,
        max_tokens=1024,
    )
    return final_summary if final_summary else ""


def stream_conversations(file_path):
    """
    Reads a large JSON file containing an array of conversations and yields
    one conversation at a time. This avoids loading the entire file into memory.
    If a conversation is too large, it is summarized before being yielded.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            initial_chunk = ''
            # Find the start of the array and process the remainder of that line
            for line in f:
                if '[' in line:
                    initial_chunk = line[line.find('[') + 1:]
                    break
            
            # Create a single stream of the first line's remainder and the rest of the file
            content_stream = initial_chunk + f.read()

            buffer = ''
            brace_count = 0
            in_string = False

            for char in content_stream:
                if char == '"' and (not buffer or buffer[-1] != '\\'):
                    in_string = not in_string

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                
                if brace_count > 0:
                    buffer += char
                elif brace_count == 0 and char == '}' and buffer:
                    # We have a full object string, which started with a '{'
                    buffer += char
                    try:
                        conversation = json.loads(buffer)
                        
                        if "mapping" in conversation:
                            conversation_content = json.dumps(conversation["mapping"], indent=2)
                            if len(conversation_content) > MAX_PROMPT_TOKENS:
                                summary = _summarize_large_conversation(conversation_content)
                                conversation["mapping"] = {"summary": summary}

                        yield conversation

                    except json.JSONDecodeError as e:
                        # This might happen if string contains braces; the logic is not a perfect parser.
                        # We will simply skip the object and reset.
                        print(f"⚠️ WARNING: Skipping an object due to JSON decode error: {e}")
                    buffer = '' # Reset for the next object

    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found at {file_path}")