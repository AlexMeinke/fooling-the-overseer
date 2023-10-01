import re


def extract_response_content(response):
    # Locate the phrase "scoring system:" in the response
    index = response.find("scoring system:")

    # If "scoring system:" is found, extract the relevant content; otherwise, use the entire response
    text = response[index + len("scoring system:") :] if index != -1 else response

    # Remove starting and ending white-spaces, newlines, or quotations
    text = text.strip(' \n"')

    # Eliminate the trailing "</s>"
    if text.endswith("</s>"):
        text = text[:-4]

    # Remove any remaining trailing and starting quotation marks
    text = text.rstrip('"').lstrip('"')

    return text


def extract_last_number(text):
    if not isinstance(text, str):
        raise ValueError("Input should be a string")

    numbers = re.findall(r"\d+", text)
    return int(numbers[-1]) if numbers else None


def format_prompt(user_prompt, system_prompt=None):
    if system_prompt is None:
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST] "
