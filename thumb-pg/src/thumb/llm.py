import os
import json
import predictionguard as pg
from tqdm.auto import tqdm
import time
import asyncio
from langchain.callbacks.openai_info import standardize_model_name, MODEL_COST_PER_1K_TOKENS, get_openai_token_cost_for_model
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage

# Set up the Prediction Guard access token
pg_access_token = 'your_prediction_guard_access_token'
os.environ['PREDICTIONGUARD_TOKEN'] = 'xQZQCxe46UmYwYhjE6C0anJXec8bccK2mOXzwyvj'

# Replace OpenAI's ChatOpenAI class with Prediction Guard's Chat class
class ChatPredictionGuard:
    def __init__(self, model='Notus-7B'):
        self.model = model

    def generate(self, messages):
        return pg.Chat.create(model=self.model, messages=messages)

# Function to format chat prompts for Prediction Guard
def format_chat_prompt(messages, test_case=None):
    formatted_messages = []
    # Assuming SystemMessage, HumanMessage, AIMessage have 'content' attribute
    for message in messages:
        if isinstance(message, SystemMessage):
            formatted_messages.append({"role": "system", "content": message.content})
        elif isinstance(message, HumanMessage):
            formatted_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            formatted_messages.append({"role": "assistant", "content": message.content})
        else:
            formatted_messages.append({"role": "user", "content": message})
    return formatted_messages

# Function to generate chat responses using Prediction Guard API
def get_responses(prompt, test_case, model, runs, pid, cid):
    chat = ChatPredictionGuard(model=model)
    formatted_prompt = format_chat_prompt(prompt, test_case)
    responses = []
    
    for _ in tqdm(range(runs)):
        try:
            start_time = time.time()
            resp = chat.generate(formatted_prompt)
            end_time = time.time()
            assistant_message = resp['choices'][0]['message']['content'].strip()
            response_data = {
                "content": assistant_message,
                "latency": end_time - start_time
            }
        except Exception as e:
            response_data = {"content": str(e), "error": True}
        finally:
            responses.append(response_data)
    
    return responses

# Adapt the asynchronous functions for Prediction Guard
async def async_generate(chat, formatted_prompt, temperature=None, tags=[]):
    response_data = {}
    try:
        start_time = time.time()
        resp = await chat.agenerate([formatted_prompt], tags=tags)
        end_time = time.time()
        response_data = {
            "content": resp['choices'][0]['message']['content'].strip(),
            "latency": end_time - start_time
        }
        if temperature is not None:
            response_data["temperature"] = temperature
    except Exception as e:
        response_data = {"content": str(e), "error": True}
    finally:
        return response_data

async def async_get_responses(batch, verbose=False):
    tasks = []
    for item in batch:
        prompt = item.get('prompt', None)
        model = item.get('model', 'Notus-7B')
        temperature = item.get('temperature', None)
        tags = item.get('tags', [])
        chat = ChatPredictionGuard(model=model)
        task = async_generate(chat, format_chat_prompt(prompt), temperature, tags=tags)
        tasks.append(task)
    responses = await asyncio.gather(*tasks)
    if verbose:
        print("Finished gathering batch responses")
    return responses

# Example usage
if __name__ == "__main__":
    example_messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What's the weather like today?")
    ]

    responses = get_responses(example_messages, None, 'Notus-7B', 1, None, None)
    for response in responses:
        print(response)

