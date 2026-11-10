import requests
import json

def call_ollama_model(prompt, model, stream=False):
    url='http://localhost:11434/api/generate'

    params = {
        'prompt':prompt,
        'model':model,
        'stream':stream
    }
    payload = json.dumps(params)

    response = requests.post(url, data=payload,
                             headers={'Content-Type':'application/json'})
    if response.status_code != 200:
        return f'{response.text}'
    else:
        response = response.json()['response']
        return response

######################################
if __name__ == '__main__':
    model = 'gemma3:1b'
    prompt = input('Enter your Question: ')
    response = call_ollama_model(prompt, model)
    print(f'result:\n {response}')
