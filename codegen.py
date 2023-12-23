import requests
import pickle
import json

run_capybara = "/home/pc/Desktop/llamacpp/gguf/mixtral/server -m /home/pc/Desktop/llamacpp/gguf/models/nous-capybara-34b.Q4_K_S.gguf -ngl 5000"
run_nous13b = "/home/pc/Desktop/llamacpp/gguf/gpu/server -m nous-hermes-llama2-13b.Q4_K_S.gguf -ngl 500"
run_openhermes7b = "/home/pc/Desktop/llamacpp/gguf/gpu/server -m openhermes-2.5-mistral-7b.Q3_K_M.gguf -ngl 500 --port 9080"

run_mixtral = ""

def has_python_code(code):
    return "```python" in code

def extract_python(code):
    loc = code.split("\n")
    python_code = ""
    inside_code = False
    for line in loc:
        if "```python" in line and not inside_code:
            inside_code = True
        elif "```" in line and inside_code:
            inside_code = False
            return python_code
        elif inside_code:
            python_code += line + "\n"
    return python_code

def dataset_leaks():
    for ins in enumerate(instructs):
        for llm in llms:
            code = llamacpp_generate(llm, ins)
            print(ins)
            print(code)
            print("================================")
        print("***************************************")

def llamacpp_generate(url, prompt, temp=0.8):
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt, "temperature": temp}
    data = json.dumps(data)
    res = requests.post(url=url, data=data, headers=headers)
    return res.json()["content"]

def load_pickle(objfile):
    with open(objfile, 'rb') as f:
        res = pickle.load(f)
    return res

metaprompt = "You are a skilled python developer. You write python code in between ```python and ```; so that, your colleagues can use your code properly."
metaprompt = "You are a skilled python developer. You write python code in between ```python and ```; the code you write is directly sent to a python interpreter."
metaprompt = """You are a skilled python developer. 
            You write python code in between ```python and ``` blocks that solve the instructions given by the user.
            An example of it is the following:
            ```python
            n = 10
            numbers = []
            i = 0
            while i < n:
                numbers.append(i)
                i += 1
            print(numbers)
            ```
            """

openhermes_neural = "http://localhost:19080/completion"

#nous_hermes_13b = "http://localhost:8080/completion"
openhermes_mistral_7b = "http://localhost:9080/completion"

llms = [openhermes_neural, openhermes_mistral_7b]
llms_dict = [
    {
        "model": "openhermes_neural",
        "url": openhermes_neural,
        "gen": []
    },
    {
        "model": "openhermes_mistral_7b",
        "url": openhermes_mistral_7b,
        "gen": []
    }
]
instructs = load_pickle("./objects/instructions.pkl")
prompts = load_pickle("./objects/prompts.pkl")
generated_code = []

has_py_accuracy = [0,0]
num_exp = 10
temperatures = [0.8, 0.7, 0.6]
for ins in enumerate(instructs[:num_exp]):
    for llm_id, llm in enumerate(llms_dict):
        for temp in temperatures:
            prompt = metaprompt + "\n" + ins[1]
            code = llamacpp_generate(llm["url"], prompt,  temp)
            print(ins[1])
            if has_python_code(code):
                has_py_accuracy[llm_id] += 1
            code = extract_python(code)
            llm["gen"].append({"instruct": ins[1], "code":code, "temp": temp})
            print(code)
            print("================================")
        print("/////////////////////////////////////////////////")
    print("***************************************")

for i, elem in enumerate(has_py_accuracy):
    has_py_accuracy[i] /= num_exp

print(has_py_accuracy)

print(llms_dict)