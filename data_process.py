import sys
import re
import requests
import pandas as pd
from io import StringIO
from contextlib import redirect_stdout
import pickle

skip_samples = ["HAS-USER-INPUT","ERROR-DURING-EVAL"]

def has_flask(code):
    return "flask" in code

def has_google(code):
    return "google.com" in code

def has_requests(code):
    return "requests" in code

def has_turtle(code):
    return "turtle" in code

def has_bignumber(code, numlen):
    numlist = re.findall(r'\d+', code)
    for number in numlist:
        if len(str(number)) > numlen:
            return True
        
    return False

def has_nested_for(code, nexp):
    loc = code.split("\n")
    max_comp = 0
    for line in loc:
        if "for" in line:
            spaces = len(line.split("for")[0])
            if spaces > max_comp:
                max_comp += 1

    return max_comp > nexp

def has_ml(code):
    return "sklearn" in code

def has_smtp(code):
    return "smtp" in code

def has_server(code):
    return "serve_forever" in code

def has_tk(code):
    return "tkinter" in code

def has_while_true(code):
    return "while True" in code

def has_url(code):
    return "example.com" in code

def has_pygame(code):
    return "pygame" in code

def has_plots(code):
    return ".show()" in code

def has_code_function(code):
    return "def" in code

def has_user_input(code):
    return "input(" in code

def skip_plots(code):
    res = ""
    loc = code.split("\n")
    for line in loc:
        if has_plots(line):
            continue
        else:
            res += line + "\n"
    return res

def wrap_code_def(code):
    res = "def func_wrapper():\n"
    loc = code.split("\n")
    for i, line in enumerate(loc):
        if i == len(loc)-1:
            res += "\t"+"return "+line + "\n"
        else:
            res += "\t"+line + "\n"
    return res

def get_py_function(code):
    loc = code.split("\n")
    res = ""
    for line in loc:
        if "def " in line:
            res = line.split("def ")[-1].split("(")[0]
            break

    return res
    
def exec_and_return(code_to_run):
    loc = {}
    exec(code_to_run, globals(), loc)
    return_workaround = loc['function_return']
    #print(return_workaround)
    return return_workaround

def get_code_output(code, func_params):
    if not has_code_function(code):
        code = wrap_code_def(code)
    if has_plots(code):
        code = skip_plots(code)
    func_name = get_py_function(code)
    code_to_run = code+"\n"+"function_return = "+func_name+"("+func_params+")"
    print(code_to_run)
    res = exec_and_return(code_to_run)
    print(res)
    return res
    
def get_evaluation(code, func_params, output):
    obtained_out = get_code_output(code, func_params)
    return obtained_out == output

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(list(obj), outp, pickle.HIGHEST_PROTOCOL)

sample_size = 7500

data = pd.read_parquet("./data/train-00000-of-00001-8b6e212f3e1ece96.parquet")
sample = data.iloc[:sample_size]

instructions = sample["instruction"].tolist()
prompt = sample["prompt"].tolist()
code_params = sample["input"].tolist()
code_truth = sample["output"].tolist()

cleaned_instructions = []
cleaned_prompt = []
cleaned_code_params = []
cleaned_code_truth = []
cleaned_code_output = []

for i, code_elem in enumerate(code_truth):
    if has_flask(code_elem) or has_google(code_elem) or has_requests(code_elem) or has_bignumber(code_elem, 4) or has_turtle(code_elem) or has_ml(code_elem) or has_smtp(code_elem) or has_nested_for(code_elem, 1) or has_server(code_elem) or has_user_input(code_elem) or has_pygame(code_elem) or has_url(code_elem) or has_tk(code_elem) or has_while_true(code_elem):
        #code_output.append("HAS-USER-INPUT")
        continue
    try:
        returned_val = get_code_output(code_elem, code_params[i])

        cleaned_instructions.append(instructions[i])
        cleaned_prompt.append(prompt[i])
        cleaned_code_params.append(code_params[i])
        cleaned_code_truth.append(code_truth[i])
        cleaned_code_output.append(returned_val)

    except:
        print("FAILED CODE.")
        #code_output.append("ERROR-DURING-EVAL")
    print("##########################################")

print(cleaned_code_truth)
print("Working samples", len(cleaned_code_truth))

data_dict = {}
data_dict["instructions"] = cleaned_instructions
data_dict["prompt"] = cleaned_prompt
data_dict["params"] = cleaned_code_params
data_dict["code"] = cleaned_code_truth
data_dict["output"] = cleaned_code_output

df_cleaned = pd.DataFrame(data_dict)

#with open('./objects/df.pkl', 'wb') as f:
#    pickle.dump(df_cleaned, f, pickle.HIGHEST_PROTOCOL)

save_object(data_dict["instructions"], "./objects/instructions.pkl")
save_object(data_dict["prompt"], "./objects/prompts.pkl")
#save_object(df_cleaned, "./objects/df_data.pkl")
#df_cleaned.to_csv("cleaned_data/train-00000-of-00001-8b6e212f3e1ece96.csv")



#ev = get_evaluation(code_res[0], code_params[0], 15)
#print(ev)

"""
print(instructions[0])
print("#####################")
print(prompt[0])
print("#####################")
print(code_params[0])
print("#####################")
print(code_res[0])
"""
