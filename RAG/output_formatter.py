import difflib
import re

def csv_output_formatter(output):

    code_match = re.search(r"```([\s\S]*?)```", output)
    if code_match:
        output = code_match.group(1)
    python_match = re.search(r"python\s*(.+)", output, re.DOTALL)
    if python_match:
        output = python_match.group(1)
    print_match = re.search(r'print\((.*?)\)', output)
    if not print_match:
        output = f"print({output})"

    return output

def eval_output_formatter(output):

    result_dict = {
        "Analysis": "",
        "Correctness": "",
        "Relevance": "",
        "Coherence": "",
    }

    for key in result_dict.keys():
        match = None
        if key != "Analysis":
            key_match = re.search(fr"{key}: (\d+)", output)
            if key_match:
                match = int(key_match.group(1))
        else:
            key_match = re.search(r"Analysis:\s*(.*?)\s*Correctness:", output, re.DOTALL)
            if key_match:
                match = key_match.group(1)
        result_dict[key] = match

    return result_dict

def strip_all(text):
   return "\n".join([line.strip() for line in text.splitlines()])

def find_best_substring_match(str1, str2):

    if len(str1) == len(str2):
        print("Strings have the same length, one string must be longer than the other!")
        return None
    
    if len(str1) > len(str2):
        db_str = str1
        query_str = str2
    else:
        db_str = str2
        query_str = str1
    
    n = len(query_str)
    best_ratio = 0
    best_match = ""
    for j in range(len(db_str)-n+1):
        substring = db_str[j:j+n]
        ratio = difflib.SequenceMatcher(None, query_str, substring).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = substring

    return best_ratio, best_match