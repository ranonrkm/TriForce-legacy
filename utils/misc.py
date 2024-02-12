from sympy import symbols, Eq, solve
from termcolor import colored

def fake2real(fake,gamma=4):
    a = 1+ gamma*fake
    x = symbols('x')
    equation = Eq(x**(gamma+1) - a*x + a - 1, 0)
    solutions = solve(equation, x)
    return solutions[1]


def spec_stream(pred_token_idx, tokenizer, color='blue'):
    decoded_token = tokenizer.decode(
            pred_token_idx,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            # spaces_between_special_tokens=False,
        )

    decoded_token = decoded_token.replace("<0x0A>", "\n")

    print(colored(decoded_token, color), flush=True, end=" ")

def log_csv(file_path, header, entry):
    try:
        with open(file_path, 'r') as f:
            contents = f.read()
    except FileNotFoundError:
        contents = ""

    if not contents:
        with open(file_path, 'a') as f:
            f.write(header)
    
    with open(file_path, 'a') as f:
        f.write(entry)

def print_config(draft, target, prefill, gen_len, gamma, top_k, top_p, temperature, file_path):
    print(colored("####################################### Config #######################################", 'blue'), flush=True)
    print(colored(f"Draft: {draft.config._name_or_path}", 'blue'), flush=True)
    print(colored(f"Target: {target.config._name_or_path}", 'blue'), flush=True)
    print(colored(f"Prefill Length: {prefill}", 'blue'), flush=True)
    print(colored(f"Generation Length: {gen_len}", 'blue'), flush=True)
    print(colored(f"Gamma: {gamma}", 'blue'), flush=True)
    print(colored(f"Sampling Method: top_k = {top_k}, top_p = {top_p}, temperature = {temperature}", 'blue'), flush=True)
    print(colored(f"Log CSV: {file_path}", 'blue'), flush=True)
    print(colored("######################################################################################\n", 'blue'), flush=True)