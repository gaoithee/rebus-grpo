import re
import numpy as np

regex_word_guess = '- \[.* = (.*)'
regex_firstpass = 'Prima lettura: (.*)'
regex_solution_word = "\d+ = (.*)"
regex_solution = "Soluzione: (.*)"

def parse_generation(data):
    try:
        word_guesses = ";".join(re.findall(regex_word_guess, data))
    except:
        word_guesses = ""
    try:
        first_pass = re.findall(regex_firstpass, data)[0]
    except:
        first_pass = ""
    try:
        solution_words = ";".join(re.findall(regex_solution_word, data))
    except:
        solution_words = ""
    try:
        solution = re.findall(regex_solution, data)[0]
    except:
        solution = ""
    
    return {
        "word_guesses": word_guesses,
        "first_pass": first_pass,
        "solution_words": solution_words,
        "solution": solution,
    }


def split_words_and_letters(tokens):
    result = {
        "words": [],
        "letters": []
    }

    for token in tokens:
        if token.isupper():
            result["letters"].append(token)
        else:
            result["words"].append(token)

    return result
    

def check_word_guesses(prompts, completions, answer, **kwargs):
    completions = [completions[i][0]['content'] for i in range(len(completions))]
    gold_dicts = [parse_generation(str(a)) for a in answer]
    predicted_dicts = [parse_generation(c) for c in completions]
    print("GOLD: ", gold_dicts)
    print("PRED: ", predicted_dicts)
    scores = []
    for gold, pred in zip(gold_dicts, predicted_dicts):
        gold_word_guesses = gold["word_guesses"].lower().split(";")
        pred_word_guesses = pred["word_guesses"].lower().split(";")    
        if pred_word_guesses == '' or len(pred_word_guesses) == 0:
            scores.append(-10)
            continue
        pwd_score = 0
        for pw, gw in zip(pred_word_guesses, gold_word_guesses):
            if pw == gw:
                pwd_score += 2
            elif len(pw) == len(gw):
                pwd_score += 0.2
            else:
                pwd_score -= 1
        scores.append(pwd_score)
    print("word guesses: ", scores)
    return scores

def check_first_pass(prompts, completions, answer, **kwargs):
    completions = [completions[i][0]['content'] for i in range(len(completions))]
    gold_dicts = [parse_generation(str(a)) for a in answer]
    predicted_dicts = [parse_generation(c) for c in completions]
    gold_first_passes = [split_words_and_letters(d["first_pass"].split(" ")) for d in gold_dicts]
    pred_first_passes = [split_words_and_letters(d["first_pass"].split(" ")) for d in predicted_dicts]
    
    scores = []
    for gold, pred, gold_fp, pred_fp in zip(gold_dicts, predicted_dicts, gold_first_passes, pred_first_passes):
        if pred['first_pass'] == '' or len(pred['first_pass']) == 0:
            scores.append(-10)
            continue
        pred_word_guesses = pred["word_guesses"].lower().split(";")
        cfp_score = 0

        for pw, pfp, gfp in zip(pred_word_guesses, pred_fp["words"], gold_fp["words"]):
            if pw == pfp == gfp:
                cfp_score += 1
            elif pw != pfp and pfp == gfp:
                cfp_score += 0.1
            else:
                cfp_score -= 2
        scores.append(cfp_score)
    print("first pass: ", scores)
    return scores

def check_solution_words(prompts, completions, answer, **kwargs):
    completions = [completions[i][0]['content'] for i in range(len(completions))]
    gold_dicts = [parse_generation(str(a)) for a in answer]
    predicted_dicts = [parse_generation(c) for c in completions]
    gold_solution_words = [gold["solution_words"].lower().split(";") for gold in gold_dicts]
    pred_solution_words = [pred["solution_words"].lower().split(";") for pred in predicted_dicts]

    scores = []

    for gold_sw, pred_sw in zip(gold_solution_words, pred_solution_words):        
        score = 0

        if len(pred_sw) == 0:
            scores.append(-10)
            continue

        if len(gold_sw) == len(pred_sw):
            score += 1
        else:
            score -= 3 * int(np.abs(len(gold_sw) - len(pred_sw)))
        
        for pw, gw in zip(pred_sw, gold_sw):
            if pw == gw:
                score += 1
            elif len(pw) == len(gw):
                score += 0.2
            else:
                score -= int(np.abs(len(pw) - len(gw)))

        scores.append(score)
    print("solution words: ", scores)
    return scores


def check_solution(prompts, completions, answer, **kwargs):
    completions = [completions[i][0]['content'] for i in range(len(completions))]
    gold_dicts = [parse_generation(str(a)) for a in answer]
    predicted_dicts = [parse_generation(c) for c in completions]
    gold_solutions = [d["solution"].lower() for d in gold_dicts]
    pred_solutions = [d["solution"].lower() for d in predicted_dicts]

    scores = []

    for gs, ps in zip(gold_solutions, pred_solutions):
        score = 0

        if ps=='':
           scores.append(-10)
           continue

        if ps == gs:
            score += 10
        else:
            score -= 5

        scores.append(score)
    print("solution: ", scores)
    return scores
