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
    gold_dict = parse_generation(str(answer[0]))
    predicted_dicts = [parse_generation(c) for c in completions]
#    print("GOLD: ", gold_dict)
#    print("PRED: ", predicted_dicts)
    gold_word_guesses = gold_dict["word_guesses"].lower().split(";")

    scores = []

    for pred in predicted_dicts:
        pred_word_guesses = pred["word_guesses"].lower().split(";")    
    
        pwd_score = 0
        for pw, gw in zip(pred_word_guesses, gold_word_guesses):
            if pw == gw:
                pwd_score += 2
            elif len(pw) == len(gw):
                pwd_score += 0.2
            else:
                pwd_score -= 1
        scores.append(pwd_score)
#     print("word guesses: ", scores)
    return scores

def check_first_pass(prompts, completions, answer, **kwargs):
    completions = [completions[i][0]['content'] for i in range(len(completions))]
    gold_dict = parse_generation(str(answer[0]))
    predicted_dicts = [parse_generation(c) for c in completions]
    
    gold_first_pass = split_words_and_letters(gold_dict["first_pass"].split(" "))
    
    scores = []

    for pred in predicted_dicts:
        pred_word_guesses = pred["word_guesses"].lower().split(";")
        pred_first_pass = split_words_and_letters(pred["first_pass"].split(" "))
                
        cfp_score = 0

        for pw, pfp, gfp in zip(pred_word_guesses, pred_first_pass["words"], gold_first_pass["words"]):
            if pw == pfp == gfp:
                cfp_score += 1
            elif pw != pfp and pfp == gfp:
                cfp_score += 0.1
            else:
                cfp_score -= 2
        
        scores.append(cfp_score)
#    print("first pass: ", scores)
    return scores

def check_solution_words(prompts, completions, answer, **kwargs):
    completions = [completions[i][0]['content'] for i in range(len(completions))]
    gold_dict = parse_generation(str(answer[0]))
    predicted_dicts = [parse_generation(c) for c in completions]
    gold_solution_words = gold_dict["solution_words"].lower().split(";")

    scores = []

    for pred in predicted_dicts:
        pred_solution_words = pred["solution_words"].lower().split(";")
        
        score = 0

        if len(gold_solution_words) == len(pred_solution_words):
            score += 1
        else:
            score -= 3 * np.abs(len(gold_solution_words) - len(pred_solution_words))
        
        for pw, gw in zip(pred_solution_words, gold_solution_words):
            if pw == gw:
                score += 1
            elif len(pw) == len(gw):
                score += 0.2
            else:
                score -= np.abs(len(pw) - len(gw))

        scores.append(score)
#    print("solution words: ", scores)
    return scores


def check_solution(prompts, completions, answer, **kwargs):
    completions = [completions[i][0]['content'] for i in range(len(completions))]
    gold_dict = parse_generation(str(answer[0]))
    predicted_dicts = [parse_generation(c) for c in completions]
    gold_solution = gold_dict["solution"].lower()

    scores = []

    for pred in predicted_dicts:
        score = 0
        pred_solution = pred["solution"].lower()

        if pred_solution == gold_solution:
            score += 5
        else:
            score -= 5

        scores.append(score)
#    print("solution: ", scores)
    return scores
