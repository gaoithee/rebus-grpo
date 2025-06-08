import re

regex_firstpass = 'Prima lettura: (.*)'
regex_solution_word = "\d+ = (.*)"
regex_solution = "Soluzione: (.*)"

def parse_generation(data):
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
        "first_pass": first_pass,
        "solution_words": solution_words,
        "solution": solution,
    }


import numpy as np

def check_chiave_risolutiva(prompts, completions, answer, **kwargs):
    compl = [completions[i][0]['content'] for i in range(len(completions))]
    gold_dicts = [parse_generation(str(a)) for a in answer]
    predicted_dicts = [parse_generation(c) for c in compl]
#    print("GOLD: ", gold_dicts)
#    print("PRED: ", predicted_dicts)
    scores = []

    for gold, pred in zip(gold_dicts, predicted_dicts):
        gold_solution_words = gold["solution_words"].lower().split(";")
        pred_solution_words = pred["solution_words"].lower().split(";") 

        score = 0
        if pred_solution_words == '' or len(pred_solution_words) == 0:
            scores.append(-10)
            continue
        
        # questo compara il numero di parole predette con il numero di parole effettive
        if len(gold_solution_words) == len(pred_solution_words):
          score += 3
        else:
          score -= 3 * int(np.abs(len(gold_solution_words) - len(pred_solution_words)))

        # questo guarda per ogni coppia di parole predette ed effettive la lunghezza
        # se concorda, bene, altrimenti cazzi
        for pw, gw in zip(pred_solution_words, gold_solution_words):
            if pw == gw:
                score += 3
            elif len(pw) == len(gw):
                score += 0.2
            else:
                score -= int(np.abs(len(pw) - len(gw)))
        scores.append(score)
    print("Check chiave: ", scores)
    return scores


def check_prima_lettura(prompts, completions, answer, **kwargs):
    compl = [completions[i][0]['content'] for i in range(len(completions))]
    gold_dicts = [parse_generation(str(a)) for a in answer]
    predicted_dicts = [parse_generation(c) for c in compl]
    
    scores = []

    for gold, pred in zip(gold_dicts, predicted_dicts):
        gold_primalet = gold["first_pass"].lower()
        pred_primalet = pred["first_pass"].lower()
        if gold_primalet != pred_primalet:
            scores.append(-10)
            continue
        scores.append(0)
    print("Prima Lettura: ", scores)
    return scores


def check_soluzione(prompts, completions, answer, **kwargs):
    compl = [completions[i][0]['content'] for i in range(len(completions))]
    gold_dicts = [parse_generation(str(a)) for a in answer]
    predicted_dicts = [parse_generation(c) for c in compl]

    scores = []

    for gold, pred in zip(gold_dicts, predicted_dicts):
        gold_solution = gold["solution"].lower()
        pred_solution = pred["solution"].lower()
        if gold_solution != pred_solution:
            scores.append(-5)
            continue
        scores.append(0)
    print("Soluzione: ", scores)
    return scores
