import re

regex_risposta = r'<risposta>(.*?)</risposta>'
regex_primalettura = r'<primalettura>(.*?)</primalettura>'
regex_parolanascosta = r'<parolanascosta>(.*?)</parolanascosta>'
regex_soluzione = r'<soluzione>(.*?)</soluzione>'

def parse_generation(data):
    try:
        # Estrai tutte le risposte e uniscile con un separatore ";"
        risposte = ";".join(re.findall(regex_risposta, data))
    except:
        risposte = ""
    
    try:
        # Estrai tutte le parole nascoste e uniscile con un separatore ";"
        parole_nascoste = ";".join(re.findall(regex_parolanascosta, data))
    except:
        parole_nascoste = ""
    
    try:
        # Estrai la soluzione
        soluzione = re.findall(regex_soluzione, data)[0]
    except:
        soluzione = ""
    
    try:
        # Estrai la prima lettura
        primalettura = ";".join(re.findall(regex_primalettura, data))
    except:
        primalettura = ""
    
    return {
        "word_guesses": risposte.lower(),
        "solution_words": parole_nascoste.lower(),
        "solution": soluzione.lower(),
        "first_pass": primalettura
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
    

def exact_match_solution_old(prompts, completions, answer, **kwargs):
    predicted_dicts = [parse_generation(completion) for completion in completions]
    gold_dict = parse_generation(answer)

    predicted = [pred['solution'].lower() for pred in predicted_dicts]
    gold = gold_dict['solution'].lower()

    scores = []
    for guess in predicted:
        if guess == "":
            scores.append(0)
            continue
        try:
            scores.append(1.0 if guess == gold else 0.0)
        except:
            scores.append(0)
            continue
    return scores


def perc_correct_words_solution_old(prompts, completions, answer, **kwargs):
    predicted_dicts = [parse_generation(completion) for completion in completions]
    gold_dict = parse_generation(answer)

    predicted = [pred['solution_words'].lower().split(";") for pred in predicted_dicts]
    gold = gold_dict['solution_words'].lower().split(";")
    
    scores = []
    
    for pred in predicted:
        if pred == "":
            scores.append(0)
            continue

        score = 0
        for pw, gw in zip(pred, gold):
            if pw == gw:
                score += 1
            elif len(pw) == len(gw):
                score += 0.5

        scores.append(score / len(gold)) 
        # e se non li normalizzassi?

    return scores


def words_letters_match_primalet_old(prompts, completions, answer, **kwargs):
    predicted_dicts = [parse_generation(completion) for completion in completions]
    gold_dict = parse_generation(answer)
    gold = split_words_and_letters(gold_dict['first_pass'].split(" "))

    scores = []
    for predicted in predicted_dicts:
        pred = split_words_and_letters(predicted['first_pass'].split(" "))

        # --- Word Score ---
        wscore = 0
        for pw, gw in zip(pred['words'], gold['words']):
            if pw == gw:
                wscore += 1
            elif len(pw) == len(gw):  # Se le lunghezze delle parole sono uguali
                wscore += 0.5

        word_score = wscore / len(gold['words']) if gold['words'] else 0

        # --- Letter Score ---
        lscore = 0
        for pw, gw in zip(pred['letters'], gold['letters']):
            if pw.lower().replace(" ", "") == gw.lower().replace(" ", ""):
                lscore += 1
        
        letter_score = lscore / len(gold['letters']) if gold['letters'] else 0

        scores.append(word_score + letter_score)

    return scores


def perc_correct_words_defres_old(prompts, completions, answer, **kwargs):
    predicted_dicts = [parse_generation(completion) for completion in completions]
    gold_dict = parse_generation(answer)

    predicted = [pred['word_guesses'].lower().split(";") for pred in predicted_dicts]
    gold = gold_dict['word_guesses'].lower().split(";")

    scores = []

    for pred in predicted:
        pred = [word for word in pred]
        
        score = 0
        for pw, gw in zip(pred, gold):
            if pw == gw:
                score += 1
            elif len(pw) == len(gw):  
                score += 0.5
        
        scores.append(score / len(gold))
    
    return scores


def exact_match_solution(prompts, completions, answer, **kwargs):
    gold_solution = parse_generation(answer)["solution"].lower()
    predicted_dicts = [parse_generation(c) for c in completions]

    scores = []
    for pred in predicted_dicts:
        pred_solution = pred["solution"].lower()
        score = 1.0 if pred_solution == gold_solution and pred_solution != "" else 0.0
        scores.append(score)
    return scores


def perc_correct_words_solution(prompts, completions, answer, **kwargs):
    gold_solution_words = parse_generation(answer)["solution_words"].lower().split(";")
    predicted_dicts = [parse_generation(c) for c in completions]

    scores = []
    for pred in predicted_dicts:
        pred_words = pred["solution_words"].lower().split(";")
        score = 0
        for pw, gw in zip(pred_words, gold_solution_words):
            if pw == gw:
                score += 1
            elif len(pw) == len(gw):
                score += 0.5
        score /= len(gold_solution_words) if gold_solution_words else 1
        scores.append(score)
    return scores


def perc_correct_words_defres(prompts, completions, answer, **kwargs):
    gold_word_guesses = parse_generation(answer)["word_guesses"].lower().split(";")
    predicted_dicts = [parse_generation(c) for c in completions]

    scores = []
    for pred in predicted_dicts:
        pred_guesses = pred["word_guesses"].lower().split(";")
        score = 0
        for pw, gw in zip(pred_guesses, gold_word_guesses):
            if pw == gw:
                score += 1
            elif len(pw) == len(gw):
                score += 0.5
        score /= len(gold_word_guesses) if gold_word_guesses else 1
        scores.append(score)
    return scores


def words_letters_match_primalet(prompts, completions, answer, **kwargs):
    gold_first_pass = split_words_and_letters(parse_generation(answer)["first_pass"].split(" "))
    predicted_dicts = [parse_generation(c) for c in completions]

    scores = []
    for pred in predicted_dicts:
        pred_first_pass = split_words_and_letters(pred["first_pass"].split(" "))

        # Word match
        word_score = 0
        for pw, gw in zip(pred_first_pass["words"], gold_first_pass["words"]):
            if pw == gw:
                word_score += 1
            elif len(pw) == len(gw):
                word_score += 0.5
        word_score /= len(gold_first_pass["words"]) if gold_first_pass["words"] else 1

        # Letter match
        letter_score = 0
        for pl, gl in zip(pred_first_pass["letters"], gold_first_pass["letters"]):
            if pl.lower().replace(" ", "") == gl.lower().replace(" ", ""):
                letter_score += 1
        letter_score /= len(gold_first_pass["letters"]) if gold_first_pass["letters"] else 1

        scores.append(word_score + letter_score)
    return scores



def combined_rewards(prompts, completions, answer, **kwargs):

    # Parse once
    gold_dict = parse_generation(answer)
    predicted_dicts = [parse_generation(c) for c in completions]

    # Prepare gold values
    gold_solution = gold_dict["solution"].lower()
    gold_solution_words = gold_dict["solution_words"].lower().split(";")
    gold_word_guesses = gold_dict["word_guesses"].lower().split(";")
    gold_first_pass = split_words_and_letters(gold_dict["first_pass"].split(" "))

    all_scores = []

    for pred in predicted_dicts:
        pred_solution = pred["solution"].lower()
        pred_solution_words = pred["solution_words"].lower().split(";")
        pred_word_guesses = pred["word_guesses"].lower().split(";")
        pred_first_pass = split_words_and_letters(pred["first_pass"].split(" "))

        # --- exact_match_solution ---
        exact_match = 1.0 if pred_solution == gold_solution and pred_solution != "" else 0.0

        # --- perc_correct_words_solution ---
        pcw_score = 0
        for pw, gw in zip(pred_solution_words, gold_solution_words):
            if pw == gw:
                pcw_score += 1
            elif len(pw) == len(gw):
                pcw_score += 0.5
        pcw_score /= len(gold_solution_words) if gold_solution_words else 1

        # --- perc_correct_words_defres ---
        pwd_score = 0
        for pw, gw in zip(pred_word_guesses, gold_word_guesses):
            if pw == gw:
                pwd_score += 1
            elif len(pw) == len(gw):
                pwd_score += 0.5
        pwd_score /= len(gold_word_guesses) if gold_word_guesses else 1

        # --- words_letters_match_primalet ---
        # Words
        word_score = 0
        for pw, gw in zip(pred_first_pass["words"], gold_first_pass["words"]):
            if pw == gw:
                word_score += 1
            elif len(pw) == len(gw):
                word_score += 0.5
        word_score /= len(gold_first_pass["words"]) if gold_first_pass["words"] else 1

        # Letters
        letter_score = 0
        for pl, gl in zip(pred_first_pass["letters"], gold_first_pass["letters"]):
            if pl.lower().replace(" ", "") == gl.lower().replace(" ", ""):
                letter_score += 1
        letter_score /= len(gold_first_pass["letters"]) if gold_first_pass["letters"] else 1

        primalet_score = word_score + letter_score

        # Collect all scores into a list
        score = (exact_match + pcw_score + pwd_score + primalet_score) / 3
        all_scores.append(score)  # Append the score to the list

    # Return a list of scores
    return all_scores  # Return the list of scores
