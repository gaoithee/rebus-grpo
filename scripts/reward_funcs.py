import re

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




def combined_rewards_broken(prompts, completions, answer, **kwargs):

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

def combined_rewards(prompts, completions, answer, **kwargs):
    # Parse once
    completions = [completions[i][0]['content'] for i in range(len(completions))]
    # print(answer)
    print("------")
    print(completions)
    print("------")
    gold_dict = parse_generation(str(answer[0]))
    predicted_dicts = [parse_generation(c) for c in completions]
    print(gold_dict)
    print(predicted_dicts)

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
        if pred_solution == gold_solution:
            exact_match = 2.0
        else:
            exact_match = 0.0  # Fallback for None case

        # --- perc_correct_words_solution ---
        pcw_score = 0
        if len(pred_solution_words) == 0:
            pcw_score = 0.0
        else:
            for pw, gw in zip(pred_solution_words, gold_solution_words):
                if pw == gw:
                    pcw_score += 2
                if len(pw) == len(gw):
                    pcw_score += 0.5
            pcw_score /= len(gold_word_guesses) if gold_word_guesses else 1

        # --- perc_correct_words_defres ---
        pwd_score = 0
        if len(pred_word_guesses) == 0:
            pwd_score = -0.0
        else:
            for pw, gw in zip(pred_word_guesses, gold_word_guesses):
                if pw == gw:
                    pwd_score += 2
                if len(pw) == len(gw):
                    pwd_score += 0.5
            pwd_score /= len(gold_word_guesses) if gold_word_guesses else 1

        # --- words_letters_match_primalet ---
        # Words
        word_score = 0
        if len(gold_first_pass["words"]) > 0:
            for pw, gw in zip(pred_first_pass["words"], gold_first_pass["words"]):
                if pw == gw:
                    word_score += 2
                elif len(pw) == len(gw):
                    word_score += 0.5
            word_score /= len(gold_first_pass["words"])

        # Letters
        letter_score = 0
        if len(gold_first_pass["letters"]) > 0:
            for pl, gl in zip(pred_first_pass["letters"], gold_first_pass["letters"]):
                if pl.lower().replace(" ", "") == gl.lower().replace(" ", ""):
                    letter_score += 1
            letter_score /= len(gold_first_pass["letters"])

        primalet_score = word_score + letter_score

        # Collect all scores into a list
        print(f"Exact match: {exact_match}")
        print(f"Percentage correct words solution: {pcw_score}")
        print(f"Percentage correct words def_res: {pwd_score}")
        print(f"PrimaLet score: {primalet_score}")
        print("--------")
        score = exact_match + pcw_score + pwd_score + primalet_score
        all_scores.append(score)
        print(score)

    return all_scores


# def combined_rewards_old(prompts, completions, answer, **kwargs):
    # Parse once
#    completions = [completions[i][0]['content'] for i in range(len(completions))]
#    print(answer)
#    print("------")
#    print(completions)
#    print("------")
#    gold_dict = parse_generation(str(answer[0]))
#    predicted_dicts = [parse_generation(c) for c in completions]
#    print(gold_dict)
#    print(predicted_dicts)

    # Prepare gold values
#    gold_solution = gold_dict["solution"].lower()
#    gold_solution_words = gold_dict["solution_words"].lower().split(";")
#    gold_word_guesses = gold_dict["word_guesses"].lower().split(";")
#    gold_first_pass = split_words_and_letters(gold_dict["first_pass"].split(" "))

#    all_scores = []

#    for pred in predicted_dicts:
#        pred_solution = pred["solution"].lower()
#        pred_solution_words = pred["solution_words"].lower().split(";")
#        pred_word_guesses = pred["word_guesses"].lower().split(";")
#        pred_first_pass = split_words_and_letters(pred["first_pass"].split(" "))

        # --- exact_match_solution ---
#        exact_match = 2.0 if pred_solution == gold_solution
#        exact_match = -5.0 if pred_solution == None

        # --- perc_correct_words_solution ---
#        pcw_score = 0
#        pwc_score = -5.0 if len(pred_solution_words) == 0
#        for pw, gw in zip(pred_solution_words, gold_solution_words):
#            if pw == gw:
#                pcw_score += 2
#            if len(pw) == len(gw):
#                pcw_score += 0.5
#        pcw_score /= len(gold_word_guesses) if gold_word_guesses else 1

        # --- perc_correct_words_defres ---
#        pwd_score = 0
#        pwd_score = -5.0 if len(pred_word_guesses) == 0
#        for pw, gw in zip(pred_word_guesses, gold_word_guesses):
#            if pw == gw:
#                pwd_score += 2
#            if len(pw) == len(gw):
#                pwd_score += 0.5
#        pwd_score /= len(gold_word_guesses) if gold_word_guesses else 1

        # --- words_letters_match_primalet ---
        # Words
#        word_score = 0
#        for pw, gw in zip(pred_first_pass["words"], gold_first_pass["words"]):
#            if pw == gw:
#                word_score += 2
#            elif len(pw) == len(gw):
#                word_score += 0.5
#            word_score /= len(gold_first_pass["words"]) if gold_first_pass["words"] else 1

        # Letters
#        letter_score = 0
#        for pl, gl in zip(pred_first_pass["letters"], gold_first_pass["letters"]):
#            if pl.lower().replace(" ", "") == gl.lower().replace(" ", ""):
#                letter_score += 1
#        letter_score /= len(gold_first_pass["letters"]) if gold_first_pass["letters"] else 1
#        primalet_score = word_score + letter_score
        # Collect all scores into a list
#        print(f"Exact match: ", exact_match)
#        print(f"Percentage correct words solution: ", pcw_score)
#        print(f"Percentage correct words def_res: ", pwd_score)
#        print(f"PrimaLet score: ", primalet_score)
#        print("--------")
#        score = exact_match + pcw_score + pwd_score + primalet_score
#        all_scores.append(score)
#        print(score)
#    return all_scores


def combined_rewards_gab(prompts, completions, answer, **kwargs):
    print(answer)
    print("------")
    print(completions)
    print("------")
    # Parse once
    gold_dict = parse_generation(str(answer[0]))
    print(gold_dict)
    predicted_dicts = [parse_generation(c) for c in completions]
    print(predicted_dicts)

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
        exact_match = 5.0 if pred_solution == gold_solution and pred_solution != "" else 0.0

        # --- perc_correct_words_solution ---
        pcw_score = 0
        for pw, gw in zip(pred_solution_words, gold_solution_words):
            if pw == gw:
                pcw_score += 2
            elif len(pw) == len(gw):
                pcw_score += 0.2
            else:
                pcw_score -= 0.25

        # --- perc_correct_words_defres ---
        pwd_score = 0
        for pw, gw in zip(pred_word_guesses, gold_word_guesses):
            if pw == gw:
                pwd_score += 1
            elif len(pw) == len(gw):
                pwd_score += 0.1
            else:
                pwd_score -= 0.5

        # --- words_letters_match_primalet ---
        # Words
        primalet_score = 0
        for pw, gw in zip(pred_first_pass["words"], pred_word_guesses):
            if pw == gw:
                primalet_score += 1
            else:
                primalet_score -= 1            


        # Collect all scores into a list
        # print(f"Exact match: ", exact_match)
        # print(f"Percentage correct words solution: ", pcw_score)
        # print(f"Percentage correct words def_res: ", pwd_score)
        # print(f"PrimaLet score: ", primalet_score)
        # print("--------")
        score = exact_match + pcw_score + pwd_score + primalet_score
        all_scores.append(score)

    return all_scores
