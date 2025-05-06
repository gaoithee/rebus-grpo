from utils import estrai_soluzione, estrai_primalet, estrai_indizi 

def exact_match_solution(prompts, completions, answer, **kwargs):
    predicted = [estrai_soluzione(completion) for completion in completions]
    gold = estrai_soluzione(answer)
    scores = []
    for guess in predicted:
        if guess is None:
            scores.append(0)
            continue
        try:
            scores.append(1.0 if guess == gold else 0.0)
        except:
            scores.append(0)
            continue
    return scores


def perc_correct_words_solution(prompts, completions, answer, **kwargs):
    gold = estrai_soluzione(answer).lower().split()
    scores = []

    for completion in completions:
        pred = estrai_soluzione(completion)
        print(pred)
        if not pred:
            continue

        pred = pred.lower().split()
        score = 0
        for pw, gw in zip(pred, gold):
            if pw == gw:
                score += 1
            elif len(pw) == len(gw):
                score += 0.5
        scores.append(score / len(gold))

    return scores


def exact_match_primalet(prompts, completions, answer, **kwargs):
    predicted = [estrai_primalet(completion) for completion in completions]
    golden = estrai_primalet(answer).lower().replace(" ", "")
    scores = []
    for guess in predicted:
        if guess is None:
            scores.append(0)
            continue
        try:
            scores.append(1.0 if guess.lower().replace(" ", "") == golden else 0.0)
        except:
            scores.append(0)
            continue
    return scores


def perc_correct_defres(prompts, completions, answer, **kwargs):
    predicted = [estrai_indizi(completion.replace("*", "")) for completion in completions] 
    golden = estrai_indizi(answer)
    word_scores = []
    letter_scores = []
    for pred in predicted:
        wscore = 0
        for pw, gw in zip(pred['words'], golden['words']):
            if pw == gw:
                wscore += 1
            elif len(pw) == len(gw):
                wscore += 0.5
        word_scores.append(wscore / len(golden['words']))

        lscore = 0
        for pw, gw in zip(pred['letters'], golden['letters']):
            if pw.lower().replace(" ", "") == gw.lower().replace(" ", ""):
                lscore += 1
        letter_scores.append(lscore / len(golden['letters']))
        
    return [word_scores[i] + letter_scores[i] for i in range(len(predicted))]