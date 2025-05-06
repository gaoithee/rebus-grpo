import re

def estrai_soluzione(input_string):
    # Extract only the solution
    match = re.search(r"Soluzione: (.+?)\n", input_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None

def estrai_indizi(input_string):

    # Extract the `[...] = relevantPart`
    pattern = r"\[([^\]]+)\] = ([^\n]+)"
    indizi = re.findall(pattern, input_string)
    risposte = [risposta for _, risposta in indizi]
    
    # Extract the `... = relevantLetters`
    pattern_sigle = r"[-–•]?\s*([A-Z]+(?:\s+[A-Z]+)*)\s*=\s*[^\n]+"    
    risposte_sigle = re.findall(pattern_sigle, input_string)
    
    # Combina le risposte estratte dalle parentesi e quelle singole
    return {'letters': risposte_sigle, 'words': risposte}

def estrai_primalet(input_string):
    # Extract the first-pass (prima lettura in italian)
    match = re.search(r"Prima lettura: (.+?)\n", input_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def estrai_rebus_e_chiave(testo):
    # Extract from the problem formulation: 
    #   - the rebus problem 
    #   - the key (i.e. bounds of number of letters of the solution)
    rebus_match = re.search(r"Rebus:\s*(.+?)\s*Chiave di lettura:", testo, re.DOTALL)
    chiave_match = re.search(r"Chiave di lettura:\s*(.+)", testo)

    rebus_raw = rebus_match.group(1).strip() if rebus_match else ""
    chiave = chiave_match.group(1).strip() if chiave_match else ""

    return rebus_raw, chiave


