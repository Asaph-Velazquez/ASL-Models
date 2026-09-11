"""Plantillas de orden ASL traduccion glosa->ingles, que seria bueno revisar"""
import random
import re
from collections import defaultdict

# slot = (nombre, {"class": [...]}|{"gloss": [...]}, prob)
TEMPLATES = {
    "room_service": [
        [("when",    {"gloss": ["NOW", "LATER", "TODAY"]},           0.30),
         ("subject", {"gloss": ["I"]},                               0.50),
         ("verb",    {"gloss": ["WANT", "NEED", "ORDER", "BRING"]},  1.00),
         ("qty",     {"gloss": ["MORE", "EXTRA", "ANOTHER"]},        0.25),
         ("object",  {"class": ["object", "food"]},                 1.00),
         ("place",   {"gloss": ["ROOM", "MY", "BATHROOM"]},          0.30),
         ("polite",  {"gloss": ["PLEASE"]},                          0.50)],
    ],
    "problem": [
        [("poss",    {"gloss": ["MY"]},                              0.55),
         ("topic",   {"class": ["place", "object"]},                 1.00),
         ("state",   {"class": ["adjective", "problem"]},            1.00),
         ("neg",     {"gloss": ["NOT"]},                             0.20),
         ("verb",    {"gloss": ["FIX", "CHECK"]},                    0.40),
         ("polite",  {"gloss": ["PLEASE"]},                          0.40)],
    ],
    "servicios": [
        [("topic",   {"class": ["place", "service"]},                1.00),
         ("wh",      {"gloss": ["WHERE", "WHEN", "HOW-MUCH"]},        0.60),
         ("verb",    {"gloss": ["INCLUDE", "OPEN", "CLOSE"]},         0.30)],
    ],
    "movilidad": [
        [("when",    {"gloss": ["NOW", "LATER"]},                    0.25),
         ("subject", {"gloss": ["I"]},                               0.40),
         ("verb",    {"gloss": ["WANT", "NEED", "CALL"]},            0.60),
         ("service", {"gloss": ["TAXI", "CAR", "VALET", "AIRPORT"]}, 1.00),
         ("polite",  {"gloss": ["PLEASE"]},                          0.40)],
    ],
    # Ramas genericas: funcionan con CUALQUIER vocabulario (aseguran cobertura aunque
    # el CSV aun no tenga las glosas hoteleras de las ramas de arriba).
    "generic_content": [
        [("a", {"class": ["object", "service", "place", "verb", "quantity",
                          "relational", "negation", "food", "adjective", "problem",
                          "time", "pronoun", "courtesy", "wh_question"]}, 1.00),
         ("b", {"class": ["object", "service", "place", "verb", "quantity",
                          "relational", "negation", "food", "adjective", "problem",
                          "time", "pronoun", "courtesy", "wh_question"]}, 1.00),
         ("c", {"class": ["object", "service", "place", "verb", "quantity"]}, 0.50)],
    ],
    "fingerspell": [
        [("l1", {"class": ["letter"]}, 1.00),
         ("l2", {"class": ["letter"]}, 1.00),
         ("l3", {"class": ["letter"]}, 0.70),
         ("l4", {"class": ["letter"]}, 0.40)],
    ],
    "number_seq": [
        [("n1", {"class": ["number"]}, 1.00),
         ("n2", {"class": ["number"]}, 1.00),
         ("n3", {"class": ["number"]}, 0.50)],
    ],
}

ENGLISH_TEMPLATES = {
    "room_service": [
        "I would like {qty} {object} {when} .",
        "Could you bring {qty} {object} to my {place} {when} , please ?",
        "I need {qty} {object} , please .",
    ],
    "problem": [
        "The {topic} in my room is {state} .",
        "There is a problem with the {topic} .",
        "My {topic} is {state} , please {verb} it .",
    ],
    "servicios": [
        "Where is the {topic} ?",
        "Is the {topic} {verb} ?",
        "What time does the {topic} {verb} ?",
    ],
    "movilidad": [
        "I need a {service} {when} .",
        "Please call a {service} .",
        "Can I get a {service} {when} ?",
    ],
    "generic_content": [
        "{a} {b} {c} .",
        "I mean : {a} {b} {c} .",
    ],
    "fingerspell": ["It is spelled {letters} ."],
    "number_seq": ["The number is {numbers} ."],
}


def clean_english(s):
    s = re.sub(r"\s+", " ", s).strip()
    for a, b in [(" ,", ","), (" .", "."), (" ?", "?"), (" !", "!"), (" :", ":")]:
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\ba\s+([aeiou])", r"an \1", s)
    s = re.sub(r"\s+([,.?!:])", r"\1", s)
    if s and s not in ".?!":
        s = s[0].upper() + s[1:]
        if s[-1] not in ".?!":
            s += "."
    return s


def build_english(branch, items, gloss_en, rng=random):
    """items: [(slot, gloss)]. Devuelve la oracion en ingles de la plantilla de la rama."""
    if branch == "fingerspell":
        letters = " ".join(gloss_en.get(g, g).replace("Letter ", "") for _, g in items)
        return clean_english(ENGLISH_TEMPLATES[branch][0].format(letters=letters))
    if branch == "number_seq":
        numbers = " ".join(gloss_en.get(g, g) for _, g in items)
        return clean_english(ENGLISH_TEMPLATES[branch][0].format(numbers=numbers))
    en = defaultdict(str, {slot: gloss_en.get(g, g).lower() for slot, g in items})
    s = rng.choice(ENGLISH_TEMPLATES[branch]).format_map(en)
    return clean_english(s)


def template_translate(gloss_sequence, vocab):
    """Fallback Capa 6: glosas -> ingles por reglas. Acotado al dominio; NO prosa libre."""
    tokens = gloss_sequence.split()
    if not tokens:
        return ""
    fc = vocab.functional_class
    letters = [t for t in tokens if fc.get(t) == "letter"]
    numbers = [t for t in tokens if fc.get(t) == "number"]
    content = [t for t in tokens if t not in letters and t not in numbers]

    if letters and not content and not numbers:
        spelled = " ".join(vocab.english.get(t, t).replace("Letter ", "") for t in letters)
        return clean_english(f"It is spelled {spelled} .")
    if numbers and not content and not letters:
        nums = " ".join(vocab.english.get(t, t) for t in numbers)
        return clean_english(f"The number is {nums} .")

    words = [vocab.english.get(t, t).lower() for t in tokens]
    verbs = {"WANT", "NEED", "ORDER", "BRING", "CALL"}
    if any(t in verbs for t in tokens):
        v = next(t for t in tokens if t in verbs)
        rest = [vocab.english.get(t, t).lower() for t in tokens
                if t not in verbs and t != "I" and t != "PLEASE"]
        s = f"I {vocab.english.get(v, v).lower()} " + " ".join(rest)
        if "PLEASE" in tokens:
            s += " , please"
        return clean_english(s + " .")
    return clean_english(" ".join(words) + " .")
