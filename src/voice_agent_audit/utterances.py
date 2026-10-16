"""Template borrower utterances per (intent, style). Hand-written, synthetic."""
from __future__ import annotations
import random

T = {
    "will_pay": {
        "hinglish": ["Haan, main kal tak payment kar dunga. {amount_words} rupees na?",
                     "Ji haan, aaj shaam tak UPI se pay kar dunga."],
        "english": ["Yes, I will pay by tomorrow.", "Sure, I'll make the payment this evening."],
        "hindi_roman": ["Haan ji, main kal tak bhugtan kar dunga.", "Ji, aaj shaam tak paise bhej dunga."],
    },
    "needs_time": {
        "hinglish": ["Abhi thoda busy hoon, aap mujhe kal call back kar sakte ho?",
                     "Salary aane do, next week baat karte hain."],
        "english": ["I'm busy right now, can you call me tomorrow?", "Please call me next week."],
        "hindi_roman": ["Abhi samay nahi hai, kal phone kijiye.", "Agle hafte baat karte hain."],
    },
    "hardship": {
        "hinglish": ["Meri job chali gayi hai, abhi payment possible nahi hai.",
                     "Hospital ka kharcha bahut ho gaya, main pay nahi kar paunga."],
        "english": ["I lost my job, I cannot pay right now.", "I have medical expenses, I can't pay."],
        "hindi_roman": ["Meri naukri chhut gayi hai, abhi bhugtan nahi ho sakta.", "Ilaaj mein bahut kharcha hua hai."],
    },
    "dispute": {
        "hinglish": ["Ye amount galat hai, maine ye loan liya hi nahi.",
                     "Mera payment already ho chuka hai, aap galat bol rahe ho."],
        "english": ["This amount is wrong, I never took this loan.", "I already paid this, you are mistaken."],
        "hindi_roman": ["Ye rakam galat hai, maine loan nahi liya.", "Maine pehle hi paisa de diya hai."],
    },
    "wrong_person": {
        "hinglish": ["Ye {name} nahi hai, aapne wrong number lagaya hai.", "Yahan koi {name} nahi rehta."],
        "english": ["Wrong number, there is no {name} here.", "You have the wrong person."],
        "hindi_roman": ["Galat number hai, yahan {name} nahi rehta.", "Aapne galat insaan ko phone kiya."],
    },
    "abusive": {
        "hinglish": ["Bakwaas band karo, dobara phone mat karna!", "Tum log pareshan karte rehte ho, chup raho!"],
        "english": ["Stop harassing me, never call again!", "Shut up and leave me alone!"],
        "hindi_roman": ["Phone mat karo dobara, pareshan mat karo!", "Chup raho, bakwaas band karo!"],
    },
}


def borrower_turn(rng: random.Random, intent: str, style: str, name: str, amount: int) -> str:
    tmpl = rng.choice(T[intent][style])
    return tmpl.format(name=name.split()[0], amount_words=str(amount))
