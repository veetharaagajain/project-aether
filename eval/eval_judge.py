"""One judge, shared by the harnesses, so a score is comparable across runs."""
import json

MODEL = 'claude-haiku-4-5'
SYSTEM = (
    "You are scoring a memory system's answer against a reference answer.\n\n"
    "Say correct only if the answer conveys the same fact as the reference. "
    "Wording, length and phrasing do not matter -- the system deliberately "
    "rephrases rather than quoting, so 'You had a sausage croissant' and "
    "'sausage croissant' are the same answer. Extra correct detail is fine. "
    "A different fact, a missing fact, or a refusal is not correct.\n\n"
    "Reply as JSON: {\"correct\": true|false, \"why\": \"a few words\"}")
SCHEMA = {'type': 'object',
          'properties': {'correct': {'type': 'boolean'},
                         'why': {'type': 'string'}},
          'required': ['correct', 'why'], 'additionalProperties': False}


def judge_one(client, question, gold, pred):
    r = client.messages.create(
        model=MODEL, max_tokens=200, system=SYSTEM,
        messages=[{'role': 'user',
                   'content': f"Question: {question}\n\nReference answer: "
                              f"{gold}\n\nThe system said: {pred}"}],
        output_config={'format': {'type': 'json_schema', 'schema': SCHEMA}})
    txt = "".join(b.text for b in r.content if b.type == 'text')
    try:
        return bool(json.loads(txt).get('correct'))
    except json.JSONDecodeError:
        return False
