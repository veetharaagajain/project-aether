"""The one place that names a model.

Three decisions -- relevance, recall and addressed -- each reached for Apple's
on-device model directly through bridge/relevance.swift. That worked, and it
meant three files named a model, which is the thing the design said should
never happen: three places a migration has to touch, and no single point where
a question could be sent somewhere else.

So: a caller says what it NEEDS, not who should do it.

  classify   is this X or not X -- a bounded judgement over given text
  extract    pull a stated fact out of given text
  compose    say given text back in one's own words
  reason     work something out that the text does not state
  world      answer from knowledge that was never in the text at all

The first three are what a small on-device model is for. The last two are what
it is not for, and the platform says so. Nothing above this module names a
provider, and adding one is adding an entry to PROVIDERS.

DELIBERATELY NOT A GRAND SCHEME. There are no manifests, no capability
negotiation and no scoring. A task names a capability, the first available
provider serving it wins, and the order is fixed and written down. That is the
smallest thing that removes the three hardcoded calls and gives one place where
a question can leave.
"""

import time

# A single reply cannot run away: this is the ceiling on what one request
# can be charged for, and it is small because every task here wants one or two
# sentences or a short JSON object.
GEMINI_MAX_OUTPUT = 512

# Nothing here can run up a bill quietly. Every paid call is counted in
# store/spend.json before the reply is returned, and the day's ceiling is
# checked before the request is made, so the failure is a refusal rather than
# an invoice. The numbers are deliberately small: this answers a handful of
# questions a day, not a workload.
SPEND_FILE = 'store/spend.json'
DAILY_CALL_LIMIT = 200
DAILY_TOKEN_LIMIT = 300_000
# Per million tokens, for the running estimate only. These are the 2.5-flash
# list prices and I have not verified them for 3.6-flash, so the usd figure in
# store/spend.json is an order-of-magnitude guide and Google's bill is the
# truth. The call and token ceilings are what actually stop anything, and they
# do not depend on the price being right.
PRICE_IN_PER_M = 0.30
PRICE_OUT_PER_M = 2.50
# claude-opus-5 list price, per million tokens.
CLAUDE_IN_PER_M = 5.00
CLAUDE_OUT_PER_M = 25.00


def classify_429(detail):
    """A 429 is two different failures wearing one status code.

    "Too many requests per minute" is transient: waiting fixes it, and a retry
    later is right. "Your prepayment credits are depleted" is not: every retry
    will fail identically, and treating it as a rate limit means hammering an
    endpoint forever while reporting a temporary condition. The first version
    of this counted both as rate_limited, which is exactly the silent failure
    the counter exists to prevent.
    """
    d = (detail or '').lower()
    if any(w in d for w in ('credit', 'billing', 'prepayment', 'quota exceeded',
                            'exceeded your current quota', 'plan and billing')):
        return 'out_of_credit'
    return 'rate_limited'


class OutwardRefused(Exception):
    """A provider that would send text off the machine was chosen, and the
    gate said no. Carries the gate's own reason."""


class OverBudget(Exception):
    """The day's ceiling for paid calls is reached. Distinct from NoProvider:
    something can do this, and is deliberately not being asked."""


def spend_path():
    from pathlib import Path
    return Path(__file__).resolve().parent / SPEND_FILE


def spend_today(provider='gemini'):
    import json
    import time
    p = spend_path()
    day = time.strftime('%Y-%m-%d')
    try:
        all_ = json.loads(p.read_text())
    except Exception:                                        # noqa: BLE001
        all_ = {}
    return all_.get(f'{provider}:{day}',
                    {'calls': 0, 'in': 0, 'out': 0, 'usd': 0.0,
                     'errors': 0, 'rate_limited': 0})


def check_budget(provider='gemini'):
    d = spend_today(provider)
    if d['calls'] >= DAILY_CALL_LIMIT:
        raise OverBudget(f"{provider} has been asked {d['calls']} times today, "
                         f"the ceiling is {DAILY_CALL_LIMIT}; nothing was sent")
    if d['in'] + d['out'] >= DAILY_TOKEN_LIMIT:
        raise OverBudget(f"{provider} has used {d['in']+d['out']} tokens today, "
                         f"the ceiling is {DAILY_TOKEN_LIMIT}; nothing was sent")
    return d


def record_spend(provider, usage, error=None, rate_limited=False,
                 prices=(PRICE_IN_PER_M, PRICE_OUT_PER_M)):
    import json
    import time
    p = spend_path()
    day = time.strftime('%Y-%m-%d')
    key = f'{provider}:{day}'
    try:
        all_ = json.loads(p.read_text())
    except Exception:                                        # noqa: BLE001
        all_ = {}
    d = all_.get(key, {'calls': 0, 'in': 0, 'out': 0, 'usd': 0.0,
                       'errors': 0, 'rate_limited': 0})
    d['calls'] += 1
    d.setdefault('out_of_credit', 0)
    ti = int((usage or {}).get('promptTokenCount') or 0)
    to = int((usage or {}).get('candidatesTokenCount') or 0)
    d['in'] += ti
    d['out'] += to
    d['usd'] = round(d['usd'] + ti * prices[0] / 1e6
                     + to * prices[1] / 1e6, 6)
    if error:
        d['errors'] += 1
    if rate_limited == 'out_of_credit':
        d['out_of_credit'] += 1
        # sticky for the rest of the day: retrying without credit cannot
        # succeed, and a provider that fails every call is worse than one that
        # says plainly that it is unavailable.
        d['halted'] = 'out of credit'
    elif rate_limited:
        d['rate_limited'] += 1
    all_[key] = d
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(all_, indent=1, sort_keys=True) + "\n")
    return d

CAPABILITIES = ('classify', 'extract', 'compose', 'reason', 'world')

# Every task the system performs, and what it needs. Callers name a task; only
# this table maps tasks to capabilities, and only PROVIDERS maps capabilities to
# models.
TASKS = {
    'judge_relevance': 'classify',    # does this fragment answer the question
    'is_addressed': 'classify',       # was this said to us
    'compose_answer': 'compose',      # one sentence from what was heard
    # the refusal, composed rather than fixed. Same capability as composing an
    # answer because it is the same job on the same material -- what differs is
    # that the sentence has to say the answer is absent and stay a refusal.
    'compose_not_found': 'compose',
    # classify-then-compose in one call: the schema forces the worth judgement
    # before the sentence, and a false worth discards the rest. Filed under
    # compose because that is the harder half and a provider serving compose
    # serves classify.
    'conclude_from': 'compose',
    'judge_supersedes': 'classify',   # does this conclusion replace that one
    'answer_with_reasoning': 'reason',
    'answer_from_world': 'world',
}


class NoProvider(Exception):
    """Nothing available can do this. Carries why each candidate could not.

    Deliberately distinct from a provider failing mid-request: a caller has to
    be able to tell "nobody can do this" from "the one who can, broke".
    """


class Provider:
    name = 'abstract'
    capabilities = ()
    local = True          # does the text stay on this machine
    costs = False         # does a request cost money
    order = 100           # lower is preferred

    def available(self):
        """(bool, reason). The reason is shown to a person, so it says what
        would have to change rather than what failed."""
        return False, 'not implemented'

    def run(self, task, payload, budget):
        raise NotImplementedError


class OnDevice(Provider):
    """Apple's on-device model, through the long-lived Swift bridge.

    Free, unlimited, and never leaves the machine, which is why it is first for
    everything it can do. It cannot reason and does not know things, so it does
    not claim those capabilities -- and because it does not claim them, the rule
    that it never answers from general knowledge is enforced by routing rather
    than by asking it nicely.
    """
    name = 'apple-on-device'
    capabilities = ('classify', 'extract', 'compose')
    local = True
    costs = False
    order = 0

    def available(self):
        # a real health check, not a liveness check: see relevance.available.
        # It is first in order for everything it serves, so a false yes here
        # means nothing ever falls through to a provider that works.
        import relevance as rel
        ok, why = rel.available()
        return ok, ('ready' if ok else (why or 'the bridge would not start'))

    def run(self, task, payload, budget):
        import json
        import relevance as rel
        cmd = {'judge_relevance': 'judge', 'is_addressed': 'addressed',
               'compose_answer': 'recall', 'conclude_from': 'conclude',
               'compose_not_found': 'notfound',
               'judge_supersedes': 'supersedes'}[task]
        body = json.dumps(payload).encode()
        return rel.request(f"{cmd} {len(body)}\n", body, budget)


class AppleCloud(Provider):
    """Private Cloud Compute. Designed for, not reachable.

    It would serve reason as well, on Apple's terms, without a per-token cost.
    It needs an entitlement that has not been applied for, so it is declared and
    unavailable, which is more useful than leaving it out: the routing failure
    names it and says what would make it work.
    """
    name = 'apple-private-cloud'
    capabilities = ('classify', 'extract', 'compose', 'reason')
    local = False
    costs = False
    order = 1

    def available(self):
        return False, ('needs the Private Cloud Compute entitlement, which has '
                       'not been applied for')


CLAUDE_MODEL = 'claude-opus-5'
CLAUDE_KEY_FILE = 'store/anthropic.key'
# Small on purpose. Every task here wants one or two sentences or a short JSON
# object, and a ceiling is the only thing standing between a loop over a day of
# episodes and a bill nobody chose.
CLAUDE_MAX_TOKENS = 1024
# Thinking is on by default on this model. Low effort keeps it cheap for what
# are bounded judgements -- is this worth concluding, does this answer that --
# rather than open-ended work.
CLAUDE_EFFORT = 'low'
# The shape conclude_from must come back in. Given as a schema rather than
# asked for in prose, so a reply that does not fit is rejected by the API
# instead of by a json.JSONDecodeError here.
# The shape compose_answer must come back in, matching what
# bridge/relevance.swift produces for the same task, so recall.compose_from
# can apply the same two checks whichever provider answered.
RECALL_SCHEMA = {
    'type': 'object',
    'properties': {
        'canAnswer': {'type': 'boolean'},
        'answer': {'type': 'string'},
        'support': {'type': 'string'},
    },
    'required': ['canAnswer', 'answer', 'support'],
    'additionalProperties': False,
}

CONCLUDE_SCHEMA = {
    'type': 'object',
    'properties': {
        'worth': {'type': 'boolean'},
        'statement': {'type': 'string'},
        'about': {'type': 'string'},
        'support': {'type': 'string'},
    },
    'required': ['worth', 'statement', 'about', 'support'],
    'additionalProperties': False,
}


def claude_key():
    """The key, from the environment or from a private file.

    Same arrangement as gemini_key, for the same reason: a launchd agent has no
    login shell, and putting a live credential in the plist would put it in
    ~/Library/LaunchAgents, which is world-readable and gets backed up.
    """
    import os
    from pathlib import Path
    k = os.environ.get('ANTHROPIC_API_KEY')
    if k:
        return k.strip(), 'environment'
    f = Path(__file__).resolve().parent / CLAUDE_KEY_FILE
    if f.exists():
        return f.read_text().strip(), str(f)
    return None, None


class PaidAPI(Provider):
    """A hosted frontier model. Costs money and sends text off the machine.

    Last by order, and gated separately: reaching it is a disclosure, so
    gate.release_outward decides whether it may happen at all. Being available
    is not permission.
    """
    name = 'claude-api'
    capabilities = ('classify', 'extract', 'compose', 'reason', 'world')
    local = False
    costs = True
    order = 9

    def available(self):
        k, _ = claude_key()
        if not k:
            return False, ('no ANTHROPIC_API_KEY in the environment and no '
                           f'{CLAUDE_KEY_FILE}')
        try:
            import anthropic          # noqa: F401
        except ImportError:
            return False, ('the anthropic client is not installed '
                           '(uv add anthropic)')
        halted = spend_today(self.name).get('halted')
        if halted:
            return False, (f'{halted} (as of today; clear store/spend.json to '
                           f'retry sooner)')
        return True, 'ready'

    def body(self, task, payload):
        """Exactly what goes over the wire, as a dict, so a caller can look at
        it before it is sent and a test can assert on it.

        Same function on both hosted providers on purpose: "what crosses the
        boundary" should be answerable by reading one method per provider, not
        by tracing a request builder.
        """
        q = (payload.get('question') or '').strip()
        ctx = payload.get('context') or []
        system = {
            'answer_with_reasoning':
                "Work out the answer to the question. If context lines are "
                "given they are things the person said; use them where they "
                "bear on the question. Answer in one or two short sentences. "
                "If you cannot work it out, say so plainly.",
            'answer_from_world':
                "Answer the question from general knowledge, in one or two "
                "short sentences. If you do not know, say so plainly rather "
                "than guessing.",
            'compose_answer':
                "You answer a question about someone using ONLY the transcript "
                "you are given.\n\n"
                "If it plainly contains the answer, give it in ONE SHORT "
                "SENTENCE OF YOUR OWN, and separately copy out the single line "
                "you took it from, exactly as it appears. Answer in your own "
                "words rather than reading the line back: the person is "
                "listening and already said the original.\n\n"
                "EACH LINE IS WRITTEN AS: [date] speaker: what they said\n\n"
                "The date and the speaker are part of the record, not part of "
                "the sentence somebody spoke. Use them. A question about WHEN "
                "something happened is usually answered by the date on the "
                "line rather than by words inside it -- \"[15 May 2023] Jon: "
                "I'm currently reading The Lean Startup\" says he started it "
                "around May 2023, and answering \"he doesn't say\" would be "
                "wrong. A question about WHO did something is answered by the "
                "speaker on the line. Copy the whole line, date and speaker "
                "included, into support.\n\n"
                "Address them as \"you\". Under about fifteen words. Use only "
                "facts that are in the transcript -- rephrasing is required, "
                "adding is not allowed.\n\n"
                "If the transcript does not contain the answer, set canAnswer "
                "to false and leave the other fields empty. Do this whenever "
                "you are unsure, and whenever the answer would come from "
                "anything you know rather than from the lines in front of you. "
                "You are not being asked what is true, you are being asked "
                "what was said.",
            'conclude_from':
                "You are shown a stretch of one person's speech, transcribed "
                "from a microphone in their home. Decide whether it contains "
                "something worth remembering about them as a standing fact -- "
                "a preference, a commitment, a relationship, a constraint. "
                "Most stretches do not, and the honest answer is usually "
                "false.\n\n"
                "Be strict about whose fact it is. A microphone in a room "
                "picks up television, films and other people, and a line of "
                "dialogue is not a fact about the listener. If the speech "
                "reads as a scene, a script, or characters talking to each "
                "other, set worth false. Only conclude something about the "
                "person when it is plainly the person speaking about their "
                "own life.\n\n"
                "statement is one sentence about the person. about is one or "
                "two words naming the subject. support is a line copied "
                "exactly from the speech that the statement rests on.",
        }.get(task, "Answer the question.")

        parts = []
        if task == 'conclude_from':
            parts += ["The speech:", payload.get('transcript') or '']
        elif task == 'compose_answer':
            # the transcript was being dropped entirely here: this branch did
            # not exist, so compose_answer fell through to the question-only
            # path below and Claude was asked about people it had never been
            # shown. Fifty-one benchmark questions came back "I don't have any
            # information about John", which is a correct answer to the
            # question it was actually asked.
            parts += [f"Question: {payload.get('question') or ''}", '',
                      "Transcript:", payload.get('transcript') or '', '',
                      "Answer only from the transcript above, or decline."]
        else:
            if ctx:
                parts += ["Things the person said, for context:"]
                parts += [f"- {c}" for c in ctx]
                parts += ['']
            parts += [f"Question: {q}"]

        out = {
            'model': CLAUDE_MODEL,
            'max_tokens': CLAUDE_MAX_TOKENS,
            'system': system,
            'messages': [{'role': 'user', 'content': "\n".join(parts)}],
            'output_config': {'effort': CLAUDE_EFFORT},
        }
        if task == 'conclude_from':
            out['output_config']['format'] = {'type': 'json_schema',
                                              'schema': CONCLUDE_SCHEMA}
        elif task == 'compose_answer':
            out['output_config']['format'] = {'type': 'json_schema',
                                              'schema': RECALL_SCHEMA}
        return out

    def run(self, task, payload, budget):
        import json
        import anthropic
        key, source = claude_key()
        if not key:
            raise NoProvider('no Anthropic key')
        check_budget(self.name)          # raises OverBudget before sending
        body = self.body(task, payload)
        sent = len(json.dumps(body))
        client = anthropic.Anthropic(api_key=key, timeout=budget, max_retries=0)
        try:
            r = client.messages.create(**body)
        except anthropic.RateLimitError as e:
            detail = str(e)[:300]
            why = classify_429(detail)
            record_spend(self.name, None, error='429', rate_limited=why)
            return {'error': f'HTTP 429: {detail}', 'status': 429,
                    'rate_limited': why == 'rate_limited',
                    'out_of_credit': why == 'out_of_credit',
                    'sent_chars': sent}
        except anthropic.APIStatusError as e:
            detail = str(e)[:300]
            why = classify_429(detail) if e.status_code == 429 else False
            record_spend(self.name, None, error=str(e.status_code),
                         rate_limited=why)
            return {'error': f'HTTP {e.status_code}: {detail}',
                    'status': e.status_code,
                    'rate_limited': why == 'rate_limited',
                    'out_of_credit': why == 'out_of_credit',
                    'sent_chars': sent}
        except Exception as e:                                # noqa: BLE001
            record_spend(self.name, None, error=type(e).__name__)
            return {'error': f'{type(e).__name__}: {e}', 'sent_chars': sent}

        usage = {'promptTokenCount': r.usage.input_tokens,
                 'candidatesTokenCount': r.usage.output_tokens}
        spent = record_spend(self.name, usage, prices=(CLAUDE_IN_PER_M,
                                                       CLAUDE_OUT_PER_M))
        # a refusal is a real outcome, not an exception: the model declined
        # rather than failed, and saying so is more useful than "it errored"
        if r.stop_reason == 'refusal':
            cat = getattr(getattr(r, 'stop_details', None), 'category', None)
            return {'error': f'the model declined ({cat})', 'usage': usage,
                    'refusal': True, 'sent_chars': sent, 'today': spent}
        text = "".join(b.text for b in r.content if b.type == 'text').strip()
        if not text:
            return {'error': f'no text in reply (stop_reason {r.stop_reason})',
                    'usage': usage, 'sent_chars': sent, 'today': spent}
        out = {'answer': text, 'usage': usage, 'sent_chars': sent,
               'model': CLAUDE_MODEL, 'key_from': source, 'today': spent,
               'stop_reason': r.stop_reason}
        if task in ('conclude_from', 'compose_answer'):
            try:
                out.update(json.loads(text))
            except json.JSONDecodeError as e:
                out['error'] = f'reply was not the JSON asked for: {e}'
        return out


# What a Gemini request carries, written out rather than assembled somewhere
# else, because this is the one place text leaves the machine and the shape of
# what leaves has to be readable here.
# Named explicitly rather than using an alias like gemini-flash-latest: a
# floating alias would change the model under a stored belief without anything
# recording that it had. gemini-2.5-flash was the first choice and the API
# refused it -- "no longer available to new users" -- which is the same failure
# in slower motion, so the version here is checked against the models endpoint
# and moved deliberately.
GEMINI_MODEL = 'gemini-3.6-flash'
GEMINI_HOST = 'generativelanguage.googleapis.com'
GEMINI_KEY_FILE = 'store/gemini.key'


def gemini_key():
    """The key, from the environment or from a private file.

    launchd agents do not inherit a login shell, so the environment is empty
    under the service and GEMINI_API_KEY only exists for a hand-started run.
    Putting it in the plist would work and would also put a live credential in
    ~/Library/LaunchAgents, which is world-readable and gets backed up. So the
    service reads it from store/gemini.key, mode 0600, inside the directory
    memory.open already chmods to 0700, and gitignored with everything else
    derived. The environment still wins when it is set, so an interactive run
    needs no file.
    """
    import os
    from pathlib import Path
    k = os.environ.get('GEMINI_API_KEY')
    if k:
        return k.strip(), 'environment'
    f = Path(__file__).resolve().parent / GEMINI_KEY_FILE
    if f.exists():
        return f.read_text().strip(), str(f)
    return None, None


class Gemini(Provider):
    """Google's hosted model. Serves what nothing here can: reason and world.

    Costs money and sends text off the machine, so it sits behind
    gate.release_outward like any other outward path -- being available is not
    permission, and the two are deliberately separate settings.
    """
    name = 'gemini'
    capabilities = ('reason', 'world')
    local = False
    costs = True
    order = 7

    def available(self):
        k, _ = gemini_key()
        if not k:
            return False, ('no GEMINI_API_KEY in the environment and no '
                           f'{GEMINI_KEY_FILE}')
        halted = spend_today(self.name).get('halted')
        if halted:
            return False, (f'{halted} (as of today; clear store/spend.json to '
                           f'retry sooner)')
        return True, 'ready'

    def body(self, task, payload):
        """Exactly what goes over the wire, as a dict, so a caller can look at
        it before it is sent and a test can assert on it."""
        import json
        q = (payload.get('question') or '').strip()
        ctx = payload.get('context') or []
        instruction = {
            'answer_with_reasoning':
                "Work out the answer to the question. If context lines are "
                "given they are things the person said; use them where they "
                "bear on the question. Answer in one or two short sentences. "
                "If you cannot work it out, say so plainly.",
            'answer_from_world':
                "Answer the question from general knowledge, in one or two "
                "short sentences. If you do not know, say so plainly rather "
                "than guessing.",
            'conclude_from':
                "You are shown a stretch of one person's speech. Decide "
                "whether it contains something worth remembering about them "
                "as a standing fact -- a preference, a commitment, a "
                "relationship, a constraint. Most stretches do not. Reply as "
                "JSON with keys: worth (boolean), statement (one sentence "
                "about the person, empty if not worth), about (one or two "
                "words naming the subject), support (one line copied exactly "
                "from the speech that the statement rests on).",
        }.get(task, "Answer the question.")

        parts = [instruction, '']
        if task == 'conclude_from':
            parts += ["The speech:", payload.get('transcript') or '']
        else:
            if ctx:
                parts += ["Things the person said, for context:"]
                parts += [f"- {c}" for c in ctx]
                parts += ['']
            parts += [f"Question: {q}"]
        text = "\n".join(parts)
        out = {
            'contents': [{'role': 'user', 'parts': [{'text': text}]}],
            'generationConfig': {'temperature': 0.0,
                                 'maxOutputTokens': GEMINI_MAX_OUTPUT},
        }
        if task == 'conclude_from':
            out['generationConfig']['responseMimeType'] = 'application/json'
        return out

    def run(self, task, payload, budget):
        import json
        import urllib.error
        import urllib.request
        key, source = gemini_key()
        if not key:
            raise NoProvider('no Gemini key')
        check_budget(self.name)          # raises OverBudget before sending
        body = self.body(task, payload)
        raw = json.dumps(body).encode()
        req = urllib.request.Request(
            f"https://{GEMINI_HOST}/v1beta/models/{GEMINI_MODEL}:generateContent",
            data=raw, method='POST',
            headers={'Content-Type': 'application/json',
                     'x-goog-api-key': key})
        try:
            with urllib.request.urlopen(req, timeout=budget) as r:
                reply = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode('utf-8', 'replace')[:300]
            # 429 is a rate limit and 4xx is usually quota or a bad key. Both
            # are named rather than folded into "it failed", because a silent
            # rate limit is the failure this whole path is most likely to hit.
            why = classify_429(detail) if e.code == 429 else False
            record_spend(self.name, None, error=str(e.code), rate_limited=why)
            return {'error': f'HTTP {e.code}: {detail}', 'status': e.code,
                    'rate_limited': why == 'rate_limited',
                    'out_of_credit': why == 'out_of_credit',
                    'sent_chars': len(raw)}
        except Exception as e:                                   # noqa: BLE001
            record_spend(self.name, None, error=type(e).__name__)
            return {'error': f'{type(e).__name__}: {e}', 'sent_chars': len(raw)}

        usage = reply.get('usageMetadata') or {}
        spent = record_spend(self.name, usage)
        try:
            text = reply['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            fin = ((reply.get('candidates') or [{}])[0]).get('finishReason')
            return {'error': f'no text in reply (finishReason {fin})',
                    'usage': usage, 'sent_chars': len(raw)}
        out = {'answer': text.strip(), 'usage': usage, 'sent_chars': len(raw),
               'model': GEMINI_MODEL, 'key_from': source, 'today': spent}
        if task in ('conclude_from', 'compose_answer'):
            try:
                out.update(json.loads(text))
            except json.JSONDecodeError as e:
                out['error'] = f'reply was not the JSON asked for: {e}'
        return out


class LocalOpen(Provider):
    """An open model running here. Would reason without leaving the machine.

    This laptop has 16 GB and typically under 2 GB free, so a model large
    enough to reason usefully does not fit alongside the recogniser, the
    embedder and the on-device model that are already resident. Declared so the
    routing failure can say that rather than being silent about the option.
    """
    name = 'local-open-model'
    capabilities = ('classify', 'extract', 'compose', 'reason')
    local = True
    costs = False
    order = 5

    def available(self):
        import shutil
        runner = next((c for c in ('ollama', 'llama-server', 'mlx_lm.server')
                       if shutil.which(c)), None)
        if not runner:
            return False, ('no local model runner installed '
                           '(ollama, llama-server or mlx_lm.server)')
        return True, f'{runner} is installed'


PROVIDERS = [OnDevice(), AppleCloud(), LocalOpen(), Gemini(), PaidAPI()]


def providers_for(capability):
    return sorted((p for p in PROVIDERS if capability in p.capabilities),
                  key=lambda p: p.order)


def survey():
    """Every provider, what it can do, and whether it can do it today."""
    out = []
    for p in sorted(PROVIDERS, key=lambda x: x.order):
        ok, why = p.available()
        out.append({'name': p.name, 'capabilities': list(p.capabilities),
                    'available': ok, 'why': why, 'local': p.local,
                    'costs': p.costs, 'order': p.order})
    return out


def route(task):
    """Which provider gets this task. Raises NoProvider with every reason."""
    cap = TASKS.get(task)
    if cap is None:
        raise NoProvider(f'unknown task {task!r}')
    tried = []
    for p in providers_for(cap):
        ok, why = p.available()
        if ok:
            return p, cap
        tried.append(f'{p.name}: {why}')
    raise NoProvider(
        f"nothing available can {cap}.\n    " + "\n    ".join(tried))


def outward_text(task, payload):
    """Everything in this payload that would leave the machine.

    Written as one function so the answer to "what crosses the boundary" is
    read off the code rather than reasoned about. Whatever is in here is what
    is sent; there is no other channel.
    """
    out = []
    for k in ('question', 'transcript', 'old', 'new', 'utterance', 'context'):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            out.append(v)
        elif isinstance(v, (list, tuple)):
            out += [str(x) for x in v if str(x).strip()]
    return out


def ask(task, payload, budget=25.0, prefer=None, db=None, caller=None):
    """Run a task on whichever provider can. Returns (reply, provider).

    Nothing above this names a model, and nothing below it decides policy: this
    picks who is capable, and the gate decides who is allowed.

    THE OUTWARD CHECK LIVES HERE, not in the callers. It used to be in
    gate.answer_or_search, which meant every other caller reached the same
    providers without one -- consolidation would have sent a day of somebody's
    speech to a hosted model with no permission asked and nothing logged, and
    it took adding a provider that actually works to notice. A non-local
    provider now requires a db and a caller, and refuses without them, so a new
    caller cannot forget: forgetting is a refusal, not a leak.
    """
    released_via = None
    if prefer:
        # a deliberate, named override for a comparison run. It does not widen
        # what a provider claims to be able to do -- Gemini still declares only
        # reason and world -- it says "ask this one anyway, and say so".
        p = next((x for x in PROVIDERS if x.name == prefer), None)
        if p is None:
            raise NoProvider(f'no provider named {prefer!r}')
        ok, why = p.available()
        if not ok:
            raise NoProvider(f'{prefer} was asked for but: {why}')
        cap = TASKS.get(task, '?')
    else:
        p, cap = route(task)
    if not p.local:
        if db is None or caller is None:
            raise OutwardRefused(
                f"{p.name} would send this off the machine, and it was asked "
                f"without a gate to ask. Pass db= and caller=. Nothing was "
                f"sent.")
        import gate as g
        leaving = outward_text(task, payload)
        ok, why = g.release_outward(db, caller, p.name,
                                    f"[{task}] {leaving[0][:120] if leaving else ''}",
                                    extra=leaving[1:])
        if not ok:
            raise OutwardRefused(why)
        released_via = why

    t = time.perf_counter()
    reply = p.run(task, payload, budget)
    return reply, {'provider': p.name, 'capability': cap, 'local': p.local,
                   'costs': p.costs, 'forced': bool(prefer),
                   'released': released_via,
                   'seconds': round(time.perf_counter() - t, 3)}
