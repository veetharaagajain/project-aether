"""The cases the grounding checks must get right, as runnable tests.

These existed only as descriptions in a chat, which is exactly how a check
loses the reason it was built. Each one is a triple of what a model returned --
the answer, and the line it claimed to have taken it from -- against the lines
it was actually shown, so the guards can be run on it directly without a model
in the loop. They are deterministic and take milliseconds.

Two halves, and both are load-bearing. The adversarial cases are inventions
that must be refused. The legitimate cases are the composed, rephrased answers
this system exists to produce, and they must be accepted -- without them a
sweep just recommends the strictest setting, and a guard that refuses honest
paraphrase is not a fix.

run: python tests/test_grounding.py     (exits non-zero on any failure)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import recall as rc                                          # noqa: E402

# Lines now carry a date and a speaker, which changes what the guards can see:
# a name or a date in a prefix is now part of the haystack answer_overlap reads,
# so an invented relationship using a real speaker's name, or an invented date
# that happens to appear on some other line, can now look supported. The last
# two adversarial cases exist for exactly that and did not need to before.
#
# what the model returned, against the lines it was shown. MUST be refused.
ADVERSARIAL = [
    dict(name='invented answer, real supporting line',
         lines=["I've been going to the beach most weekends this summer.",
                "The water was freezing but I went in anyway."],
         answer="You go to the beach every weekend with your brother Tom.",
         support="I've been going to the beach most weekends this summer.",
         why="the brother is invented; the quoted line is real"),
    dict(name='invented answer, invented supporting line',
         lines=["I've been going to the beach most weekends this summer."],
         answer="You bought a surfboard in April.",
         support="I bought a surfboard in April.",
         why="neither the answer nor the line it cites was ever said"),
    dict(name='half invented, attached to a true one',
         lines=["I went for a morning jog in the park yesterday.",
                "Then I taught two yoga classes."],
         answer="You went for a morning jog in the park and then drove to Bristol.",
         support="I went for a morning jog in the park yesterday.",
         why="the jog is true, the drive to Bristol is not. This one passed "
             "every whole-answer threshold ever shipped: it scores 0.71 "
             "because the true half carries it. Only the per-clause check "
             "catches it."),
    dict(name='general knowledge, real supporting line',
         lines=["I've been going to the beach most weekends this summer.",
                "The water was freezing but I went in anyway."],
         answer="The sea is cold in summer because of upwelling and thermal inertia.",
         support="The water was freezing but I went in anyway.",
         why="answered from what the model knows, citing a real line"),
    dict(name='invented relationship using a real speaker name',
         lines=["[15 May 2023] Caroline: I've been going to the beach most weekends.",
                "[22 May 2023] Melanie: The water was freezing but I went in."],
         answer="You go to the beach with your sister Caroline.",
         support="[15 May 2023] Caroline: I've been going to the beach most weekends.",
         why="Caroline is real and is a speaker; the sister is invented. Before "
             "lines carried names this could not have happened, because the "
             "name was not in the haystack at all"),
    dict(name='invented date lifted from another line',
         lines=["[15 May 2023] Jon: I started a new job.",
                "[22 June 2023] Jon: I'm reading The Lean Startup."],
         answer="You started The Lean Startup on 15 May 2023.",
         support="[22 June 2023] Jon: I'm reading The Lean Startup.",
         why="the date is real but belongs to a different line; every word of "
             "the answer appears somewhere in the haystack"),
]

# the answers this system is designed to produce. MUST be accepted.
LEGITIMATE = [
    dict(name='rephrased, as compose_from is meant to',
         lines=["I had a sausage croissant for breakfast."],
         answer="You had a sausage croissant.",
         support="I had a sausage croissant for breakfast."),
    dict(name='rephrased across two lines',
         lines=["The beach is a great place for finding peace.",
                "I also like sitting by the window in my Mom's house."],
         answer="The beach and the window seat at your Mom's house.",
         support="The beach is a great place for finding peace."),
    dict(name='inflected, which is why stemming exists',
         lines=["KKC conducted the audit last March."],
         answer="KKC audited it in March.",
         support="KKC conducted the audit last March."),
    dict(name='a true compound sentence',
         lines=["I went for a morning jog in the park yesterday.",
                "Then I taught two yoga classes."],
         answer="You went for a morning jog and then taught two yoga classes.",
         support="I went for a morning jog in the park yesterday."),
    dict(name='answered from the date on the line, which is the point',
         lines=["[15 May 2023] Jon: I'm currently reading The Lean Startup."],
         answer="You started The Lean Startup around May 2023.",
         support="[15 May 2023] Jon: I'm currently reading The Lean Startup."),
    dict(name='answered from the speaker on the line',
         lines=["[15 May 2023] Caroline: I went to the LGBTQ support group.",
                "[15 May 2023] Melanie: That sounds good."],
         answer="Caroline went to the LGBTQ support group.",
         support="[15 May 2023] Caroline: I went to the LGBTQ support group."),
]


# Cases that get through and are not fixed. Kept here rather than deleted,
# because a hole nobody wrote down is a hole nobody remembers. They are
# reported separately and do not fail the suite, so the suite stays honest
# about what it does and does not cover.
KNOWN_GAPS = [
    dict(name='relation invented across a released window',
         lines=["[18 August 2026] owner: >> I had toast.",
                "[18 August 2026] owner:    Sarah had the flu that week.",
                "[18 August 2026] owner:    It rained all afternoon."],
         answer="Sarah had toast.",
         support="[18 August 2026] owner:    Sarah had the flu that week.",
         why="Every word was said; only the relation between them is invented, "
             "so overlap is 1.00. Reachable since a fragment began releasing "
             "its whole window, because the haystack is now several "
             "neighbouring utterances rather than one. No lexical rule "
             "separates it from legitimate rephrasing across two lines, which "
             "is also drawing words from more than one line -- that case "
             "scores 0.20 against its own cited line, lower than this one's "
             "0.50, so a per-support threshold would refuse the honest case "
             "and admit this one. Needs entailment, or a citation per claim."),
]


def verdict(case):
    """Would compose_from emit this. Returns (accepted, why)."""
    joined = "\n".join(case['lines'])
    if rc.grounded(case.get('support', ''), case['lines']) is None:
        return False, 'support not grounded'
    ov = rc.answer_overlap(case['answer'], joined)
    if ov < rc.ANSWER_MIN_OVERLAP:
        return False, f'whole-answer overlap {ov:.2f}'
    weak = rc.weakest_clause(case['answer'], joined)
    if weak and weak[1] < rc.CLAUSE_MIN_OVERLAP:
        return False, f'clause {weak[0][:34]!r} scores {weak[1]:.2f}'
    stray = rc.date_tokens(case['answer']) - rc.date_tokens(case.get('support', ''))
    if stray:
        return False, f'date {"/".join(sorted(stray))} not on the cited line'
    return True, f'overlap {ov:.2f}'


def main():
    bad = 0
    print("must be REFUSED:")
    for c in ADVERSARIAL:
        ok, why = verdict(c)
        flag = 'LET THROUGH' if ok else 'refused'
        if ok:
            bad += 1
        print(f"  [{'FAIL' if ok else 'pass'}] {c['name']:<44} {flag}: {why}")
    print("\nmust be ACCEPTED:")
    for c in LEGITIMATE:
        ok, why = verdict(c)
        if not ok:
            bad += 1
        print(f"  [{'pass' if ok else 'FAIL'}] {c['name']:<44} "
              f"{'accepted' if ok else 'REFUSED'}: {why}")
    print("\nknown gaps, not fixed and not counted as failures:")
    for c in KNOWN_GAPS:
        ok, why = verdict(c)
        state = 'still gets through' if ok else 'NOW CAUGHT -- promote it'
        print(f"  [gap ] {c['name']:<44} {state}: {why}")
        print(f"         {c['why']}")
    print(f"\n{len(ADVERSARIAL) + len(LEGITIMATE)} cases, {bad} failure(s), "
          f"{len(KNOWN_GAPS)} known gap(s)")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
