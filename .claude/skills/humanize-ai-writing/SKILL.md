---
name: humanize-ai-writing
description: Use this skill whenever the user wants writing edited so it reads as natural and human rather than AI-generated — e.g. "humanize this", "make this sound less like AI/ChatGPT", "remove the AI tells", "this sounds robotic, can you fix it", "polish this before I publish it", or any request to proofread/edit a draft that turns out to be stiff, generic, or formulaic. Also trigger when the user asks what makes writing "sound like AI" or wants a draft audited against Wikipedia's "Signs of AI Writing" patterns (em dashes, rule-of-three lists, vague attribution, stock phrases like "in conclusion", etc.). Applies to blog posts, emails, essays, marketing copy, LinkedIn posts, reports, and any other prose — not code.
---

# Humanize AI Writing

## What this is for

Large language models default to a recognizable set of habits: certain
punctuation choices, certain rhetorical crutches, certain stock phrases,
certain formatting reflexes. None of these habits is wrong in isolation —
humans use em dashes and triplets too — but AI text leans on them so
consistently that the accumulation is what reads as "AI-generated," flat,
or generic.

This skill catalogs those habits (drawn from Wikipedia's crowd-sourced
["Signs of AI Writing"](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)
guide, maintained by WikiProject AI Cleanup editors who catch undisclosed
AI content) and gives a process for editing them out of a piece of text
while preserving its actual content — the facts, the argument, the
citations, the voice the user is going for.

**The goal is not to hit zero matches on a checklist.** A piece of writing
can be entirely free of every pattern below and still be bland, and it can
legitimately use an em dash or a three-item list without being a tell. The
point is that AI text overuses these constructions *reflexively*, in every
paragraph, regardless of whether they're the best choice — so the fix is
to make each one a deliberate choice again, not to ban them outright.

## Workflow

1. **Read the whole draft first.** Get a sense of the actual content —
   what it's arguing, what facts it's conveying — before touching style.
   Never let pattern-hunting cause you to drop or distort real information,
   citations, numbers, code, or technical terms.

2. **Scan against the pattern catalog below**, section by section. For a
   short piece (a paragraph, an email), just do this silently and move to
   rewriting. For a longer piece, or if the user explicitly asks for an
   audit/review rather than a rewrite, list what you found: quote the
   exact phrase, name the pattern, and note where it is.

3. **Rewrite with the patterns removed**, not just deleted. Cutting "In
   conclusion, it is clear that X plays a vital role" down to "X matters"
   is usually a better fix than cutting it to nothing — the sentence
   still needs to do its job. Where the text was hiding a lack of real
   content behind a formulaic construction (a vague attribution, a false
   range, a tacked-on "highlighting the significance of Y"), that's a
   signal to either find something concrete to say or cut the sentence
   entirely, not just rephrase the padding into different padding.

4. **Vary sentence structure and length on purpose.** AI writing tends
   toward a uniform rhythm — similar sentence lengths, similar openings,
   similar paragraph shapes. Reading the rewrite aloud (mentally) is a
   decent test: real writers front-load some sentences, trail off others,
   start with "But" or "And" sometimes, and don't resolve every paragraph
   with a neat summary.

5. **Deliver both the clean rewrite and, if useful, a short note on what
   changed and why** — especially for edge cases where you kept something
   that matches a pattern below because it genuinely was the best
   phrasing in context. Don't just silently comply with every rule; use
   judgment the same way a good human editor would.

## Pattern catalog

### Wording to reconsider

AI models reach for a small, recognizable vocabulary far more often than
typical human writing does. None of these words are banned — "crucial"
is sometimes exactly the right word — but if several of them show up in
one piece, that's a tell worth addressing:

delve, intricate, tapestry, pivotal, underscore(s), landscape (as in "the
X landscape"), foster, testament, boast(s), elevate, realm, navigate (as
in "navigate challenges"), robust, seamless, cutting-edge, multifaceted,
holistic, leverage (as a verb), unlock, unleash, game-changer, dive into,
dive deeper, at the end of the day, in today's fast-paced world, it goes
without saying.

Fix: replace with a plainer, more specific word, or cut the sentence's
claim down to what's actually true and concrete.

### Sentence-level constructions

- **Negative parallelism ("It's not X, it's Y")** — e.g. "It's not just a
  product launch, it's a paradigm shift." A legitimate contrast device
  that AI overuses as a reflex for adding drama to a plain statement.
  Fix: state the actual point plainly, or use the construction only where
  a real, meaningful contrast exists.

- **Rule-of-three lists** — reflexive triplets in adjectives ("innovative,
  transformative, and groundbreaking"), benefits, or takeaways, used even
  when one item or a differently-sized list would fit better. Fix: ask
  whether there really are exactly three distinct things to say; if not,
  use the number that's actually true.

- **False ranges** — "from intimate gatherings to global movements," "our
  services range from strategic planning to implementation." Sounds like
  it's describing a spectrum but is really just two loosely related
  things dressed up to sound comprehensive. Fix: name the actual scope,
  or cut it.

- **Overemphasis on significance** — "plays a vital/pivotal role," "serves
  as a testament to," "stands as a reminder that," attached to facts that
  don't carry that much weight. Fix: state the fact and let it carry its
  own weight, or explain the actual mechanism of why it matters instead
  of asserting that it matters.

- **Superficial tacked-on analysis** — a plain fact followed by
  "highlighting the shift," "underscoring the importance," "illustrating
  the impact," where the analysis is just a label restating the fact
  rather than adding insight. Fix: either explain the real implication or
  drop the tag entirely.

- **Vague attribution** — "studies show," "experts say," "observers have
  noted," "some critics argue," with no name attached. Fix: name the
  actual source, or if there isn't one, don't imply there is.

- **Editorializing asides** — "it's important to note that," "no
  discussion would be complete without," inserting a value judgment about
  what matters into text that's supposed to be reporting facts. Fix: cut
  the aside; if the point is actually important, just make it directly.

### Discourse habits

- **Compulsive summaries** — "In summary," "Overall," "In conclusion,"
  restating what was just said even in a short passage that doesn't need
  a recap. Fix: cut the summary sentence, or if a wrap-up is genuinely
  useful (long document), make it add something rather than repeat.

- **Leftover conversational residue** — "I hope this helps!", "Let me
  know if you need anything else," "Certainly! Here's...", "Of course!" —
  phrases that belong to a chat exchange, not the document itself. Fix:
  delete outright; these should never survive into a final draft.

- **Letter-style phrasing in non-letter content** — "I hope this message
  finds you well" dropped into a blog post or report where it doesn't
  belong. Fix: cut, or replace with content-appropriate framing.

### Formatting habits

- Heavy, inconsistent **boldface** on terms that don't need emphasis.
- Bullet lists formatted as "**Term:** definition" where a sentence would
  read more naturally, or numbered lists used for things that aren't
  actually sequential.
- **Title-case headings** applied inconsistently, or headings that are
  themselves generic ("Conclusion," "The Bottom Line") rather than
  specific to the content under them.
- Emoji used as header decoration or bullet markers in contexts where
  that register doesn't fit the rest of the piece.

Fix: use formatting because it helps the reader scan real structure, not
as a default texture. A well-argued paragraph is often better than a
list.

### Punctuation

- **Em dash overkill** — reaching for an em dash for emphasis in spots
  where a comma, colon, or parenthesis reads more naturally, especially
  when it happens multiple times per paragraph. A single well-placed em
  dash is fine; several per paragraph is the tell.
- Skipping the **en dash** for ranges (dates, scores) and using a hyphen
  instead — "1990-2000" instead of "1990–2000." Minor, but a real
  divergence between typical AI and human typesetting.

## When NOT to flag something

- A single instance of any one pattern is usually not worth flagging on
  its own — humans write this way too sometimes. The signal is
  *accumulation*: several different patterns, or the same one repeatedly,
  in a short span of text.
- Don't strip real content to chase a lower pattern count. If removing a
  triplet would remove actual information (three genuinely distinct
  causes, say), keep it.
- Technical writing, legal text, and other genres with their own
  established conventions (numbered lists, defined terms in bold) should
  be judged against those conventions, not this general-purpose catalog.
- If the user's own natural voice includes one of these habits — they
  always use em dashes, say — respect that unless they've asked you to
  change it specifically.

## Output

For a short piece: give the rewritten text directly, ready to use.

For a longer piece or an explicit audit request: give a brief list of
what was flagged (quote, pattern name, one-line reason), then the full
rewritten text below it. Keep the flagged list short and skimmable —
it's there to show the user what changed, not to relitigate every
sentence.

Don't add a "Note: none of these signs alone prove AI authorship" caveat
to every response — that belongs in this skill file, not in the user's
finished document.
