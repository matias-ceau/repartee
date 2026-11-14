# Sample persona: Camille Martin

**Everything in this folder is fictional.** Camille Martin, her family,
friends, colleagues, chats, journal, essays and medical notes are invented for
testing and demonstrating Repartee. They are not based on a real person.

This folder is a complete Repartee home. Point Repartee at it with
`--home examples/persona` (or copy it somewhere first, so the pipeline's
outputs do not land in the code repository).

| Path | What it is |
|---|---|
| `FACTS.md` | The persona bible: biography, people, timeline 2019–2025, stances (one changes over time), tastes, behaviour patterns and measurable style rules. Every other file agrees with it. It is documentation, not a source. |
| `repartee.yaml` | Home config: owner `Camille Martin` (aliases `Camille`, `Cam`) and three sources. `git.autocommit` is off because this home lives inside the code checkout. |
| `sources/whatsapp/` | Four chat exports in four formats: iOS French from Léa's phone (`Léa Fontaine/_chat.txt`), iOS English (UK) group from Priya's phone (`Copenhagen crew/_chat.txt`), Android French 24-hour from her mother's phone (`Discussion WhatsApp avec Maman.txt`), Android English (US) 12-hour from Camille's phone (`WhatsApp Chat with Sam Whitaker.txt`). Each has system lines, media placeholders, a deleted message, an edited marker and multi-line messages, in the exporting phone's language. |
| `sources/vault/` | Obsidian vault: `Journal/` (15 dated entries, French with some English), `Essays/` (3 English essays), `Health/` (3 medical notes), people/work/project notes, and `Templates/` (excluded by config). |
| `sources/tastes/` | `music.csv` (artist, track, plays, year) and `books.md`. |
| `interviews/2025-06-01-childhood.md` | An interview transcript in Repartee's internal format. |
| `recall.yaml` | 25 recall questions with expected facts and the files that contain them. |
