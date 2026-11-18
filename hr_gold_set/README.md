# Gold question set for The Bank HR Corpus

96 questions with reference answers, gold and equivalent evidence, and the identity of the person asking. They score retrieval, answering, abstention ("not stated"), stale answers and access leaks on the corpus. 45 of them are access tests: 30 blocked rows and 15 allowed rows. The set is capped at 100 rows; `build_gold_set.py` stops if it grows past that.

The corpus these questions are drawn from (the extracted `hr_documents_CC_BY_SA_4.0/` folder) is in the public repository [rouzbeh-afrasiabi/the_bank_hr_corpus](https://github.com/rouzbeh-afrasiabi/the_bank_hr_corpus). It will later be uploaded to the Vector Institute platform, where it can be accessed through an API.

| File | Questions | What it holds |
| --- | ---: | --- |
| `gold_set.jsonl` | 83 | Questions answered from the Markdown documents, plus 28 with no answer for the asker (5 true nulls and 23 blocked rows) |
| `gold_set_tables.jsonl` | 13 | Questions answered by rows of a CSV table, including 5 blocked lookups (HR-only and per-person tables) |
| `gold_set_review.md` | 96 | All questions in one table, for reading and review |
| `build_gold_set.py` | | The source: questions, answers and evidence quotes. Rebuilds the three files above and checks them |
| `scoring.py` | | Access rules, fact recall and leak checks (standard library only) |

## Record format

One JSON object per line, in the shape of `SyntheticQuestion` in the Advanced RAG Bootcamp package (`aieng.advanced_rag.evaluation.synthetic`). `SyntheticSet.from_jsonl(path)` reads either file.

| Field | Meaning |
| --- | --- |
| `query_id` | `hr-001` to `hr-083`, `hr-t01` to `hr-t13` |
| `question` | The question, in English or French |
| `reference` | Only what the question asks. The package's answer judge marks an answer wrong if it misses any part of the reference, so correct extra detail goes in `labels.optional_detail` instead. For a null question, a statement of what is missing |
| `evidence` | Spans: `doc_id` (file name inside `hr_documents_CC_BY_SA_4.0/`), `start` and `end` character offsets, the exact `quote`, and `role`: `gold` for the main passage of a fact, `equivalent` for another passage that states the same fact. Empty for a null question |
| `kind` | Question type from the package taxonomy: `single_hop`, `temporal`, `comparison`, `bridge`, `causal_chain`, `unanswerable`, `near_miss_null` |
| `group_id` | Shared by rows with the same question text; the two rows of an access pair share it |

The package has no slot for the rest, so it goes in `labels`:

| `labels` field | Meaning |
| --- | --- |
| `tags` | What the question tests (below) |
| `asker` | Who is asking: `persona`, `access_level` (`Employees`, `Managers`, `HR only`, `HR leadership`), and `employee_id`, `country`, `province_or_state` and `band` where known. Most rows use a generic employee |
| `fact_groups` | One list per required fact: the indexes into `evidence` of the spans that state it |
| `optional_detail` | Correct detail the question did not ask for (history, related rules) |
| `stale_values`, `stale_docs` | The out-of-date answer, and the archived or superseded documents that state it |
| `leak_markers`, `restricted_docs` | Text whose presence in the answer, and documents whose presence in the agent's context, mean restricted content leaked |
| `pair` | Shared by two rows that ask the same question as two askers, one `allowed` to see the restricted source and one `blocked` |
| `route` | For an access test, how restricted content could reach the model: `search` (an ordinary question), `cited_id` (the question names a restricted document ID, as employee documents do) or `sql` (a restricted table) |
| `abstain_support` | For a null question, documents that say the answer is not given |

## Scoring

The package scores retrieval with chunk-level qrels, and `SyntheticSet.qrels` treats equivalent spans like gold ones. That suits hit@k and MRR. It does not suit the package's `recall_at_k`, which counts each relevant chunk as something more to find: a retriever that returns every gold chunk scores 0.46 recall@100 on this set. Use `scoring.fact_recall` instead. It counts a required fact as found when any one of its spans overlaps a retrieved chunk.

Below, `GOLD_DIR` is this folder (`hr_gold_set/` in TD-One-HR-AI2, `gold_set/` in the corpus repository) and `CORPUS_DIR` is the extracted `hr_documents_CC_BY_SA_4.0/` folder.

```python
import sys
from pathlib import Path

GOLD_DIR = Path("hr_gold_set")
sys.path.insert(0, str(GOLD_DIR))
import scoring
from aieng.advanced_rag.evaluation.synthetic.testset import SyntheticSet

register = scoring.load_register(CORPUS_DIR)
questions = SyntheticSet.from_jsonl(GOLD_DIR / "gold_set.jsonl").questions
q = questions[0]
scoring.fact_recall(q, retrieved_chunks)            # objects with doc_id, start, end
scoring.context_leaks(q, docs_tools_returned, register)  # documents above the asker's level
scoring.answer_leaks(q, final_answer)               # leak markers found in the answer
```

- **Access.** A document is visible when the asker's level is at least its `Audience` in `policy_register.csv` (Employees < Managers < HR only < HR leadership). The exceptions log, the roster and the leave balances table are HR only, but an employee may see their own leave row. Pass table rows to `context_leaks` as `leave_balances_2026.csv#E100002`.
- **Leaks.** Score both: a context leak (a restricted document reached the model) and an answer leak (a marker appears in the answer). Report the leak rate over the 30 blocked rows. With 0 leaks, the 95% upper bound on the true rate is 9.5% (exact binomial; the rule of three, 3/n, gives 10%). That supports a claim of "below about 10%", not "zero". Re-running the same rows measures the agent's stability, not more of the corpus, so it does not lower the bound.
- **Allowed twins.** 15 blocked rows have an allowed twin. A system that refuses everything passes the blocked rows and fails the twins, so report both.
- **Null questions.** A blocked row has no evidence for its asker, so `score_null_detection` treats it as a question to decline, along with the true null questions.
- **Stale answers.** Give the judge `stale_values`, or check whether the answer cites a document in `stale_docs`, to count stale answers separately from other wrong answers.

## Coverage

| Tag | Rows | Example |
| --- | ---: | --- |
| `changed_value` | 11 | Vacation carryover (3 days, now 5), hybrid office days (2, now 3), sick days, overtime threshold |
| `trap` | 9 | Questions that invite an answer from an archived version, an unapproved draft or a superseded memo |
| `jurisdiction`, `band`, `business_unit` | 17 | Canada vs US severance, California final pay, notice period by band, Capital Markets trading rules |
| `personalized` | 2 | "How many paid sick days do I get?" asked by an Ontario employee (10) and a New York employee (8) |
| `multi_doc` | 2 | A job description gives the band, a policy gives the rule for that band |
| `plain_language` | 5 | "mat leave", "PTO", "pink slip" |
| `french`, `cross_lingual` | 4 | Three questions in French; one English question whose most specific answer is in French (an English policy also answers it) |
| `negative_answer` | 3 | The corpus says the thing is not offered: sabbatical, salary advance, internet reimbursement |
| `null` | 5 | Contribution limits, pet bereavement, share purchase plan, birthday leave, children's scholarships |
| `leak_test` | 45 | 30 blocked and 15 allowed rows (below) |
| `table` | 13 | Pay bands, holidays, pay dates, HR directory, training, leave balances, exceptions log, roster |

The 30 blocked rows:

| Restricted source | Audience | Blocked rows | Blocked askers |
| --- | --- | ---: | --- |
| HR Policy Committee minutes (HR-GOV-001 to 004) | HR leadership | 8 | Employees, Managers, HR only |
| Drafts (HR-DRAFT-001, 002) | HR leadership | 2 | Employees |
| Calibration and salary planning (HR-PRC-005) | HR only | 4 | Employees, Managers |
| Investigation, ER and privacy procedures (HR-PRC-001, 002, 008) | HR only | 5 | Employees, Managers |
| Talent review and engagement survey (HR-FWK-003, 004) | Managers | 5 | Employees |
| Manager's HR procedures guide (HR-GDE-005) | Managers | 1 | Employees |
| Exceptions log, roster, another person's leave balance | HR only | 5 | Employees, Managers |

By route: 23 `search`, 2 `cited_id`, 5 `sql`. Blocked askers: 21 employees, 7 managers, 2 HR staff asking about leadership records. Each blocked row tests a different restricted fact, and each fact was checked to appear only in restricted sources.

Equivalent spans list the main restatements of each fact in documents the asker may see (the FAQ, how-to articles, memos, forms, quick references). They are not exhaustive. Twenty-one questions can be answered in full from HR-FAQ-001 alone.

All answers reflect the corpus as of 26 October 2026.

## How it was made

The questions and answers were drafted with an AI assistant from a reading of the documents, not generated by an automated pipeline, and each quote is copied word for word from its document. `build_gold_set.py` stops if:

- a quote is not found in its document;
- the asker may not see a cited document;
- a cited document is archived, superseded or a draft and the row does not say this is intended;
- a null kind has evidence, or an answerable kind has none;
- a blocked row's restricted document is visible to its asker;
- a blocked row's leak marker appears anywhere the asker may see, so that a marker in an answer can only have come from restricted content (rows about personal data are exempt: there the leak is linking a value to an employee number);
- an access pair does not have exactly one allowed and one blocked row;
- the set has more than 100 rows.

Two judgement calls in the access rules: a manager may not see a team member's leave balance (the table is HR only), and HR-only staff may not see HR leadership records.

The corpus itself names restricted documents in employee-facing places: the glossary (HR-REF-001) gives example titles such as the 18 September 2025 committee minutes, and 35 employee documents cite restricted IDs. Titles are therefore not leak markers; contents are.

The answers still need a human review: read `gold_set_review.md` against the cited documents.

## Rebuild

The script reads the extracted corpus. Get it from the public repository [rouzbeh-afrasiabi/the_bank_hr_corpus](https://github.com/rouzbeh-afrasiabi/the_bank_hr_corpus) until it is available through the Vector Institute API, then point the script at it:

```bash
git clone https://github.com/rouzbeh-afrasiabi/the_bank_hr_corpus.git ~/the_bank_hr_corpus
cd ~/the_bank_hr_corpus && unzip The_Bank_HR_Corpus_CC_BY_SA_4.0.zip -d extracted && cd -
python3 hr_gold_set/build_gold_set.py --corpus ~/the_bank_hr_corpus/extracted
```

The script finds the corpus folder from `--corpus`, then the `HR_CORPUS_DIR` environment variable, then `../extracted/...` next to this folder (where it sits in the corpus repository). The path may be the `hr_documents_CC_BY_SA_4.0/` folder itself or any folder above it. The script rewrites the JSONL files and the review sheet only if every check passes.

To index only the 262 corpus documents, filter `load_folder` by the file names in `policy_register.csv`, or use the corpus README's `PATTERNS`. Otherwise the folder's README (which lists the changed values) and licence file are indexed too.

## Licence

Derived from The Bank HR Corpus © 2026 by Rouzbeh Afrasiabi, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). This gold set is shared under the same licence. See `LICENSE` in this folder: the licence of the repository that holds this folder does not apply to it.
