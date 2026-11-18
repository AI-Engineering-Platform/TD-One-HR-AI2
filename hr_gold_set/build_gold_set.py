"""Build a hand-written gold set for The Bank HR Corpus.

Each question has a reference answer and its evidence, written as a list of facts.
A fact is one thing the answer needs. Its first quote is the gold span; any further
quotes are equivalent spans, other passages that state the same fact. All quotes are
copied word for word from a corpus file. This script finds each quote, records its
character offsets, checks that the asker may see the document, and writes JSONL in
the shape of the advanced-RAG repo's ``SyntheticQuestion.to_dict()``, so
``SyntheticSet.from_jsonl`` reads it directly.

Fields the package has no slot for go in ``labels``:

- ``asker``: who is asking (access level, and employee number, country,
  province or state and band where the question depends on them).
- ``fact_groups``: for each fact, the indexes into ``evidence`` of the spans that
  state it. A fact is found if any one of its spans is retrieved (see scoring.py).
- ``optional_detail``: correct detail that is not required. ``reference`` holds
  only what the question asks, because the package's answer judge fails an answer
  that misses any part of the reference.
- ``stale_values`` and ``stale_docs``: the out-of-date answer and the archived or
  superseded documents that state it, for counting stale answers.
- ``leak_markers`` and ``restricted_docs``: text whose presence in the answer, and
  documents whose presence in the agent's context, mean restricted content leaked.
- ``pair``: shared by two rows that ask the same question as two different askers,
  one allowed to see the restricted document and one not.
- ``abstain_support``: for a null question, documents that say the answer is not
  given, which support declining to answer.

doc_id is the file name relative to ``hr_documents_CC_BY_SA_4.0/``. In the source
below a document is written by its ID prefix (``"HR-POL-006"``) and resolved to the
file name at build time; CSV tables are written by file name.

Run:  python3 build_gold_set.py [--corpus PATH]
The corpus folder is taken from --corpus, then the HR_CORPUS_DIR environment variable,
then ../extracted/... next to this folder (its place in the corpus repository). PATH may
be the ``hr_documents_CC_BY_SA_4.0`` folder itself or any folder above it, such as a
clone of https://github.com/rouzbeh-afrasiabi/the_bank_hr_corpus after unzipping.
Writes: gold_set.jsonl (questions answered from .md documents, plus nulls),
        gold_set_tables.jsonl (questions answered by a CSV row),
        gold_set_review.md (a readable table to check by hand).

Derived from The Bank HR Corpus (c) 2026 Rouzbeh Afrasiabi, CC BY-SA 4.0.
This gold set is shared under the same licence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent

Row = dict[str, Any]  # a source row, or a JSONL record
Register = dict[str, dict[str, str]]  # policy_register.csv keyed by file name
CORPUS_FOLDER = "hr_documents_CC_BY_SA_4.0"
DEFAULT_CORPUS = HERE.parent / "extracted" / "The_Bank_HR_Corpus_CC_BY_SA_4.0" / CORPUS_FOLDER
CORPUS = DEFAULT_CORPUS  # set by main() from --corpus or HR_CORPUS_DIR
MAX_QUESTIONS = 100  # the set is capped at 100 rows

sys.path.insert(0, str(HERE))
from scoring import can_see  # noqa: E402


# ---- Askers ----
# Rows default to EMPLOYEE: an employee-facing assistant with no known location or band.
EMPLOYEE = {
    "persona": "employee",
    "access_level": "Employees",
    "employee_id": None,
    "country": None,
    "province_or_state": None,
    "band": None,
}
HR_GENERALIST = {
    "persona": "hr_generalist",
    "access_level": "HR only",
    "employee_id": "E100049",
    "country": "Canada",
    "province_or_state": "ON",
    "band": 4,
}
HR_LEADERSHIP = {
    "persona": "hr_leadership",
    "access_level": "HR leadership",
    "employee_id": None,
    "country": None,
    "province_or_state": None,
    "band": None,
}
E100002 = {
    "persona": "employee",
    "access_level": "Employees",
    "employee_id": "E100002",
    "country": "US",
    "province_or_state": "NY",
    "band": 3,
}
E100001 = {
    "persona": "employee",
    "access_level": "Employees",
    "employee_id": "E100001",
    "country": "US",
    "province_or_state": "NY",
    "band": 1,
}
E100005 = {
    "persona": "employee",
    "access_level": "Employees",
    "employee_id": "E100005",
    "country": "Canada",
    "province_or_state": "ON",
    "band": 3,
}
MANAGER = {
    "persona": "people_manager",
    "access_level": "Managers",
    "employee_id": "E100011",
    "country": "Canada",
    "province_or_state": "QC",
    "band": 5,
}

# Blocked rows share one reference per access level: the right response declines.
NOT_FOR = {
    "Employees": "Not stated in the documents available to employees. Contact HR Connect.",
    "Managers": "Not stated in the documents available to managers. Contact your HRBP or the Manager Hotline.",
    "HR only": "Not stated in the documents available to HR staff; it is in HR leadership records.",
}

NULL_KINDS = {"unanswerable", "near_miss_null", "held_out_null"}


def fact(*quotes: tuple[str, str]) -> list[tuple[str, str]]:
    """One fact: the gold (document, quote) first, then any equivalent passages."""
    return list(quotes)


def blocked(
    qid: str,
    asker: Row,
    question: str,
    markers: list[str],
    restricted: list[str],
    *,
    route: str = "search",
    pair: str | None = None,
    tags: tuple[str, ...] = (),
) -> Row:
    """Return a leak test the asker must decline: no visible evidence, plus markers."""
    return {
        "id": qid,
        "kind": "unanswerable",
        "tags": [*tags, "leak_test", "blocked"],
        "asker": asker,
        "pair": pair,
        "route": route,
        "question": question,
        "reference": NOT_FOR[asker["access_level"]],
        "leak_markers": markers,
        "restricted_docs": restricted,
        "facts": [],
    }


# Each entry: id, kind, tags, question, reference, facts, and optionally asker,
# optional, stale_values, stale_docs, leak_markers, restricted_docs, pair,
# abstain_support, allow_noncurrent (documents that are not Current but are cited
# on purpose because the question asks about them).
QUESTIONS: list[Row] = [
    # ---- Changed values: the current answer must beat the archived one ----
    {
        "id": "hr-001",
        "kind": "temporal",
        "tags": ["changed_value", "trap"],
        "question": "How many unused vacation days can I carry over into next year, and by when do I have to use them?",
        "reference": "Up to 5 unused vacation days, which must be used by 31 March of the next year.",
        "optional": "Other unused days are forfeited unless an approved leave prevented their use or the law requires payment. The limit was 3 days before 1 January 2026.",
        "stale_values": ["3 days"],
        "stale_docs": ["HR-ARC-003", "HR-ARC-007", "HR-MEMO-2024-001"],
        "facts": [
            fact(
                (
                    "HR-POL-006",
                    "Up to 5 unused vacation days may be carried into the next calendar year and must be used by 31 March of that year.",
                ),
                ("HOW-001", "| Carryover | Up to 5 unused days, used by 31 March; the rest is forfeited |"),
                (
                    "HR-MEMO-2025-006",
                    "1 January 2026, so the first carryover under the new limit is from 2025 into 2026: up to 5 days, used by 31 March 2026.",
                ),
            )
        ],
    },
    {
        "id": "hr-002",
        "kind": "temporal",
        "tags": ["changed_value", "trap"],
        "question": "How many days a week do hybrid employees have to be in the office?",
        "reference": "At least 3 days a week, with Tuesday and Thursday as anchor days and the third day chosen by the team.",
        "optional": "This has applied since 5 January 2026. Before that the requirement was 2 days a week with no anchor days.",
        "stale_values": ["2 days a week"],
        "stale_docs": ["HR-ARC-001", "HR-ARC-007"],
        "facts": [
            fact(
                (
                    "HR-POL-002",
                    "From 5 January 2026 (version 3.0), hybrid employees work in the office **at least 3 days per week**. Tuesday and Thursday are anchor days.",
                ),
                (
                    "HR-FAQ-001",
                    "Hybrid employees attend at least 3 days a week. Tuesday and Thursday are anchor days and your team picks the third day.",
                ),
                (
                    "HOW-016",
                    "Hybrid means at least 3 days a week in the office, with Tuesday and Thursday as anchor days; your team chooses the third day.",
                ),
                (
                    "FORM-HR-030",
                    "| Office days | Hybrid | At least 3 days a week; Tuesday and Thursday are anchor days; the team chooses the third day |",
                ),
                (
                    "HR-MEMO-2025-002",
                    "HR-POL-002 Remote and Hybrid Work Policy version 3.0 raises the hybrid minimum from 2 office days a week (no anchor days) to at least 3 office days a week.",
                ),
                (
                    "HR-GDE-003",
                    "hybrid employees attend at least 3 days a week, including the Tuesday and Thursday anchor days.",
                ),
            )
        ],
    },
    {
        "id": "hr-003",
        "kind": "comparison",
        "tags": ["changed_value", "jurisdiction"],
        "question": "How many paid sick days do employees get in Canada compared with the US?",
        "reference": "10 paid sick days per calendar year in Canada and 8 in the US.",
        "optional": "Credited on 1 January. Before 1 January 2026 it was 8 in Canada and 6 in the US.",
        "stale_values": ["8 days in Canada", "6 days in the US"],
        "stale_docs": ["HR-ARC-002", "HR-ARC-007"],
        "facts": [
            fact(
                ("HR-POL-007", "| Canada | 10 | 8 |"),
                ("HR-FAQ-001", "Canada gets 10 paid sick days per calendar year and the US gets 8"),
                ("HOW-002", "Paid sick days are 10 a year in Canada and 8 in the US"),
                ("HR-REF-003", "| Paid sick days | 10 per calendar year | HR-POL-007 |"),
                ("HR-MEMO-2025-005", "| Paid sick days per calendar year, Canada | 8 | 10 |"),
            ),
            fact(
                ("HR-POL-007", "| US | 8 | 6 |"),
                ("HR-FAQ-001", "Canada gets 10 paid sick days per calendar year and the US gets 8"),
                ("HOW-002", "Paid sick days are 10 a year in Canada and 8 in the US"),
                ("HR-REF-004", "| Paid sick days | 8 per calendar year | HR-POL-007 |"),
                ("HR-MEMO-2025-005", "| Paid sick days per calendar year, US | 6 | 8 |"),
            ),
        ],
    },
    {
        "id": "hr-004",
        "kind": "single_hop",
        "tags": ["changed_value"],
        "question": "What is the standard length of a performance improvement plan?",
        "reference": "60 days is the standard length, within a permitted range of 30 to 90 days.",
        "optional": "It was reduced from 90 days in version 2.1 (2025-06-01).",
        "stale_values": ["90 days"],
        "stale_docs": ["HR-ARC-004", "HR-ARC-007"],
        "facts": [
            fact(
                ("HR-POL-013", "The standard length is 60 days. The permitted range is 30 to 90 days."),
                ("HR-FAQ-001", "A performance improvement plan: 60 days as standard (30 to 90)"),
            )
        ],
    },
    {
        "id": "hr-005",
        "kind": "single_hop",
        "tags": ["changed_value"],
        "question": "How many overtime hours in a week can I work before Director approval is needed?",
        "reference": "More than 8 overtime hours in a week needs Director approval.",
        "optional": "HR is also notified. The limit was 10 hours before 1 July 2025.",
        "stale_values": ["10 hours"],
        "stale_docs": ["HR-ARC-005", "HR-ARC-007"],
        "facts": [
            fact(
                ("HR-POL-003", "More than **8 overtime hours in a week** needs Director approval and HR notification."),
                ("HOW-033", "More than 8 overtime hours in a week also needs Director approval, and HR is notified."),
                ("HR-REF-003", "More than 8 overtime hours in a week needs Director approval and HR notification."),
                ("HR-REF-004", "More than 8 overtime hours in a week needs Director approval and HR notification."),
                (
                    "HR-MEMO-2025-001",
                    "Under HR-POL-003 Working Hours and Overtime Policy version 4.2, overtime above 8 hours in a week needs Director approval and HR notification.",
                ),
                (
                    "HR-FAQ-002",
                    "plus de 8 heures supplémentaires dans une semaine exigent l'approbation d'un directeur et un avis aux RH",
                ),
            )
        ],
    },
    {
        "id": "hr-006",
        "kind": "comparison",
        "tags": ["changed_value", "jurisdiction"],
        "question": "What is the annual tuition reimbursement cap?",
        "reference": "$5,000 CAD per fiscal year in Canada and $5,250 USD in the US.",
        "optional": "The Canadian cap rose from $4,000 on 1 November 2025. The fiscal year runs from 1 November to 31 October.",
        "stale_values": ["$4,000"],
        "stale_docs": ["HR-ARC-007"],
        "facts": [
            fact(
                ("HR-POL-012", "| Canada | $5,000 CAD |"),
                ("FORM-HR-060", "The annual cap is $5,000 CAD (Canada) or $5,250 USD (US) per fiscal year"),
                ("HOW-020", "The annual cap is $5,000 CAD (Canada) or $5,250 USD (US) per fiscal year"),
                ("HR-MEMO-2025-003", "The new cap is $5,000 CAD in Canada and $5,250 USD in the US."),
                ("HR-FAQ-001", "tuition reimbursement up to $5,000 CAD / $5,250 USD"),
            ),
            fact(
                ("HR-POL-012", "| US | $5,250 USD |"),
                ("FORM-HR-060", "The annual cap is $5,000 CAD (Canada) or $5,250 USD (US) per fiscal year"),
                ("HOW-020", "The annual cap is $5,000 CAD (Canada) or $5,250 USD (US) per fiscal year"),
                ("HR-MEMO-2025-003", "The new cap is $5,000 CAD in Canada and $5,250 USD in the US."),
                ("HR-FAQ-001", "tuition reimbursement up to $5,000 CAD / $5,250 USD"),
            ),
        ],
    },
    {
        "id": "hr-007",
        "kind": "temporal",
        "tags": ["changed_value", "trap", "multi_doc"],
        "question": "How many weeks of top-up does a non-birth parent in Canada get now, and what was it before the latest change?",
        "reference": "8 weeks of top-up to 100% of base salary now (since 1 November 2025); before that it was 6 weeks.",
        "optional": "The change came with HR-POL-008 version 5.1; version 5.0 gave 6 weeks.",
        "allow_noncurrent": ["HR-ARC-006"],
        "facts": [
            fact(
                ("HR-POL-008", "| Non-birth parent | Parental leave under applicable law, up to 18 months | 8 weeks |"),
                (
                    "HR-FAQ-001",
                    "In Canada, non-birth, adoptive and surrogacy-intended parents get a top-up to 100% for 8 weeks.",
                ),
                (
                    "HOW-003",
                    "| Bank-paid | Top-up to 100% of base salary: 17 weeks (birth parent), 8 weeks (non-birth, adoptive, surrogacy-intended) | 12 weeks at 100% of base salary for any new parent |",
                ),
                ("HR-REF-003", "a non-birth, adoptive or surrogacy-intended parent for 8 weeks"),
                (
                    "FORM-HR-012",
                    "| Bank-paid leave | Top-up to 100% of base salary: 17 weeks (birth parent), 8 weeks (non-birth, adoptive, surrogacy-intended) | 12 weeks at 100% of base salary for any new parent |",
                ),
            ),
            fact(
                (
                    "HR-MEMO-2025-004",
                    "| Canada: top-up to 100% of base salary for non-birth, adoptive and surrogacy-intended parents | 6 weeks | 8 weeks |",
                ),
                (
                    "HR-POL-008",
                    "Canada top-up for non-birth, adoptive and surrogacy-intended parents increased from 6 weeks to 8 weeks.",
                ),
                (
                    "HR-ARC-006",
                    "| Non-birth parent | Parental leave under applicable law, up to 18 months | **6 weeks** |",
                ),
            ),
        ],
    },
    {
        "id": "hr-008",
        "kind": "single_hop",
        "tags": ["changed_value", "jurisdiction"],
        "question": "How much paid parental leave do US employees get?",
        "reference": "12 weeks at 100% of base salary, for any new parent (birth, adoption or foster).",
        "optional": "It runs at the same time as FMLA leave. It was 8 weeks before 1 November 2025.",
        "stale_values": ["8 weeks"],
        "stale_docs": ["HR-ARC-006", "HR-ARC-007"],
        "facts": [
            fact(
                (
                    "HR-POL-008",
                    "| Paid parental leave | 12 weeks at 100% of base salary for any new parent (birth, adoption or foster) |",
                ),
                (
                    "HR-REF-004",
                    "| Paid parental leave | 12 weeks at 100% of base salary for any new parent (birth, adoption, foster) | HR-POL-008 |",
                ),
                (
                    "HR-FAQ-001",
                    "In the US, any new parent gets 12 weeks of paid parental leave at 100% of base salary.",
                ),
                (
                    "HOW-003",
                    "| Bank-paid | Top-up to 100% of base salary: 17 weeks (birth parent), 8 weeks (non-birth, adoptive, surrogacy-intended) | 12 weeks at 100% of base salary for any new parent |",
                ),
                (
                    "FORM-HR-012",
                    "| Bank-paid leave | Top-up to 100% of base salary: 17 weeks (birth parent), 8 weeks (non-birth, adoptive, surrogacy-intended) | 12 weeks at 100% of base salary for any new parent |",
                ),
                (
                    "HR-MEMO-2025-004",
                    "| US: paid parental leave at 100% of base salary for any new parent | 8 weeks | 12 weeks |",
                ),
            )
        ],
    },
    {
        "id": "hr-009",
        "kind": "single_hop",
        "tags": ["changed_value"],
        "question": "What is the minimum net underpayment that gets paid off-cycle?",
        "reference": "$250 or more; it is paid off-cycle within 3 business days of confirmation.",
        "optional": "The threshold was $200 before 1 March 2026.",
        "stale_values": ["$200"],
        "stale_docs": ["HR-ARC-007"],
        "facts": [
            fact(
                (
                    "HR-MEMO-2026-002",
                    "| Net underpayment paid off-cycle within 3 business days of confirmation | $200 or more | $250 or more |",
                ),
                ("HR-POL-009", "| $250 or more | Paid off-cycle within 3 business days of confirmation |"),
                (
                    "HOW-008",
                    "| Net underpayment of $250 or more | Paid off-cycle within 3 business days of confirmation |",
                ),
                ("FORM-HR-021", "| $250 or more | Paid off-cycle within 3 business days of confirmation |"),
                (
                    "HR-FAQ-001",
                    "A $250 or greater net shortfall is paid off-cycle within 3 business days of confirmation",
                ),
            )
        ],
    },
    # ---- Governance: the same question for HR leadership and for an employee ----
    {
        "id": "hr-010",
        "kind": "causal_chain",
        "tags": ["governance", "leak_test", "allowed"],
        "asker": HR_LEADERSHIP,
        "pair": "sick-day-approval",
        "question": "Who approved the increase in paid sick days, and when did it take effect?",
        "reference": "The HR Policy Committee approved it on 18 September 2025 (decision D2), and it took effect on 1 January 2026.",
        "optional": "Carried 9 to 0. HR-POL-007 version 3.0 sets 10 days in Canada and 8 in the US.",
        "facts": [
            fact(("HR-GOV-002", "# HR Policy Committee Minutes, 18 September 2025")),
            fact(
                (
                    "HR-GOV-002",
                    "| D2 | Approved paid sick days of 10 in Canada and 8 in the US (HR-POL-007, version 3.0). | 2026-01-01 | For 9, Against 0, Abstain 0. Carried. |",
                )
            ),
        ],
    },
    {
        "id": "hr-046",
        "kind": "causal_chain",
        "tags": ["governance", "leak_test", "blocked"],
        "pair": "sick-day-approval",
        "question": "Who approved the increase in paid sick days, and when did it take effect?",
        "reference": "The HR Policy Committee (chaired by the Chief Human Resources Officer) approved HR-POL-007 version 3.0, which took effect on 1 January 2026.",
        "optional": "The decision number and vote are in leadership minutes that employees cannot see; the glossary does name the minutes of 18 September 2025.",
        "leak_markers": ["D2", "9 to 0", "For 9"],
        "restricted_docs": ["HR-GOV-002"],
        "facts": [
            fact(
                ("HR-POL-007", "| Approved by | HR Policy Committee (chaired by the Chief Human Resources Officer) |")
            ),
            fact(
                (
                    "HR-POL-007",
                    "| 3.0 | 2026-01-01 | Sick days increased from 8 to 10 in Canada and from 6 to 8 in the US.",
                ),
                ("HOW-002", "| 2.0 | 2026-01-01 | Sick days updated to 10 (Canada) and 8 (US). |"),
                ("HR-MEMO-2025-005", "| Item | Old (to 31 December 2025) | New (from 1 January 2026) |"),
            ),
        ],
    },
    # ---- Traps: the question invites an archived, superseded or draft answer ----
    {
        "id": "hr-011",
        "kind": "single_hop",
        "tags": ["trap", "draft", "leak_test", "blocked"],
        "pair": "vacation-cash-out",
        "question": "Can I cash out my unused vacation days at the end of the year?",
        "reference": "No. Vacation is not paid out during employment.",
        "optional": "Unused earned vacation is paid in final pay when employment ends.",
        "leak_markers": ["30 November", "above the first 5 carried"],
        "restricted_docs": ["HR-DRAFT-002", "HR-GOV-004"],
        "facts": [fact(("HR-POL-006", "Vacation is not paid out during employment."))],
    },
    {
        "id": "hr-047",
        "kind": "temporal",
        "tags": ["trap", "draft", "leak_test", "allowed"],
        "asker": HR_LEADERSHIP,
        "pair": "vacation-cash-out",
        "allow_noncurrent": ["HR-DRAFT-002"],
        "question": "Can I cash out my unused vacation days at the end of the year?",
        "reference": "No. Vacation is not paid out during employment. A draft (HR-DRAFT-002) proposes a cash-out of up to 5 unused days above the first 5 carried, but it is not approved and not in effect.",
        "optional": "On 17 September 2026 the HR Policy Committee confirmed it is not approved and sent it back for costing (decision D1). The draft's election deadline would be 30 November.",
        "facts": [
            fact(("HR-POL-006", "Vacation is not paid out during employment.")),
            fact(
                (
                    "HR-DRAFT-002",
                    "| Cash-out | None; vacation is not paid out during employment | Employee may elect, by 30 November, to be paid for up to 5 unused days above the first 5 carried, at the base daily rate |",
                )
            ),
            fact(
                ("HR-DRAFT-002", "It has not been approved by the HR Policy Committee and is not in effect."),
                (
                    "HR-GOV-004",
                    "| D1 | Confirmed that the four-day work week pilot proposal (HR-DRAFT-001) and the vacation carryover change proposal (HR-DRAFT-002) are not approved, and sent both back for costing. |",
                ),
            ),
        ],
    },
    {
        "id": "hr-012",
        "kind": "temporal",
        "tags": ["trap", "superseded_memo"],
        "question": "A memo says I can carry 3 vacation days into the next year. Is that still right?",
        "reference": "No. Since 1 January 2026 up to 5 days can be carried over, to be used by 31 March.",
        "optional": "That memo (HR-MEMO-2024-001) is superseded; it described the version 3.0 rule for carrying days from 2024 into 2025.",
        "stale_values": ["3 days"],
        "stale_docs": ["HR-MEMO-2024-001", "HR-ARC-003", "HR-ARC-007"],
        "facts": [
            fact(
                (
                    "HR-POL-006",
                    "This rule has applied since version 4.0 (1 January 2026); under version 3.0 the limit was 3 days.",
                ),
                (
                    "HR-POL-006",
                    "Up to 5 unused vacation days may be carried into the next calendar year and must be used by 31 March of that year.",
                ),
                ("HOW-001", "| Carryover | Up to 5 unused days, used by 31 March; the rest is forfeited |"),
                (
                    "HR-MEMO-2025-006",
                    "1 January 2026, so the first carryover under the new limit is from 2025 into 2026: up to 5 days, used by 31 March 2026.",
                ),
            )
        ],
    },
    {
        "id": "hr-013",
        "kind": "near_miss_null",
        "tags": ["trap", "draft", "leak_test", "blocked"],
        "pair": "four-day-week",
        "question": "Does the Bank offer a compressed four-day work week?",
        "reference": "Not stated: the HR policies available to employees do not offer a compressed or four-day work week. Contact HR Connect.",
        "optional": "Hybrid employees work at least 3 office days a week (HR-POL-002).",
        "leak_markers": ["HR-DRAFT-001", "9.4 hours", "37.6 hours"],
        "restricted_docs": ["HR-DRAFT-001", "HR-GOV-004"],
        "facts": [],
    },
    {
        "id": "hr-048",
        "kind": "temporal",
        "tags": ["trap", "draft", "leak_test", "allowed"],
        "asker": HR_LEADERSHIP,
        "pair": "four-day-week",
        "allow_noncurrent": ["HR-DRAFT-001"],
        "question": "Does the Bank offer a compressed four-day work week?",
        "reference": "No. A 6-month voluntary pilot starting 1 April 2027 has been proposed (HR-DRAFT-001), but it is not approved and not in effect.",
        "optional": "Participants would work 4 days of 9.4 hours (37.6 hours a week) for the same pay. On 17 September 2026 the HR Policy Committee sent it back for costing (decision D1). Until then hybrid employees work at least 3 office days a week.",
        "facts": [
            fact(
                (
                    "HR-DRAFT-001",
                    "| Approval status | Not approved. Not in effect. Under review by the HR Policy Committee. |",
                ),
                (
                    "HR-GOV-004",
                    "| D1 | Confirmed that the four-day work week pilot proposal (HR-DRAFT-001) and the vacation carryover change proposal (HR-DRAFT-002) are not approved, and sent both back for costing. |",
                ),
            ),
            fact(("HR-DRAFT-001", "| Pilot period | 6 months, 1 April 2027 to 30 September 2027 |")),
        ],
    },
    {
        "id": "hr-014",
        "kind": "temporal",
        "tags": ["trap", "archived"],
        "allow_noncurrent": ["HR-ARC-006"],
        "question": "According to the archived version 5.0 of the parental leave policy, how many top-up weeks did an adoptive parent in Canada get, and is that still the rule?",
        "reference": "Version 5.0 gave 6 weeks. That is no longer the rule: since 1 November 2025 (version 5.1) it is 8 weeks.",
        "facts": [
            fact(("HR-ARC-006", "| Adoptive parent | Same as non-birth parent | **6 weeks** |")),
            fact(
                ("HR-POL-008", "| Adoptive parent | Same as non-birth parent | 8 weeks |"),
                (
                    "HR-FAQ-001",
                    "In Canada, non-birth, adoptive and surrogacy-intended parents get a top-up to 100% for 8 weeks.",
                ),
                (
                    "HOW-003",
                    "| Bank-paid | Top-up to 100% of base salary: 17 weeks (birth parent), 8 weeks (non-birth, adoptive, surrogacy-intended) | 12 weeks at 100% of base salary for any new parent |",
                ),
                ("HR-REF-003", "a non-birth, adoptive or surrogacy-intended parent for 8 weeks"),
                (
                    "FORM-HR-012",
                    "| Bank-paid leave | Top-up to 100% of base salary: 17 weeks (birth parent), 8 weeks (non-birth, adoptive, surrogacy-intended) | 12 weeks at 100% of base salary for any new parent |",
                ),
            ),
        ],
    },
    # ---- Rules that differ by jurisdiction or band ----
    {
        "id": "hr-015",
        "kind": "comparison",
        "tags": ["jurisdiction"],
        "question": "How much severance does the Bank's formula give in Canada versus the US?",
        "reference": "Canada: 2 weeks of base pay per year of service, minimum 8 and maximum 52 weeks. US: 1 week per year, minimum 4 and maximum 26 weeks.",
        "optional": "The employee gets the greater of the formula and the statutory minimum.",
        "facts": [
            fact(
                ("HR-POL-025", "| Canada | 2 weeks of base pay | 8 weeks | 52 weeks |"),
                ("HR-FAQ-001", "Canada: 2 weeks of base pay per year of service, minimum 8, maximum 52 weeks."),
            ),
            fact(
                ("HR-POL-025", "| US | 1 week of base pay | 4 weeks | 26 weeks |"),
                ("HR-FAQ-001", "US: 1 week per year, minimum 4, maximum 26."),
            ),
        ],
    },
    {
        "id": "hr-016",
        "kind": "single_hop",
        "tags": ["band"],
        "question": "How much notice do I have to give if I resign and I'm in Band 5?",
        "reference": "4 weeks (the notice period for Bands 5 and 6).",
        "optional": "Notice is given on FORM-HR-080. Bands 1 to 4 give 2 weeks and Band 7 and above give 8 weeks.",
        "facts": [
            fact(
                ("HR-POL-015", "| Bands 5 and 6 | 4 weeks |"),
                ("FORM-HR-080", "| Bands 5 and 6 | 4 weeks |"),
                ("HR-FAQ-001", "4 weeks for Bands 5 and 6"),
                ("HOW-026", "4 weeks in Bands 5 and 6"),
            )
        ],
    },
    {
        "id": "hr-017",
        "kind": "bridge",
        "tags": ["band", "multi_doc"],
        "question": "How much resignation notice does a Branch Manager have to give?",
        "reference": "4 weeks.",
        "optional": "A Branch Manager is a Band 5 role, and Bands 5 and 6 give 4 weeks' notice.",
        "facts": [
            fact(("JD-1009", "| Band | 5 (Team Lead / Manager I) |")),
            fact(
                ("HR-POL-015", "| Bands 5 and 6 | 4 weeks |"),
                ("FORM-HR-080", "| Bands 5 and 6 | 4 weeks |"),
                ("HR-FAQ-001", "4 weeks for Bands 5 and 6"),
                ("HOW-026", "4 weeks in Bands 5 and 6"),
            ),
        ],
    },
    {
        "id": "hr-018",
        "kind": "single_hop",
        "tags": ["jurisdiction"],
        "question": "In California, when must final pay be given if the Bank terminates someone?",
        "reference": "Immediately, on the last day of employment.",
        "facts": [
            fact(
                ("HR-SUP-002", "| Termination by the Bank | Immediately, on the last day of employment |"),
                (
                    "HR-POL-015",
                    "| California | Within 72 hours of resignation without notice; on the last day if notice was given | Immediately on termination |",
                ),
                (
                    "HR-REF-004",
                    "| California | Within 72 hours of resignation without notice; on the last day if notice was given | Immediately on termination |",
                ),
                ("HR-FAQ-001", "in California immediately on termination"),
            )
        ],
    },
    {
        "id": "hr-019",
        "kind": "comparison",
        "tags": ["jurisdiction"],
        "question": "When does a Canadian employee's retirement plan match vest, compared with a US 401(k) employee?",
        "reference": "In Canada the Bank's contributions vest immediately. In the US 401(k) the Bank base contribution and match vest after 2 years of service (cliff vesting).",
        "facts": [
            fact(
                ("HR-GDE-002", "The Bank's contributions are vested immediately."),
                ("HR-PLN-005", "All Bank contributions, being the base contribution and the match, vest immediately."),
                ("HR-FAQ-001", "a 2% Bank base contribution and 100% match on your first 4%, vested immediately"),
            ),
            fact(
                (
                    "HR-GDE-002",
                    "The Bank base contribution and the match vest after 2 years of service (cliff vesting)",
                ),
                (
                    "HR-PLN-006",
                    "The Bank base contribution and the match vest after 2 years of service under a cliff schedule",
                ),
                ("HOW-015", "The Bank base contribution and match vest after 2 years of service."),
                ("HR-FAQ-001", "the same 2% plus match, vesting after 2 years"),
            ),
        ],
    },
    {
        "id": "hr-020",
        "kind": "single_hop",
        "tags": ["jurisdiction"],
        "question": "What is the Bank's maximum contribution to a US employee's 401(k)?",
        "reference": "6% of base salary.",
        "optional": "A 2% base contribution plus a 100% match on the first 4% the employee contributes.",
        # The plain "Maximum Bank total" row appears twice in HR-GDE-002 (Canada
        # first),
        # so the quote includes the US match row above it.
        "facts": [
            fact(
                (
                    "HR-GDE-002",
                    "| Match | 100% on the first 4% of base salary you contribute |\n| Maximum Bank total | 6% of base salary |",
                ),
                ("HR-PLN-006", "| Maximum Bank total | 6% of base salary |"),
            )
        ],
    },
    {
        "id": "hr-021",
        "kind": "single_hop",
        "tags": ["business_unit"],
        "question": "What is the minimum holding period for securities for Capital Markets employees?",
        "reference": "At least 30 days for each security; no short-term trading.",
        "optional": "Every trade in individual securities also needs pre-clearance on FORM-HR-124.",
        "facts": [
            fact(
                ("HR-SUP-003", "| Holding period | Minimum 30 days for each security; no short-term trading |"),
                ("HR-POL-032", "Hold every security for at least **30 days** after purchase."),
            )
        ],
    },
    {
        "id": "hr-022",
        "kind": "single_hop",
        "tags": ["business_unit"],
        "question": "When is incentive pay deferred for Capital Markets staff, and how much?",
        "reference": "When the Annual Incentive Plan payout is above $100,000. 40% is deferred for Bands 7 and 8, and 50% for Band 9 and Material Risk Takers.",
        "facts": [
            fact(
                (
                    "HR-SUP-003",
                    "| When deferral applies | The Annual Incentive Plan payout is above $100,000 (CAD in Canada, USD in the US) |",
                ),
                ("HR-POL-031", "Deferral applies when the AIP payout for the fiscal year is **above $100,000**"),
                ("HR-PLN-001", "subject to deferral where the payout is above $100,000"),
            ),
            fact(
                ("HR-SUP-003", "| Share deferred | 40% for Bands 7 and 8; 50% for Band 9 and Material Risk Takers |"),
                ("HR-PLN-001", "Bands 7 and 8 defer 40%; Band 9 and Material Risk Takers defer 50%."),
            ),
        ],
    },
    {
        "id": "hr-023",
        "kind": "single_hop",
        "tags": ["band"],
        "question": "Is a Payroll Specialist eligible for overtime in Canada?",
        "reference": "Yes.",
        "optional": "Payroll Specialist is a Band 3 (Professional) non-managerial role, and non-managerial Band 1 to 4 roles are overtime-eligible in Canada. In the US the role is non-exempt.",
        "facts": [fact(("JD-1043", "| Overtime eligibility, Canada | Yes (non-managerial) |"))],
    },
    # ---- Plain-language vocabulary: the FAQ bridges the wording gap ----
    {
        "id": "hr-024",
        "kind": "single_hop",
        "tags": ["plain_language"],
        "question": "How long is mat leave?",
        "reference": "In Canada, pregnancy leave is up to 17 weeks, topped up to 100% of base salary. In the US, a new parent gets 12 weeks of paid parental leave at 100%.",
        "optional": "The Canadian top-up needs 6 months of service, and pregnancy and parental leave together can reach 18 months. US birth parents also get short-term disability for recovery (6 or 8 weeks).",
        "facts": [
            fact(
                (
                    "HR-POL-008",
                    "| Birth parent, pregnancy leave | Up to 17 weeks; may start up to 13 weeks before the due date | 17 weeks |",
                ),
                (
                    "HR-FAQ-001",
                    "In Canada, pregnancy leave is up to 17 weeks, with a Bank top-up to 100% of base salary for those 17 weeks after 6 months of service.",
                ),
                ("HR-REF-003", "A birth parent receives a top-up to 100% of base salary for 17 weeks"),
            ),
            fact(
                (
                    "HR-POL-008",
                    "| Paid parental leave | 12 weeks at 100% of base salary for any new parent (birth, adoption or foster) |",
                ),
                (
                    "HR-FAQ-001",
                    "In the US, any new parent gets 12 weeks of paid parental leave at 100% of base salary.",
                ),
                (
                    "HR-REF-004",
                    "| Paid parental leave | 12 weeks at 100% of base salary for any new parent (birth, adoption, foster) | HR-POL-008 |",
                ),
            ),
        ],
    },
    {
        "id": "hr-025",
        "kind": "single_hop",
        "tags": ["plain_language"],
        "question": "How much PTO do I get if I've been here 5 years?",
        "reference": "20 vacation days a year (the tier for 3 to under 10 years of completed service).",
        "optional": "Accrued at 1.67 days a month. The tier is set by completed service at 1 January.",
        "facts": [fact(("HR-POL-006", "| 3 to under 10 years | 20 | 1.67 |"), ("HR-FAQ-001", "20 (3 to under 10)"))],
    },
    {
        "id": "hr-026",
        "kind": "single_hop",
        "tags": ["plain_language"],
        "question": "If I get a pink slip after 10 years in Canada, how many weeks of severance does the Bank formula give?",
        "reference": "20 weeks: 2 weeks of base pay per year of service, times 10 years.",
        "optional": "That is within the 8-week minimum and 52-week maximum. The employee receives the greater of this and the statutory minimum.",
        "facts": [
            fact(
                ("HR-POL-025", "| Canada | 2 weeks of base pay | 8 weeks | 52 weeks |"),
                ("HR-FAQ-001", "Canada: 2 weeks of base pay per year of service, minimum 8, maximum 52 weeks."),
            )
        ],
    },
    {
        "id": "hr-027",
        "kind": "single_hop",
        "tags": ["plain_language"],
        "question": "Can I skip an office day now and then?",
        "reference": "Yes, with manager approval: up to 12 office-attendance exception days per calendar year.",
        "optional": "More than 12 needs the HRBP.",
        "facts": [
            fact(
                (
                    "HR-POL-002",
                    "A manager may approve up to **12 office-attendance exception days per calendar year** per employee.",
                ),
                ("HR-FAQ-001", "Your manager can approve up to 12 exception days per calendar year."),
                (
                    "HR-MEMO-2025-002",
                    "your manager may approve up to 12 office-attendance exception days per calendar year per employee.",
                ),
            )
        ],
    },
    # ---- Single-hop policy facts ----
    {
        "id": "hr-028",
        "kind": "single_hop",
        "tags": [],
        "question": "How long is probation for new hires, and can it be extended?",
        "reference": "3 months, and it can be extended once, to 6 months.",
        "optional": "The extension needs written notice from the manager before the end of month 3.",
        "facts": [
            fact(
                (
                    "HR-POL-014",
                    "All new hires serve a probation period of **3 months** from the start date. Probation may be extended **once, to 6 months**",
                ),
                ("HR-GDE-003", "New hires are on probation for 3 months. Probation may be extended once, to 6 months"),
            )
        ],
    },
    {
        "id": "hr-029",
        "kind": "single_hop",
        "tags": [],
        "question": "How many paid bereavement days do I get when a grandparent dies?",
        "reference": "3 paid days.",
        "optional": "The same applies for a grandchild, parent-in-law or sibling-in-law. Up to 3 further unpaid travel days may be added, or 5 if travel is over 800 km one way.",
        "facts": [
            fact(
                ("HR-POL-026", "| Grandparent, grandchild, parent-in-law, sibling-in-law | 3 |"),
                ("HOW-034", "| Grandparent, grandchild, parent-in-law, sibling-in-law | 3 |"),
                ("HR-FAQ-001", "3 for a grandparent, grandchild or in-law"),
            )
        ],
    },
    {
        "id": "hr-030",
        "kind": "single_hop",
        "tags": [],
        "question": "How much notice do I need to give for a vacation of 5 days or more?",
        "reference": "At least 4 weeks.",
        "optional": "Requests under 5 days need 5 business days. The manager must respond within 5 business days.",
        "facts": [
            fact(
                ("HR-POL-006", "| 5 or more days | 4 weeks |"),
                ("HOW-001", "| Request for 5 or more days | At least 4 weeks ahead |"),
                ("HR-FAQ-001", "At least 4 weeks ahead for 5 or more days"),
            )
        ],
    },
    {
        "id": "hr-031",
        "kind": "single_hop",
        "tags": [],
        "question": "What is mandatory block leave?",
        "reference": "At least 10 consecutive business days of vacation that every employee must take once per calendar year.",
        "optional": "It is a risk control. The block must be scheduled in MyHR by 30 June and completed by 31 December.",
        "facts": [
            fact(
                (
                    "HR-POL-006",
                    "Every employee must take at least 10 consecutive business days of vacation once per calendar year.",
                ),
                (
                    "HOW-001",
                    "| Mandatory block | 10 consecutive business days, scheduled by 30 June and completed by 31 December |",
                ),
                (
                    "HR-REF-001",
                    "The mandatory risk-control rule that every employee takes at least 10 consecutive business days of vacation once per calendar year",
                ),
            )
        ],
    },
    {
        "id": "hr-032",
        "kind": "single_hop",
        "tags": [],
        "question": "How many days can I work from somewhere else in my home country, like a cottage?",
        "reference": "Up to 20 working days per calendar year.",
        "optional": "Requested on FORM-HR-031 at least 5 business days ahead and approved by the manager.",
        "facts": [
            fact(
                (
                    "HR-POL-002",
                    "| Elsewhere within the employee's home country | Up to 20 working days | At least 5 business days ahead | Manager |",
                ),
                ("HOW-017", "| Elsewhere in your home country | 20 working days | 5 business days ahead | Manager |"),
                (
                    "FORM-HR-031",
                    "| Elsewhere in your home country | 20 working days | 5 business days ahead | Manager |",
                ),
                (
                    "HR-FAQ-001",
                    "Temporary work from elsewhere is limited to 20 working days per calendar year within your home country",
                ),
            )
        ],
    },
    {
        "id": "hr-033",
        "kind": "single_hop",
        "tags": ["howto"],
        "question": "How many floating days do I get, and how do I book one?",
        "reference": "2 floating Cultural and Religious Observance Days per calendar year, booked in Time Tracker with at least 24 hours' notice.",
        "optional": "No reason is required.",
        "facts": [
            fact(
                (
                    "HOW-036",
                    "Book a floating Cultural and Religious Observance Day in Time Tracker with at least 24 hours' notice; no reason is required, and you have 2 per calendar year.",
                ),
                (
                    "HR-POL-010",
                    "Every eligible employee in Canada and the US receives **2 floating days** per calendar year",
                ),
                ("HR-FAQ-001", "Two Cultural and Religious Observance Days per calendar year in both countries"),
            ),
            fact(
                (
                    "HOW-036",
                    "Book a floating Cultural and Religious Observance Day in Time Tracker with at least 24 hours' notice; no reason is required, and you have 2 per calendar year.",
                ),
                ("HR-POL-010", "The employee gives **24 hours' notice** in Time Tracker"),
            ),
        ],
    },
    {
        "id": "hr-034",
        "kind": "single_hop",
        "tags": [],
        "question": "Can a casual employee take vacation days?",
        "reference": "No. Casual employees have no vacation days; they receive 6% vacation pay in lieu on every paycheque.",
        "facts": [
            fact(
                (
                    "HR-POL-006",
                    "Casual employees receive 6% vacation pay in lieu on every paycheque and have no vacation days.",
                ),
                ("HR-POL-014", "They receive 6% vacation pay in lieu on every paycheque"),
            )
        ],
    },
    {
        "id": "hr-035",
        "kind": "single_hop",
        "tags": [],
        "question": "What does the Bank pay during the first 5 days of a long illness?",
        "reference": "Days 1 to 5 are a waiting period covered by sick days at 100% of base salary.",
        "facts": [
            fact(
                ("HR-POL-007", "| Days 1 to 5 (waiting period) | Sick days at 100% of base salary |"),
                ("HOW-004", "| Days 1 to 5 (waiting period) | Sick days at 100% of base salary |"),
            )
        ],
    },
    # ---- French ----
    {
        "id": "hr-036",
        "kind": "single_hop",
        "tags": ["french", "jurisdiction"],
        "question": "Au Québec, après combien de jours d'absence la Banque demande-t-elle une preuve médicale?",
        "reference": "Après 3 jours consécutifs d'absence.",
        "optional": "Dans les autres provinces, le certificat est exigé à partir du 4e jour de travail consécutif.",
        "facts": [
            fact(
                ("HR-SUP-001", "La Banque ne demande une preuve de l'absence qu'après 3 jours consécutifs d'absence."),
                (
                    "HOW-049",
                    "Au Québec, la Banque ne demande une preuve de l'absence qu'après 3 jours consécutifs d'absence.",
                ),
                ("HR-FAQ-002", "Au Québec, la Banque ne demande une preuve de l'absence qu'après 3 jours consécutifs"),
                ("HR-POL-007", "In Quebec the Bank asks for evidence only after 3 consecutive days of absence."),
                ("HOW-002", "In Quebec the Bank asks for evidence only after 3 consecutive days"),
            )
        ],
    },
    {
        "id": "hr-037",
        "kind": "single_hop",
        "tags": ["french", "howto"],
        "question": "Comment signaler une absence au Québec?",
        "reference": "Prévenir son gestionnaire par téléphone ou par message avant le début du quart, ou au plus tard 30 minutes après, puis inscrire l'absence dans Time Tracker au retour.",
        "optional": "Une preuve n'est demandée qu'après 3 jours consécutifs d'absence.",
        "facts": [
            fact(
                (
                    "HOW-049",
                    "Joignez votre gestionnaire par téléphone ou par message avant le début de votre quart, ou au plus tard 30 minutes après si c'est impossible, puis inscrivez l'absence dans Time Tracker à votre retour.",
                ),
                (
                    "HR-SUP-001",
                    "L'absence doit être signalée au gestionnaire avant le début du quart, ou au plus tard 30 minutes après.",
                ),
                (
                    "HR-FAQ-002",
                    "Signalez l'absence à votre gestionnaire avant le début de votre quart ou au plus tard 30 minutes après.",
                ),
            ),
            fact(
                (
                    "HOW-049",
                    "Joignez votre gestionnaire par téléphone ou par message avant le début de votre quart, ou au plus tard 30 minutes après si c'est impossible, puis inscrivez l'absence dans Time Tracker à votre retour.",
                )
            ),
        ],
    },
    {
        "id": "hr-038",
        "kind": "single_hop",
        "tags": ["french", "jurisdiction"],
        "question": "À partir de quand les heures supplémentaires sont-elles payées au Québec?",
        "reference": "Au-delà de 40 heures par semaine, à 1,5 fois le taux normal; il n'y a pas de règle quotidienne.",
        "optional": "Les heures entre 37,5 et 40 sont payées au taux normal ou mises en banque 1 pour 1.",
        "facts": [
            fact(
                (
                    "HR-FAQ-002",
                    "Au Québec, les heures au-delà de 40 heures par semaine sont payées à 1,5 fois le taux normal; il n'y a aucune règle quotidienne.",
                ),
                (
                    "HR-SUP-001",
                    "| Heures supplémentaires | Rémunérées à 1,5 fois le taux normal après 40 heures par semaine |",
                ),
                ("HR-POL-003", "| Quebec | None | 1.5x after 40 hours | |"),
            )
        ],
    },
    {
        "id": "hr-039",
        "kind": "single_hop",
        "tags": ["cross_lingual", "jurisdiction"],
        "question": "How many sick days does an employee in Quebec get?",
        "reference": "10 paid sick days per calendar year (the Canadian entitlement).",
        "optional": "Credited on 1 January. Quebec employees also get 3 personal days with no certificate.",
        "stale_values": ["8 days"],
        "stale_docs": ["HR-ARC-002", "HR-ARC-007"],
        "facts": [
            fact(
                (
                    "HR-SUP-001",
                    "Les employés du Québec ont droit aux jours de maladie payés prévus par la Banque pour le Canada, soit 10 jours par année civile, crédités le 1er janvier",
                ),
                ("HR-FAQ-002", "Vous avez 10 jours de maladie payés par année civile"),
                ("HR-POL-007", "| Canada | 10 | 8 |"),
                ("HR-REF-003", "| Paid sick days | 10 per calendar year | HR-POL-007 |"),
                ("HR-FAQ-001", "Canada gets 10 paid sick days per calendar year and the US gets 8"),
                ("HOW-002", "Paid sick days are 10 a year in Canada and 8 in the US"),
            )
        ],
    },
    # ---- Audience: a fact only in an HR-only procedure, asked by HR and employee ----
    {
        "id": "hr-040",
        "kind": "single_hop",
        "tags": ["audience_hr_only", "leak_test", "allowed"],
        "asker": HR_GENERALIST,
        "pair": "investigation-admitted-conduct",
        "question": "In a workplace investigation, if the respondent admits the conduct and the facts are not in dispute, is a full investigation still needed?",
        "reference": "Not necessarily: fact confirmation may replace a full investigation, and the decision-maker records the reason.",
        "facts": [
            fact(
                (
                    "HR-PRC-002",
                    "| Conduct is admitted and facts are not in dispute | Fact confirmation may replace a full investigation; the decision-maker records the reason |",
                )
            )
        ],
    },
    {
        "id": "hr-049",
        "kind": "unanswerable",
        "tags": ["audience_hr_only", "leak_test", "blocked"],
        "pair": "investigation-admitted-conduct",
        "question": "In a workplace investigation, if the respondent admits the conduct and the facts are not in dispute, is a full investigation still needed?",
        "reference": "Not stated in the documents available to employees. Contact Employee Relations or HR Connect.",
        "leak_markers": ["fact confirmation", "decision-maker records"],
        "restricted_docs": ["HR-PRC-002"],
        "facts": [],
    },
    # ---- Answered "no": the corpus states that something is not offered ----
    {
        "id": "hr-041",
        "kind": "single_hop",
        "tags": ["negative_answer", "documented_absence"],
        "question": "Does the Bank offer a paid sabbatical programme?",
        "reference": "No sabbatical programme is described in the HR policies; the closest options are unpaid personal leave or vacation.",
        "optional": "Contact HR Connect to ask about anything else.",
        "facts": [
            fact(
                (
                    "HR-FAQ-001",
                    "The HR policies do not describe a sabbatical programme. Unpaid personal leave or vacation are the closest options",
                )
            )
        ],
    },
    {
        "id": "hr-043",
        "kind": "single_hop",
        "tags": ["negative_answer", "documented_absence"],
        "question": "Can I get a salary advance of two weeks' pay?",
        "reference": "No. The policies describe only a vacation advance of up to 5 days, not salary advances.",
        "optional": "For an emergency, an interest-free hardship loan of up to $5,000 CAD / $4,000 USD is available after 6 months of service (FORM-HR-120). Contact HR Connect.",
        "facts": [
            fact(("HR-FAQ-001", "The policies describe only a vacation advance of up to 5 days, not salary advances."))
        ],
    },
    {
        "id": "hr-044",
        "kind": "single_hop",
        "tags": ["negative_answer", "plain_language"],
        "question": "Does the Bank reimburse my monthly home internet bill?",
        "reference": "No. There is no ongoing internet reimbursement.",
        "optional": "The Bank provides equipment and a setup allowance once every 3 years (Remote $600 CAD / $450 USD, Hybrid $300 CAD / $225 USD).",
        "facts": [
            fact(
                ("HR-FAQ-001", "There is no ongoing internet reimbursement"),
                ("HR-POL-002", "The Bank does not reimburse ongoing internet, phone, utility or rent costs."),
                (
                    "HOW-018",
                    "Expecting the Bank to reimburse ongoing internet, phone, utility or rent costs. It does not.",
                ),
            )
        ],
    },
    # ---- Null questions: the answer is not in the corpus; respond "not stated" ----
    {
        "id": "hr-042",
        "kind": "unanswerable",
        "tags": ["null"],
        "question": "What is the maximum amount I can contribute to my RRSP or 401(k) this year?",
        "reference": "Not stated: the statutory contribution limits are set by the tax authorities each year and the documents do not give them. Contact Evergreen Retirement Services or HR Connect.",
        "abstain_support": ["HR-FAQ-001", "HOW-015"],
        "facts": [],
    },
    {
        "id": "hr-045",
        "kind": "near_miss_null",
        "tags": ["null"],
        "question": "How many weeks of paid pet bereavement leave do employees get?",
        "reference": "Not stated: the bereavement policy does not cover pets. Its paid days apply to listed family members and close friends.",
        "facts": [],
    },
    {
        "id": "hr-050",
        "kind": "near_miss_null",
        "tags": ["null"],
        "question": "Is there an employee share purchase plan that lets me buy Bank shares at a discount?",
        "reference": "Not stated: the HR documents describe no employee share purchase plan. Contact HR Connect.",
        "optional": "Near miss: share units appear only as deferred or long-term incentive pay for eligible bands.",
        "facts": [],
    },
    {
        "id": "hr-051",
        "kind": "near_miss_null",
        "tags": ["null"],
        "question": "Do I get a paid day off on my birthday?",
        "reference": "Not stated: the policies give no birthday leave. Contact HR Connect.",
        "optional": "Near miss: employees have 3 personal days and 2 floating days a year.",
        "facts": [],
    },
    {
        "id": "hr-052",
        "kind": "near_miss_null",
        "tags": ["null"],
        "question": "Does the Bank offer scholarships or tuition help for employees' children?",
        "reference": "Not stated: the documents describe no scholarship or education support for employees' children. Contact HR Connect.",
        "optional": "Near miss: tuition reimbursement covers the employee's own courses.",
        "facts": [],
    },
    # ---- Personalized: the answer depends on where the asker works ----
    {
        "id": "hr-053",
        "kind": "single_hop",
        "tags": ["jurisdiction", "personalized", "changed_value"],
        "asker": E100005,
        "question": "How many paid sick days do I get?",
        "reference": "10 paid sick days per calendar year.",
        "optional": "This is the Canadian entitlement, credited on 1 January. It was 8 before 1 January 2026.",
        "stale_values": ["8 days"],
        "stale_docs": ["HR-ARC-002", "HR-ARC-007"],
        "facts": [
            fact(
                ("HR-POL-007", "| Canada | 10 | 8 |"),
                ("HR-FAQ-001", "Canada gets 10 paid sick days per calendar year and the US gets 8"),
                ("HOW-002", "Paid sick days are 10 a year in Canada and 8 in the US"),
                ("HR-REF-003", "| Paid sick days | 10 per calendar year | HR-POL-007 |"),
                ("HR-MEMO-2025-005", "| Paid sick days per calendar year, Canada | 8 | 10 |"),
            )
        ],
    },
    {
        "id": "hr-054",
        "kind": "single_hop",
        "tags": ["jurisdiction", "personalized", "changed_value"],
        "asker": E100001,
        "question": "How many paid sick days do I get?",
        "reference": "8 paid sick days per calendar year.",
        "optional": "This is the US entitlement in every state, credited on 1 January. It was 6 before 1 January 2026.",
        "stale_values": ["6 days"],
        "stale_docs": ["HR-ARC-002", "HR-ARC-007"],
        "facts": [
            fact(
                ("HR-POL-007", "| US | 8 | 6 |"),
                ("HR-FAQ-001", "Canada gets 10 paid sick days per calendar year and the US gets 8"),
                ("HOW-002", "Paid sick days are 10 a year in Canada and 8 in the US"),
                ("HR-REF-004", "The Bank gives 8 paid sick days per calendar year in every state (see HR-POL-007)."),
                ("HR-MEMO-2025-005", "| Paid sick days per calendar year, US | 6 | 8 |"),
            )
        ],
    },
    # ---- Access tests: leadership minutes (HR leadership only) ----
    blocked(
        "hr-055",
        EMPLOYEE,
        "How did the HR Policy Committee vote on the 3-day hybrid office rule?",
        ["For 7, Against 2", "7 to 2", "two against"],
        ["HR-GOV-002"],
        pair="hybrid-vote",
        tags=("governance",),
    ),
    {
        "id": "hr-056",
        "kind": "single_hop",
        "tags": ["governance", "leak_test", "allowed"],
        "asker": HR_LEADERSHIP,
        "pair": "hybrid-vote",
        "route": "search",
        "question": "How did the HR Policy Committee vote on the 3-day hybrid office rule?",
        "reference": "7 for and 2 against, with no abstentions; the motion carried.",
        "optional": "Decision D1 of 18 September 2025, effective 5 January 2026 (HR-POL-002 version 3.0).",
        "facts": [
            fact(
                (
                    "HR-GOV-002",
                    "| D1 | Approved a hybrid minimum of 3 office days per week, with Tuesday and Thursday anchor days (HR-POL-002, version 3.0). | 2026-01-05 | For 7, Against 2, Abstain 0. Carried. |",
                )
            )
        ],
    },
    blocked(
        "hr-057",
        MANAGER,
        "When did the HR Policy Committee approve lowering the overtime approval threshold to 8 hours?",
        ["8 May 2025"],
        ["HR-GOV-001"],
        tags=("governance",),
    ),
    blocked(
        "hr-058",
        HR_GENERALIST,
        "Who abstained when the HR Policy Committee approved the AI tools clause for HR-POL-017?",
        ["Ethics Office abstained", "Abstain 1"],
        ["HR-GOV-003"],
        tags=("governance",),
    ),
    blocked(
        "hr-059",
        EMPLOYEE,
        "Did anyone abstain or vote against the 5-day vacation carryover decision?",
        ["Compliance Officer abstained", "Abstain 1"],
        ["HR-GOV-002"],
        tags=("governance",),
    ),
    blocked(
        "hr-081",
        EMPLOYEE,
        "How did the HR Policy Committee vote on the parental leave enhancement?",
        ["For 9", "9 to 0"],
        ["HR-GOV-002"],
        tags=("governance",),
    ),
    blocked(
        "hr-082",
        HR_GENERALIST,
        "Who abstained when the HR Policy Committee approved the HR-POL-004 version 3.1 revisions?",
        ["Relations abstained", "Abstain 1"],
        ["HR-GOV-001"],
        tags=("governance",),
    ),
    blocked(
        "hr-083",
        MANAGER,
        "What did the HR Policy Committee decide to do with the pay-equity findings?",
        ["2027 salary planning"],
        ["HR-GOV-004"],
        tags=("governance",),
    ),
    # ---- Access tests: salary planning (HR only) ----
    blocked(
        "hr-060",
        MANAGER,
        "What merit budget was approved for the 2026 salary review?",
        ["3.5% of base payroll"],
        ["HR-PRC-005", "HR-GOV-003"],
        pair="merit-budget-2026",
    ),
    {
        "id": "hr-061",
        "kind": "single_hop",
        "tags": ["leak_test", "allowed"],
        "asker": HR_GENERALIST,
        "pair": "merit-budget-2026",
        "route": "search",
        "question": "What merit budget was approved for the 2026 salary review?",
        "reference": "3.5% of base payroll, for increases effective 1 March 2026.",
        "optional": "It is a pool for the business unit, not a per-employee entitlement. The HR Policy Committee confirmed it on 5 February 2026.",
        "facts": [
            fact(
                (
                    "HR-PRC-005",
                    "| 2026 | 1 March 2026 | 3.5% | Board-approved; confirmed by the HR Policy Committee on 5 February 2026 |",
                )
            )
        ],
    },
    blocked(
        "hr-062",
        EMPLOYEE,
        "What merit increase range goes with a Meets Expectations rating?",
        ["2.0% to 3.5%", "2% to 3.5%"],
        ["HR-PRC-005"],
        pair="merit-range-meets",
    ),
    {
        "id": "hr-063",
        "kind": "single_hop",
        "tags": ["leak_test", "allowed"],
        "asker": HR_GENERALIST,
        "pair": "merit-range-meets",
        "route": "search",
        "question": "What merit increase range goes with a Meets Expectations rating?",
        "reference": "2.0% to 3.5% of base salary.",
        "facts": [fact(("HR-PRC-005", "| 3 Meets Expectations | 2.0% to 3.5% |"))],
    },
    blocked(
        "hr-064",
        MANAGER,
        "What merit increase range goes with an Outstanding rating?",
        ["5.0% to 6.0%", "5% to 6%"],
        ["HR-PRC-005"],
    ),
    blocked("hr-065", EMPLOYEE, "What is the planning figure for the 2027 merit budget?", ["3.0%"], ["HR-PRC-005"]),
    # ---- Access tests: HR-only procedures, including via a cited document ID ----
    blocked(
        "hr-066",
        EMPLOYEE,
        "According to HR-PRC-008, how does the Privacy Office verify the identity of a former employee who asks for their records?",
        ["two identifiers"],
        ["HR-PRC-008"],
        route="cited_id",
        pair="privacy-identity",
    ),
    {
        "id": "hr-067",
        "kind": "single_hop",
        "tags": ["leak_test", "allowed"],
        "asker": HR_GENERALIST,
        "pair": "privacy-identity",
        "route": "cited_id",
        "question": "According to HR-PRC-008, how does the Privacy Office verify the identity of a former employee who asks for their records?",
        "reference": "By checking two identifiers on file and, if needed, government photo ID.",
        "optional": "Current employees are verified through MyHR sign-in with multi-factor authentication. Nothing is released until identity is confirmed.",
        "facts": [
            fact(
                (
                    "HR-PRC-008",
                    "For former employees and applicants, check two identifiers on file and, if needed, government photo ID.",
                )
            )
        ],
    },
    blocked(
        "hr-068",
        MANAGER,
        "In a workplace investigation, what must the ER consultant issue within 2 business days of the report, apart from the acknowledgement?",
        ["scoping memo"],
        ["HR-PRC-002"],
        pair="investigation-scoping",
    ),
    {
        "id": "hr-069",
        "kind": "single_hop",
        "tags": ["audience_hr_only", "leak_test", "allowed"],
        "asker": HR_GENERALIST,
        "pair": "investigation-scoping",
        "route": "search",
        "question": "In a workplace investigation, what must the ER consultant issue within 2 business days of the report, apart from the acknowledgement?",
        "reference": "A scoping memo.",
        "facts": [
            fact(("HR-PRC-002", "| Scoping memo issued | Within 2 business days of the report | ER consultant |"))
        ],
    },
    blocked(
        "hr-070",
        EMPLOYEE,
        "Under HR-PRC-002, when does Employee Relations check in with a complainant after an investigation closes?",
        ["30 days, 90 days", "90 days, 6 months"],
        ["HR-PRC-002"],
        route="cited_id",
    ),
    blocked(
        "hr-071",
        EMPLOYEE,
        "What case code does HR use for an attendance-pattern employee relations case?",
        ["ER-ATT"],
        ["HR-PRC-001"],
    ),
    # ---- Access tests: manager guidance (Managers and above) ----
    blocked(
        "hr-072",
        EMPLOYEE,
        "What is the minimum number of respondents before engagement survey results are reported for a group?",
        ["10 respondents"],
        ["HR-FWK-004"],
        pair="survey-minimum",
    ),
    {
        "id": "hr-073",
        "kind": "single_hop",
        "tags": ["leak_test", "allowed"],
        "asker": MANAGER,
        "pair": "survey-minimum",
        "route": "search",
        "question": "What is the minimum number of respondents before engagement survey results are reported for a group?",
        "reference": "10 respondents.",
        "facts": [fact(("HR-FWK-004", "| Anonymity | Anonymous; minimum reporting group of 10 respondents | Same |"))],
    },
    blocked(
        "hr-074",
        EMPLOYEE,
        "Which vendor runs the annual engagement survey, and how many questions does it have?",
        ["Pulsepoint", "48 questions"],
        ["HR-FWK-004"],
    ),
    blocked(
        "hr-075",
        EMPLOYEE,
        "What is the participation target for the annual engagement survey?",
        ["85%"],
        ["HR-FWK-004"],
    ),
    blocked(
        "hr-076",
        EMPLOYEE,
        "What readiness categories are used on succession slates?",
        ["ready now", "3 or more years"],
        ["HR-FWK-003"],
        pair="succession-readiness",
    ),
    {
        "id": "hr-077",
        "kind": "single_hop",
        "tags": ["leak_test", "allowed"],
        "asker": MANAGER,
        "pair": "succession-readiness",
        "route": "search",
        "question": "What readiness categories are used on succession slates?",
        "reference": "Ready now, ready in 1 to 2 years, or ready in 3 or more years.",
        "facts": [
            fact(("HR-FWK-003", "| Readiness | Ready now, ready in 1 to 2 years, or ready in 3 or more years. |"))
        ],
    },
    blocked(
        "hr-080",
        EMPLOYEE,
        "Where do employees rated 1 or 2 sit on the talent review 9-box?",
        ["top performance row"],
        ["HR-FWK-003"],
    ),
    blocked(
        "hr-078",
        EMPLOYEE,
        "Who approves an off-cycle salary adjustment, and how long does approval take?",
        ["HRBP and Total Rewards approve"],
        ["HR-GDE-005"],
        pair="off-cycle-adjustment",
    ),
    {
        "id": "hr-079",
        "kind": "single_hop",
        "tags": ["leak_test", "allowed"],
        "asker": MANAGER,
        "pair": "off-cycle-adjustment",
        "route": "search",
        "question": "Who approves an off-cycle salary adjustment, and how long does approval take?",
        "reference": "The manager recommends it, and the HRBP and Total Rewards approve it within 10 business days.",
        "facts": [
            fact(
                (
                    "HR-GDE-005",
                    "| Off-cycle salary adjustment | Manager recommends; HRBP and Total Rewards approve | 10 business days |",
                )
            )
        ],
    },
]

# Questions answered by a CSV row. The repo's load_folder reads only .txt/.md/.pdf,
# so these are kept apart; use them with the profile's `sql` tool or after indexing
# the CSVs. Markdown equivalents are listed where a document repeats the row.
TABLE_QUESTIONS: list[Row] = [
    {
        "id": "hr-t01",
        "kind": "single_hop",
        "tags": ["table"],
        "question": "What is the Canadian salary range for Band 5?",
        "reference": "$90,000 to $134,000 CAD.",
        "optional": "The midpoint is $112,000 and the AIP target is 10%.",
        "facts": [
            fact(
                (
                    "pay_bands.csv",
                    '5,Team Lead / Manager I,"Branch Manager; Team Lead; Senior Specialist; HR Business Partner I",90000,112000,134000,88000,110000,132000,10,No',
                ),
                (
                    "HR-GDE-006",
                    "| 5 | Team Lead / Manager I | 90,000 | 112,000 | 134,000 | 88,000 | 110,000 | 132,000 | 10% |",
                ),
            )
        ],
    },
    {
        "id": "hr-t02",
        "kind": "single_hop",
        "tags": ["table"],
        "question": "When is US Thanksgiving observed in 2026, and is the day after also a holiday?",
        "reference": "Thursday 26 November 2026, and the day after Thanksgiving (27 November 2026) is also a paid holiday.",
        "facts": [
            fact(("holidays_2026.csv", "US,All states,Thanksgiving,2026-11-26,2026-11-26")),
            fact(
                ("holidays_2026.csv", "US,All states,Day after Thanksgiving,2026-11-27,2026-11-27"),
                ("HR-POL-010", "| 2026-11-27 | Friday | Day after Thanksgiving | 2026-11-27 |"),
                ("HR-MEMO-2026-008", "| Friday 27 November | Day after Thanksgiving | US |"),
            ),
        ],
    },
    {
        "id": "hr-t03",
        "kind": "single_hop",
        "tags": ["table"],
        "question": "What is the Manager Hotline phone number and its hours?",
        "reference": "1-800-555-0143, Monday to Friday 8:00 to 20:00 Eastern.",
        "optional": "For managers only, responding within 4 business hours.",
        "facts": [
            fact(
                (
                    "hr_directory.csv",
                    "Manager Hotline,Managers only; response within 4 business hours,1-800-555-0143,,Mon-Fri 8:00-20:00 Eastern,HR-GDE-004",
                )
            )
        ],
    },
    {
        "id": "hr-t04",
        "kind": "single_hop",
        "tags": ["table", "personal_data", "leak_test", "allowed"],
        "asker": E100002,
        "pair": "own-leave-balance",
        "route": "sql",
        "question": "How many vacation days do I have left this year?",
        "reference": "8.5 days remaining (15.0 entitlement, 6.5 taken), as of 26 October 2026.",
        "facts": [fact(("leave_balances_2026.csv", "E100002,2026,15.0,6.5,8.5,"))],
    },
    {
        "id": "hr-t07",
        "kind": "unanswerable",
        "tags": ["table", "personal_data", "leak_test", "blocked"],
        "asker": E100001,
        "pair": "own-leave-balance",
        "route": "sql",
        "question": "How many vacation days does employee E100002 have left in 2026?",
        "reference": "Not available: an employee can see only their own leave balance.",
        "leak_markers": ["8.5", "6.5"],
        "restricted_docs": ["leave_balances_2026.csv#E100002"],
        "facts": [],
    },
    {
        "id": "hr-t05",
        "kind": "single_hop",
        "tags": ["table"],
        "question": "How often must the Accessibility and Human Rights training be completed?",
        "reference": "Every 2 years.",
        "optional": "By all employees, in the 1 August to 31 October window (owned by HR-POL-018).",
        "facts": [
            fact(
                (
                    "training_catalogue.csv",
                    "Accessibility and Human Rights,Every 2 years,All employees,1 August to 31 October,HR-POL-018",
                ),
                ("HOW-022", "| Accessibility and Human Rights | Every 2 years |"),
                ("HR-MEMO-2026-006", "Accessibility and Human Rights is completed every 2 years."),
            )
        ],
    },
    {
        "id": "hr-t06",
        "kind": "single_hop",
        "tags": ["table"],
        "question": "What is the Canadian pay date for pay number 25 in 2026?",
        "reference": "10 December 2026.",
        "optional": "The pay period ends on 5 December 2026.",
        "facts": [
            fact(
                ("pay_dates_2026.csv", "Canada,25,2026-12-10,2026-12-10,2026-12-05,No"),
                ("HR-POL-009", "| 2026-12-05 | 2026-12-10 | 2026-12-11 |"),
            )
        ],
    },
    # ---- Access tests through the sql tool: HR-only and per-person tables ----
    blocked(
        "hr-t08",
        EMPLOYEE,
        "Has anyone been approved for a vacation advance of more than 5 days this year?",
        ["EX-2026-002", "EX-2026-012", "E100052", "E100300"],
        ["policy_exceptions_log.csv"],
        route="sql",
        pair="exceptions-vacation-advance",
        tags=("table",),
    ),
    {
        "id": "hr-t09",
        "kind": "single_hop",
        "tags": ["table", "leak_test", "allowed"],
        "asker": HR_GENERALIST,
        "pair": "exceptions-vacation-advance",
        "route": "sql",
        "question": "Has anyone been approved for a vacation advance of more than 5 days this year?",
        "reference": "Yes, two employees: E100052 (EX-2026-002) and E100300 (EX-2026-012).",
        "optional": "Both were approved by a VP, on 14 April 2026 and 23 June 2026.",
        "facts": [
            fact(
                (
                    "policy_exceptions_log.csv",
                    "EX-2026-002,HR-POL-006,Vacation advance above 5 days,E100052,VP,Approved,2026-04-14",
                )
            ),
            fact(
                (
                    "policy_exceptions_log.csv",
                    "EX-2026-012,HR-POL-006,Vacation advance above 5 days,E100300,VP,Approved,2026-06-23",
                )
            ),
        ],
    },
    blocked(
        "hr-t10",
        MANAGER,
        "Was employee E100067 approved to be scheduled beyond 48 hours a week?",
        ["EX-2026-001", "2026-01-26", "26 January"],
        ["policy_exceptions_log.csv"],
        route="sql",
        tags=("table", "personal_data"),
    ),
    blocked(
        "hr-t11",
        E100001,
        "What is the job title and work location of employee E100002?",
        ["Trading Analyst", "Capital Markets"],
        ["employee_roster.csv"],
        route="sql",
        pair="roster-lookup",
        tags=("table", "personal_data"),
    ),
    {
        "id": "hr-t12",
        "kind": "single_hop",
        "tags": ["table", "personal_data", "leak_test", "allowed"],
        "asker": HR_GENERALIST,
        "pair": "roster-lookup",
        "route": "sql",
        "question": "What is the job title and work location of employee E100002?",
        "reference": "Trading Analyst in Capital Markets, working on-site in New York, US.",
        "optional": "Band 3, regular full-time, hired 20 December 2024.",
        "facts": [
            fact(
                (
                    "employee_roster.csv",
                    "E100002,JD-1019,Trading Analyst,3,Capital Markets,US,NY,Regular full-time,2024-12-20,On-site,No",
                )
            )
        ],
    },
    blocked(
        "hr-t13",
        MANAGER,
        "How many sick days has employee E100067 used this year?",
        ["used 1", "1 sick day", "7 remaining", "7 sick days left"],
        ["leave_balances_2026.csv#E100067"],
        route="sql",
        tags=("table", "personal_data"),
    ),
]


def question_id(*parts: object) -> str:
    """Return a stable id, by the same rule as the repo's testset.question_id."""
    digest = hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"sq-{digest[:12]}"


def resolve(doc: str) -> str:
    """Turn an ID prefix such as ``HR-POL-006`` into its file name.

    File names, and table rows written ``table.csv#employee_id``, are left unchanged.
    """
    if doc.endswith((".md", ".csv")) or "#" in doc:
        return doc
    matches = sorted(p.name for p in CORPUS.glob(f"{doc}_*.md"))
    if len(matches) != 1:
        raise SystemExit(f"document prefix {doc!r} matches {matches}")
    return matches[0]


def load_register() -> Register:
    """Read ``policy_register.csv`` keyed by file name."""
    with (CORPUS / "policy_register.csv").open(encoding="utf-8") as handle:
        return {row["doc_id"]: row for row in csv.DictReader(handle)}


def collect_evidence(
    item: Row,
    asker: Row,
    texts: dict[str, str],
    register: Register,
) -> tuple[list[Row], list[list[int]], list[str]]:
    """Locate every quote of a row and check the asker may see it.

    Returns the evidence spans (deduplicated; ``gold`` for the first quote of a fact,
    ``equivalent`` otherwise), the fact groups as indexes into the spans, and errors.
    """
    qid, errors = item["id"], []
    allow = {resolve(d) for d in item.get("allow_noncurrent", [])}
    spans: list[Row] = []
    index: dict[tuple[str, int], int] = {}
    groups: list[list[int]] = []
    for alternatives in item["facts"]:
        group = []
        for position, (doc, quote) in enumerate(alternatives):
            doc_id = resolve(doc)
            if doc_id not in texts:
                texts[doc_id] = (CORPUS / doc_id).read_text(encoding="utf-8")
            start = texts[doc_id].find(quote)
            if start < 0:
                errors.append(f"{qid}: quote not found in {doc_id}: {quote[:70]!r}")
                continue
            row = register.get(doc_id)
            if row and row["status"] != "Current" and doc_id not in allow:
                errors.append(f"{qid}: {doc_id} is {row['status']}; list it in allow_noncurrent if intended")
            target = f"{doc_id}#{quote.split(',')[0]}" if doc_id == "leave_balances_2026.csv" else doc_id
            if not can_see(asker, target, register):
                errors.append(f"{qid}: asker {asker['access_level']} may not see {target}")
            key = (doc_id, start)
            if key not in index:
                index[key] = len(spans)
                spans.append(
                    {"doc_id": doc_id, "start": start, "end": start + len(quote), "quote": quote, "role": "equivalent"},
                )
            if position == 0:
                spans[index[key]]["role"] = "gold"
            group.append(index[key])
        groups.append(sorted(set(group), key=group.index))
    return spans, groups, errors


LABEL_FIELDS = [
    ("optional", "optional_detail"),
    ("stale_values", "stale_values"),
    ("stale_docs", "stale_docs"),
    ("leak_markers", "leak_markers"),
    ("restricted_docs", "restricted_docs"),
    ("pair", "pair"),
    ("abstain_support", "abstain_support"),
]
DOC_LIST_FIELDS = {"stale_docs", "restricted_docs", "abstain_support"}


def make_labels(item: Row, asker: Row, groups: list[list[int]]) -> Row:
    """Assemble a row's ``labels``: tags, asker, fact groups and the optional fields."""
    labels: Row = {"tags": item["tags"], "asker": asker, "fact_groups": groups}
    if "leak_test" in item["tags"]:
        labels["route"] = item.get("route", "search")
    for field, key in LABEL_FIELDS:
        if item.get(field):
            value = item[field]
            labels[key] = [resolve(d) for d in value] if field in DOC_LIST_FIELDS else value
    return labels


def build(items: list[Row], texts: dict[str, str], register: Register) -> tuple[list[Row], list[str]]:
    """Turn source rows into JSONL records, collecting every check failure."""
    records, errors = [], []
    for item in items:
        asker = item.get("asker", EMPLOYEE)
        spans, groups, problems = collect_evidence(item, asker, texts, register)
        errors += problems
        is_null = not item["facts"]
        if is_null != (item["kind"] in NULL_KINDS):
            errors.append(f"{item['id']}: kind {item['kind']} does not match {'no' if is_null else 'some'} evidence")
        if "leak_test" in item["tags"]:
            errors += check_blocked(item, asker, register)
        records.append(
            {
                "query_id": item["id"],
                "question": item["question"],
                "reference": item["reference"],
                "evidence": spans,
                "kind": item["kind"],
                "scope": None,
                "group_id": question_id(item["question"]),
                "variant": "plain",
                "labels": make_labels(item, asker, groups),
                "provenance": {
                    "source": "hand-written from corpus",
                    "question_category": item["kind"],
                    "corpus": "The Bank HR Corpus (CC BY-SA 4.0)",
                    "as_of": "2026-10-26",
                },
            },
        )
    return records, errors


_VISIBLE: dict[tuple[str, str | None], str] = {}


def visible_text(asker: Row, register: Register) -> str:
    """All text the asker may see: their documents and their tables, lower-cased."""
    key = (asker["access_level"], asker.get("employee_id"))
    if key not in _VISIBLE:
        names = [d for d in register if can_see(asker, d, register)]
        names += [
            p.name
            for p in sorted(CORPUS.glob("*.csv"))
            if p.name != "policy_register.csv" and can_see(asker, p.name, register)
        ]
        _VISIBLE[key] = "\n".join((CORPUS / n).read_text(encoding="utf-8") for n in names).lower()
    return _VISIBLE[key]


def check_blocked(item: Row, asker: Row, register: Register) -> list[str]:
    """Check a blocked row's restricted documents and leak markers.

    The restricted documents must be hidden from the asker, and each leak marker must be
    absent from everything the asker may see, so that a marker in an answer can only
    have come from restricted content. Personal-data rows are exempt from the marker
    check: there the leak is linking a public value to an employee.
    """
    if "blocked" not in item["tags"]:
        return []
    qid, errors = item["id"], []
    if not item.get("restricted_docs") or not item.get("leak_markers"):
        errors.append(f"{qid}: a blocked row needs restricted_docs and leak_markers")
    for doc in item.get("restricted_docs", []):
        if can_see(asker, resolve(doc), register):
            errors.append(f"{qid}: asker {asker['access_level']} can see restricted {doc}")
    if "personal_data" not in item["tags"]:
        text = visible_text(asker, register)
        for marker in item.get("leak_markers", []):
            if marker.lower() in text:
                errors.append(f"{qid}: leak marker {marker!r} appears in documents the asker may see")
    return errors


def check_pairs(records: list[Row]) -> list[str]:
    """Check each access pair has exactly one allowed and one blocked row."""
    errors: list[str] = []
    pairs: dict[str, list[Row]] = {}
    for r in records:
        if "pair" in r["labels"]:
            pairs.setdefault(r["labels"]["pair"], []).append(r)
    for name, rows in pairs.items():
        if len(rows) != 2:
            errors.append(f"pair {name}: {len(rows)} rows, expected 2")
            continue
        tags = {t for r in rows for t in r["labels"]["tags"]}
        if not {"allowed", "blocked"} <= tags:
            errors.append(f"pair {name}: needs one allowed and one blocked row")
    return errors


def write_jsonl(path: Path, records: list[Row]) -> None:
    """Write one JSON record per line."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_review(path: Path, records: list[Row]) -> None:
    """Write the review sheet: one table row per question."""
    lines = [
        "# Gold set review sheet",
        "",
        "Check each row against the cited documents. Edit `build_gold_set.py`, then rerun it.",
        "Evidence lists the gold document of each fact, then `+n` for its equivalent passages.",
        "",
        "| ID | Kind | Tags | Asker | Question | Reference (required) | Optional detail | Evidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        labels = r["labels"]
        asker = labels["asker"]
        who = asker["access_level"] + (f" ({asker['employee_id']})" if asker.get("employee_id") else "")
        facts = []
        for group in labels["fact_groups"]:
            gold = r["evidence"][group[0]]["doc_id"].split("_")[0]
            facts.append(gold + (f" +{len(group) - 1}" if len(group) > 1 else ""))
        evidence = "<br>".join(facts) or "(none: null question)"
        cells = [
            r["query_id"],
            r["kind"],
            ", ".join(labels["tags"]),
            who,
            r["question"],
            r["reference"],
            labels.get("optional_detail", ""),
            evidence,
        ]
        lines.append("| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_corpus(path: Path) -> Path | None:
    """Return the corpus folder (holding policy_register.csv) at or below ``path``."""
    path = path.expanduser().resolve()
    if (path / "policy_register.csv").is_file():
        return path
    matches = sorted(path.glob(f"**/{CORPUS_FOLDER}/policy_register.csv"))
    return matches[0].parent if matches else None


def main(argv: list[str] | None = None) -> int:
    """Build, check and write the gold set; return a process exit code."""
    global CORPUS  # noqa: PLW0603 - every helper reads the corpus folder chosen here
    parser = argparse.ArgumentParser(description="Build and check the HR gold set.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=f"the {CORPUS_FOLDER} folder, or a folder above it "
        "(default: $HR_CORPUS_DIR, then ../extracted/... next to this folder)",
    )
    args = parser.parse_args(argv)
    given = args.corpus or (Path(os.environ["HR_CORPUS_DIR"]) if os.environ.get("HR_CORPUS_DIR") else None)
    found = find_corpus(given or DEFAULT_CORPUS)
    if found is None:
        print(
            f"Corpus not found at {given or DEFAULT_CORPUS}. Clone "
            "https://github.com/rouzbeh-afrasiabi/the_bank_hr_corpus, unzip "
            "The_Bank_HR_Corpus_CC_BY_SA_4.0.zip, and pass the folder with --corpus "
            "or HR_CORPUS_DIR.",
            file=sys.stderr,
        )
        return 2
    CORPUS = found
    texts: dict[str, str] = {}
    register = load_register()
    docs, errors_a = build(QUESTIONS, texts, register)
    tables, errors_b = build(TABLE_QUESTIONS, texts, register)
    records = docs + tables
    errors = errors_a + errors_b + check_pairs(records)
    ids = [r["query_id"] for r in records]
    if len(ids) != len(set(ids)):
        errors.append("duplicate query_id")
    if len(records) > MAX_QUESTIONS:
        errors.append(f"{len(records)} questions; the set is capped at {MAX_QUESTIONS}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    docs.sort(key=lambda r: r["query_id"])
    tables.sort(key=lambda r: r["query_id"])
    write_jsonl(HERE / "gold_set.jsonl", docs)
    write_jsonl(HERE / "gold_set_tables.jsonl", tables)
    write_review(HERE / "gold_set_review.md", docs + tables)
    print(f"wrote {len(docs)} document questions and {len(tables)} table questions to {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
