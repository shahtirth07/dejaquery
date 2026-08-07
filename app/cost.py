"""
Turns token counts into a running cost log for the live demo meter.

The $/token rate below is illustrative, not pulled from a live price
sheet — model pricing changes and varies by cache-hit rate. For the
real demo, either read Snowflake's own AI_OBSERVABILITY_EVENTS-style
usage table if you route calls through Cortex, or swap in the current
rate from https://docs.claude.com/en/docs/about-claude/pricing. Don't
present this constant as an authoritative price on stage — present the
*shape* of the drop (cold vs warm token counts), which is the real,
defensible claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ILLUSTRATIVE_RATE_PER_1K_INPUT_TOKENS = 0.003
ILLUSTRATIVE_RATE_PER_1K_OUTPUT_TOKENS = 0.015


@dataclass
class CostEntry:
    question: str
    path: str  # "cold" | "warm" | "unmatched"
    input_tokens: int
    output_tokens: int

    @property
    def estimated_usd(self) -> float:
        return (
            self.input_tokens / 1000 * ILLUSTRATIVE_RATE_PER_1K_INPUT_TOKENS
            + self.output_tokens / 1000 * ILLUSTRATIVE_RATE_PER_1K_OUTPUT_TOKENS
        )


@dataclass
class CostLog:
    entries: list[CostEntry] = field(default_factory=list)

    def add(self, question: str, path: str, input_tokens: int, output_tokens: int) -> CostEntry:
        entry = CostEntry(
            question=question,
            path=path,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.entries.append(entry)
        return entry

    def total_usd(self) -> float:
        return sum(e.estimated_usd for e in self.entries)

    def savings_pct(self) -> float | None:
        """% cheaper the whole session was vs. if every question had gone cold."""
        cold_entries = [e for e in self.entries if e.path == "cold"]
        if not cold_entries:
            return None
        avg_cold_cost = sum(e.estimated_usd for e in cold_entries) / len(cold_entries)
        hypothetical_all_cold = avg_cold_cost * len(self.entries)
        actual = self.total_usd()
        if hypothetical_all_cold == 0:
            return None
        return round((1 - actual / hypothetical_all_cold) * 100, 1)

    def as_dicts(self) -> list[dict]:
        return [
            {
                "question": e.question,
                "path": e.path,
                "input_tokens": e.input_tokens,
                "output_tokens": e.output_tokens,
                "estimated_usd": round(e.estimated_usd, 5),
            }
            for e in self.entries
        ]
