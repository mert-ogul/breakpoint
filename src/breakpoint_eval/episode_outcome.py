"""Episode outcome labels shared by simulation and report code."""

from enum import StrEnum


class EpisodeOutcome(StrEnum):
    SUCCESS = "success"
    TERMINATED_NO_SUCCESS = "terminated_no_success"
    TIME_LIMIT_TRUNCATION = "time_limit_truncation"
    POLICY_FAILED = "policy_failed"
    RUNTIME_ERROR = "runtime_error"
