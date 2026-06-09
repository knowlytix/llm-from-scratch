"""The governance harness ties the engine, policies, grounding and audit logger together."""

from __future__ import annotations

import time
import uuid
from typing import Any

from llm_from_scratch.governance.audit import AuditLogger
from llm_from_scratch.governance.grounding import GroundingCheck
from llm_from_scratch.governance.policies import InputPolicy, OutputPolicy


class GovernanceHarness:
    """A thin wrapper around an inference engine with policies and audit log."""

    def __init__(self, engine, *,
                 input_policy: InputPolicy | None = None,
                 output_policy: OutputPolicy | None = None,
                 grounding: GroundingCheck | None = None,
                 audit_logger: AuditLogger | None = None,
                 model_checkpoint: str = "unknown",
                 tokenizer_hash: str = "unknown",
                 abstain_message: str = "I cannot respond to that request.") -> None:
        self.engine = engine
        self.input_policy = input_policy or InputPolicy()
        self.output_policy = output_policy or OutputPolicy()
        self.grounding = grounding
        self.audit_logger = audit_logger
        self.model_checkpoint = model_checkpoint
        self.tokenizer_hash = tokenizer_hash
        self.abstain_message = abstain_message

    def generate(self, prompt: str, *, context: str | None = None,
                 task_type: str | None = None, **gen_kwargs) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        t0 = time.time()

        # Input policy.
        ip = self.input_policy.validate(prompt, task_type=task_type)
        if ip.status == "block":
            latency = (time.time() - t0) * 1000
            response = self.abstain_message
            op = self.output_policy.validate(response)
            record = self._log(request_id, prompt, gen_kwargs, ip, op, None, "blocked_input", latency)
            return {"response": response, "status": "blocked_input", "audit": record}

        # Generate.
        response = self.engine.generate(prompt, **gen_kwargs)
        # Output policy.
        op = self.output_policy.validate(response)
        if op.status == "block":
            latency = (time.time() - t0) * 1000
            record = self._log(request_id, prompt, gen_kwargs, ip, op, None, "blocked_output", latency)
            return {"response": self.abstain_message, "status": "blocked_output", "audit": record}

        # Grounding (optional).
        ground_score = None
        if self.grounding is not None and context is not None:
            ground_score = self.grounding.score(response, context)
            if ground_score < self.grounding.threshold:
                latency = (time.time() - t0) * 1000
                record = self._log(request_id, prompt, gen_kwargs, ip, op, ground_score, "abstained_low_grounding", latency)
                return {"response": self.abstain_message, "status": "abstained_low_grounding",
                        "grounding_score": ground_score, "audit": record}

        latency = (time.time() - t0) * 1000
        record = self._log(request_id, prompt, gen_kwargs, ip, op, ground_score, "pass", latency)
        return {"response": response, "status": "pass", "grounding_score": ground_score, "audit": record}

    def _log(self, request_id, prompt, gen_kwargs, ip, op, ground, final, latency):
        if self.audit_logger is None:
            return None
        return self.audit_logger.log(
            request_id=request_id,
            prompt=prompt,
            model_checkpoint=self.model_checkpoint,
            tokenizer_hash=self.tokenizer_hash,
            generation_config=gen_kwargs,
            input_result=ip,
            output_result=op,
            grounding_score=ground,
            final_status=final,
            latency_ms=latency,
        )
