from pydantic import BaseModel, Field, field_validator
from typing import List, Union
import json


class SWEIssue(BaseModel):
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str
    version: str
    fail_to_pass: List[str] = Field(alias="FAIL_TO_PASS")
    pass_to_pass: List[str] = Field(alias="PASS_TO_PASS")
    environment_setup_commit: str

    @field_validator("fail_to_pass", "pass_to_pass", mode="before")
    @classmethod
    def parse_json_list(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v
