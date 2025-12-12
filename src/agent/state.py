from typing import TypedDict
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

class Confidence(TypedDict):
    score: float
    reasoning: str


class Evidence(TypedDict):
    description: str
    reasoning: str
    source_node: str
    

class Hypothesis(TypedDict):
    hypothesis: str
    evidence: Evidence = None
    confidence: Confidence = None


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


class OneShotCodeGenState(TypedDict, total=False):
    # Input
    problem: str

    # Output
    generated_code: str
    
    # Intermediates
    hypothesis: str
    evidence: str
    evidence_evaluation: float
    reflection: str

    # Optional bookkeeping
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int