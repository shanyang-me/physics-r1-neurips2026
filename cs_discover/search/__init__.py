"""Search harness: the propose -> test -> refine loop that the RL trainer will plug
into (by replacing the frozen proposer with a learned one)."""
from .proposer import Proposer, MutationProposer, LLMProposer
from .evolution import evolve, SearchResult
