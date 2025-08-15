from pydantic import BaseModel


class Proof(BaseModel):
    contents: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.contents)


class Theorem(BaseModel):
    id: int
    type: str
    label: str
    title: str
    categories: list[str]
    contents: list[str]
    refs: list[str]
    ref_ids: list[int]
    proofs: list[Proof]


class Dataset(BaseModel):
    theorems: list[Theorem]


class NaturalProofs(BaseModel):
    dataset: Dataset
