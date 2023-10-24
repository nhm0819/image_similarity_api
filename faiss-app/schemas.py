from typing import Optional
from pydantic import BaseModel


class SimilaritySearchInput(BaseModel):
    query: list
    filter_ids: list[int]
    k: Optional[int]
    limit: Optional[int] | None = 0
    radius: float | None = 0.3


class AddVectorInput(BaseModel):
    vec_url: str
    img_id: int
    # start_img_id: int
    # end_img_id: int
    vec_id: Optional[int]
    requester_name: Optional[str]

class RemoveVectorInput(BaseModel):
    img_ids: list[int]
    requester_name: Optional[str]