from datetime import datetime
from typing import Optional
from pydantic import Field, BaseModel


class Image(BaseModel):
    img_id: Optional[int] = Field(defulat=None, primary_key=True)
    img_url: Optional[str]
    vec_url: Optional[str]
    # vec_id: Optional[int]
    # zip_id: Optional[int]
    proj_id: int | None = -1
    # img_file: Optional[str]
    requester_name: Optional[str]
    inference_status: Optional[bool]
    reference_status: Optional[bool]

    # class Config:
    #     orm_mode = True


# class Vector(BaseModel):
#     vec_id: Optional[int]
#     zip_id: Optional[int]
#     vec_url: Optional[str]
#     proj_id: int | None = -1
#     start_img_id: Optional[int]
#     end_img_id: Optional[int]
#     requester_name: Optional[str]
#     request_status: Optional[bool]
#     reference_status: Optional[bool]
#
#     images: list[Image] = []
#
#     class Config:
#         orm_mode = True
#
#
# class Zip(BaseModel):
#     zip_id: int
#     zip_url: str
#     proj_id: int | None = -1
#     time_created: Optional[datetime] = Field(default_factory=datetime.now)
#     time_updated: Optional[datetime] = Field(default_factory=datetime.now)
#     start_vec_id: Optional[int]
#     end_vec_id: Optional[int]
#     requester_name: Optional[str]
#     request_status: Optional[bool]


class AddZipFileInput(BaseModel):
    zip_url: str
    proj_id: int | None = -1
    requester_name: str


class AddImgFileInput(BaseModel):
    img_url: str
    proj_id: int | None = -1
    requester_name: str


class AddVectorInput(BaseModel):
    # vec_url: str
    # start_img_id: int
    # end_img_id: int
    # img_id: int
    img_url: Optional[str]
    # vec_id: Optional[int]
    requester_name: Optional[str]


class SearchGCSInput(BaseModel):
    gcs_url_list: list[str]
    proj_id: int | None = -1
    # filter_ids: list[int]
    filter_url_list: list[str]
    k: Optional[int] | None = 0
    limit: Optional[int] | None = 0
    radius: Optional[float] | None = 0.3
    requester_name: Optional[str]


class SearchImageInput(BaseModel):
    img_as_text_list: list[str]
    proj_id: int | None = -1
    filter_ids: list[int]
    # k: Optional[int]
    radius: Optional[float] | None = 0.3
    requester_name: Optional[str]


class RemoveVectorInput(BaseModel):
    img_ids: list[int]
    requester_name: Optional[str]
