from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.schema import Sequence
from database import Base


# class Zip(Base):
#     __tablename__ = "image_retrieval_zip"
#
#     zip_id = Column(Integer, primary_key=True, index=True)
#     zip_url = Column(String(255), unique=True)
#     proj_id = Column(Integer)
#     requester_name = Column(String(50))
#     request_status = Column(Boolean)
#     time_created = Column(DateTime(timezone=True), server_default=func.now())
#     time_updated = Column(DateTime(timezone=True), onupdate=func.now())
#     start_vec_id = Column(Integer)
#     end_vec_id = Column(Integer)


# class Vector(Base):
#     __tablename__ = "image_retrieval_vec"
#     vec_id = Column(Integer, primary_key=True, index=True)
#     zip_id = Column(Integer)
#     proj_id = Column(Integer)
#     requester_name = Column(String(50))
#     request_status = Column(Boolean)
#     reference_status = Column(Boolean)
#     vec_url = Column(String(255), unique=True)
#     start_img_id = Column(Integer)
#     end_img_id = Column(Integer)
#
#     images = relationship("Image", back_populates="vector")


class Image(Base):
    __tablename__ = "image_retrieval_img"
    # img_id_seq = Sequence("img_id_seq", metadata=Base.metadata, start=1)
    # img_id = Column(Integer, img_id_seq, primary_key=True, server_default=img_id_seq.next_value())
    img_id = Column(Integer, primary_key=True, index=True)
    img_url = Column(String(255), unique=True)
    vec_url = Column(String(255))
    # vec_id = Column(Integer, ForeignKey("image_retrieval_vec.vec_id"))
    # zip_id = Column(Integer)
    proj_id = Column(Integer)
    # img_file = Column(String(255), unique=True)
    requester_name = Column(String(50))
    inference_status = Column(Boolean)
    reference_status = Column(Boolean)

    # vector = relationship("Vector", back_populates="images")
