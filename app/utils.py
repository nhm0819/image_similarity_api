from google.cloud import storage
import time
import zipfile
import glob
import os
import sql_models
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


# def is_exist_zip(db: AsyncSession, zip_url: str):
#     select_list = db.query(sql_models.Zip).filter(sql_models.Zip.zip_url == zip_url).all()
#     if len(select_list) > 0:
#         return 1
#     return 0
#
#
# def is_exist_vec(db: Session, vec_url: str):
#     select_list = db.query(sql_models.Vector).filter(sql_models.Vector.vec_url == vec_url).all()
#     if len(select_list) > 0:
#         return 1
#     return 0


async def is_exist_img(db: AsyncSession, img_url: str):
    # select_list = db.query(sql_models.Image).filter(sql_models.Image.img_url == img_url).all()

    query = select(sql_models.Image).where(sql_models.Image.img_url == img_url)
    res = await db.execute(query)
    select_list = res.scalars().all()

    if len(select_list) > 0:
        return 1
    return 0


def extract(storage_client, zip_url, extract_folder):
    split_path = zip_url.split('/')
    bucket_name = split_path[2]
    blob_name = '/'.join(split_path[3:])

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name=blob_name)
    if not blob.exists():
        return "Cannot find file"

    st = time.time()
    filename = os.path.basename(blob.name)
    blob.download_to_filename(filename)
    et = time.time()
    print("download time :", et - st)

    st = time.time()
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    et = time.time()
    print("extract time :", et - st)

    ext_list = ["jpg", "jpeg", "png", "JPG", "PNG", "JPEG"]
    file_list = glob.glob(os.path.join(extract_folder, "**/*.*"), recursive=True)
    img_list = [filepath for filepath in file_list if filepath[-3:] in ext_list]

    return img_list
