# fastapi
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

# python packages
import requests
import os
import json
import time
import asyncio
import datetime
import numpy as np
from pytz import timezone

# gcs connect
from google.cloud import storage

if not os.path.exists("/secrets/service_account.json"):
    import sys
    sys.path.insert(0, os.path.join(os.getcwd(), "app"))
    IS_LOCAL = True
else:
    IS_LOCAL = False

# sqlalchemy
from sqlalchemy.orm import Session
from sqlalchemy import func, select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

# custom module
import schemas
import sql_models
# from database import SessionLocal, engine
from async_database import AsyncSessionScoped, engine, AsyncSessionMaker
from inference import image_to_vector, inference_task, sync_inference, faiss_search
from utils import is_exist_img

# gcs storage client
if IS_LOCAL:
    STORAGE_CLIENT = storage.Client.from_service_account_json("/Users/nhm/work/keys/mlworks-key.json")
    FAISS_URL = f"http://127.0.0.1:8001"
else:
    STORAGE_CLIENT = storage.Client.from_service_account_json("/secrets/service_account.json")
    FAISS_API_HOSTNAME = os.environ["FAISS_API"]
    FAISS_URL = f"http://{FAISS_API_HOSTNAME}"
BUCKET_NAME = os.environ["gcs_bucket_name"] if os.environ.get("gcs_bucket_name") else "image_retrieval"

# create db table
# sql_models.Base.metadata.drop_all(bind=engine)
# sql_models.Base.metadata.create_all(bind=engine)

# fastapi
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# inference
MODEL_NAME = os.environ["MODEL_NAME"]
SERVICE_HOSTNAME = os.environ["SERVICE_HOSTNAME"]

url = f"http://{SERVICE_HOSTNAME}"
infer_route = f"/v2/models/{MODEL_NAME}/infer"


# DB Session

async def get_db():
    db = AsyncSessionScoped()
    try:
        yield db
    finally:
        await db.close()

#
# @app.get("/get_zip_files")
# def get_zip_files(db: Session = Depends(get_db)):
#     zips = db.query(sql_models.Zip).all()
#     zips_dict = {}
#     for zip_object in zips:
#         zips_dict[zip_object.zip_id] = {"zip_url": zip_object.zip_url,
#                                         "proj_id": zip_object.proj_id,
#                                         "requester_name": zip_object.requester_name,
#                                         "request_status": zip_object.request_status,
#                                         "time_created": zip_object.time_created,
#                                         "time_updated": zip_object.time_updated,
#                                         "start_vec_id": zip_object.start_vec_id,
#                                         "end_vec_id": zip_object.end_vec_id,
#                                         }
#     return zips_dict
#
# @app.get("/get_vector_files")
# def get_vector_files(db: Session = Depends(get_db)):
#     vectors = db.query(sql_models.Vector).all()
#     vectors_dict = {}
#     for vector in vectors:
#         vectors_dict[vector.vec_id] = {
#             "vec_url": vector.vec_url,
#             "proj_id": vector.proj_id,
#             "zip_id": vector.zip_id,
#             "start_img_id": vector.start_img_id,
#             "end_img_id": vector.end_img_id,
#             "requester_name": vector.requester_name,
#             "request_status": vector.request_status,
#             "reference_status": vector.reference_status
#         }
#     return vectors_dict

@app.get("/get_images")
async def get_images(db: AsyncSession = Depends(get_db)):
    query = select(sql_models.Image)
    # images = await db.query(sql_models.Image).all()
    res = await db.execute(query)
    images = res.scalars().all()

    images_list = [image.__dict__ for image in images]
    _ = [image.pop('_sa_instance_state', None) for image in images_list]
    # _ = [image.pop('vector', None) for image in images_list]

    # images_dict = {}
    # for image in images:
    #     images_dict[image.img_id] = {
    #         "img_file": image.img_file,
    #         "vec_id": image.vec_id,
    #         "zip_id": image.zip_id,
    #         "proj_id": image.proj_id,
    #         "vec_url": image.vec_url,
    #         "requester_name": image.requester_name,
    #         "reference_status": image.reference_status,
    #         "inference_status": image.inference_status
    #     }
    json_response = jsonable_encoder(images_list)
    return JSONResponse(json_response)


@app.get("/get_image_ids")
async def get_image_ids(db: AsyncSession = Depends(get_db)):
    # images = db.query(sql_models.Image.img_id, sql_models.Image.img_url).all()

    query = select(sql_models.Image.img_id, sql_models.Image.img_url)
    # images = await db.query(sql_models.Image).all()
    res = await db.execute(query)
    images = res.all()

    images_tuple = [tuple(image) for image in images]

    json_response = jsonable_encoder(images_tuple)
    return JSONResponse(json_response)


@app.post("/similarity_search")
async def similarity_search(search_input: schemas.SearchGCSInput, db: AsyncSession = Depends(get_db)):
    start_time = time.time()

    gcs_url_list = search_input.gcs_url_list
    proj_id = search_input.proj_id
    k = search_input.k
    limit = search_input.limit
    radius = search_input.radius

    # image to vector
    npz = await inference_task(gcs_url_list, data_type="gcs_url")

    # get filter ids
    # filter_ids = search_input.filter_ids # filter ids : [start_id_1, end_id_1, start_id_2, end_id_2, ...]
    filter_url_list = search_input.filter_url_list  # filter ids : [start_id_1, end_id_1, start_id_2, end_id_2, ...]

    filter_ids = []
    for filter_url in filter_url_list:
        res = await db.execute(select(sql_models.Image.img_id).where(sql_models.Image.img_url == filter_url))
        img_id = res.scalar()
        if img_id is not None:
            filter_ids.append(img_id)
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Filter Error : {filter_url} doesn't exist",
            )

    # # similarity search
    # search_req_dict = {"query": npz[:, 1:].tolist(),
    #                    "filter_ids": filter_ids,
    #                    # "k": k,
    #                    "radius": radius}
    # resp = requests.post(url=f"{FAISS_URL}/similarity_search", json=search_req_dict)
    # resp_json = json.loads(resp.content)

    resp_json = await faiss_search(url=f"{FAISS_URL}/similarity_search",
                                   query=npz[:, 1:].tolist(),
                                   filter_ids=filter_ids,
                                   k=k,
                                   limit=limit,
                                   radius=radius)
    print("resp_json : ", resp_json)
    # images = await db.query(sql_models.Image).all()
    resp_json["img_urls"] = []
    for img_ids in resp_json["distances_img_ids"]:
        img_url = []
        for img_id in img_ids:
            res = await db.execute(select(sql_models.Image.img_url).where(sql_models.Image.img_id == img_id))
            imgurl = res.scalar()
            print(imgurl)
            img_url.append(imgurl)
        resp_json["img_urls"].append(img_url)

    resp_json["success"] = np.array(npz[:,0], dtype=bool).tolist()

    print(resp_json)

    print("search time :", time.time() - start_time)
    return resp_json


# @app.post("/similarity_search_img",
#           description="img_as_text : read image file to bytes --> base64.encode --> decode('utf-8')")
# def similarity_search_img(search_input: schemas.SearchImageInput):
#     img_as_text_list = search_input.img_as_text_list
#     proj_id = search_input.proj_id
#     # k = search_input.k
#     radius = search_input.radius
#     filter_ids = search_input.filter_ids
#
#     # image to vector
#     npz = asyncio.run(inference_task(inference_list=img_as_text_list, data_type="img_as_text"))
#
#     # similarity search
#     search_req_dict = {"query": npz[:, 1:].tolist(),
#                        "filter_ids": filter_ids,
#                        # "k": k,
#                        "radius": radius}
#     resp = requests.post(url=f"{FAISS_URL}/similarity_search", json=search_req_dict)
#     resp_json = json.loads(resp.content)
#
#     print("response status :", resp.status_code)
#     return resp_json


# @app.post("/add_zip_file")
# def add_zip_file(add_zip_file_input: schemas.AddZipFileInput,
#                  db: Session = Depends(get_db)):
#
#     # parameter setting
#     zip_url = add_zip_file_input.zip_url
#     requester_name = add_zip_file_input.requester_name
#     proj_id = add_zip_file_input.proj_id
#
#     time_created = datetime.datetime.now(timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S")
#     extract_folder = f"/app/extract/{time_created}/"
#
#     if IS_LOCAL:
#         zip_url = "gs://mlworks_seg2d_dev/upload/406868/1680499579/example5.zip"
#         extract_folder = f"./{time_created}/"
#
#     # extract zip file
#     if is_exist_zip(db=db, zip_url=zip_url):
#         return {"zip_url": zip_url, "requests_status": "failed", "description": "already exist url"}
#
#     if not os.path.exists(os.path.dirname(extract_folder)):
#         os.makedirs(os.path.dirname(extract_folder), exist_ok=True)
#
#     img_list = extract(STORAGE_CLIENT, zip_url, extract_folder=extract_folder)
#     if isinstance(img_list, str):
#         return "Cannot find gcs file"
#     num_files = len(img_list)
#
#     # query zip id
#     max_zip_id = db.query(func.max(sql_models.Zip.zip_id)).scalar()
#     zip_id = int(max_zip_id) + 1 if max_zip_id is not None else 1
#
#     # calculate num of vectors
#     n_img_per_task = int(os.environ["n_img_per_vector"])
#     div_vectors = num_files // n_img_per_task
#     mod_vectors = num_files % n_img_per_task
#     n_vectors = div_vectors + 1
#
#     # query vec id
#     max_vec_id = db.query(func.max(sql_models.Vector.vec_id)).scalar()
#     start_vec_id = int(max_vec_id)+1 if max_vec_id is not None else 1
#     end_vec_id = start_vec_id + n_vectors - 1
#
#     # insert Zip object
#     zip_schema = schemas.Zip(zip_id=zip_id, zip_url=zip_url, proj_id=proj_id, requester_name=requester_name,
#                              start_vec_id=start_vec_id, end_vec_id=end_vec_id,
#                              time_created=time_created)
#     zip_model = sql_models.Zip(**zip_schema.dict())
#     db.add(zip_model)
#     db.commit()
#     db.refresh(zip_model)
#
#     # insert Vector objects
#     vector_schemas = [
#         {"vec_id": vec_id, "zip_id": zip_id, "proj_id": proj_id, "requester_name": requester_name}
#         for vec_id in range(start_vec_id, end_vec_id + 1)
#     ]
#     db.bulk_insert_mappings(
#         sql_models.Vector,
#         vector_schemas,
#     )
#     db.commit()
#
#     # query Image object
#     max_img_id = db.query(func.max(sql_models.Image.img_id)).scalar()
#     start_img_id = int(max_img_id)+1 if (max_img_id is not None) and (max_img_id != 0) else 1
#     end_img_id = start_img_id + num_files - 1
#
#     # insert image objects
#     image_item = schemas.Image
#     image_item.zip_url = zip_url
#
#     image_schemas = [
#             {"img_id": img_id, "vec_id": start_vec_id+int(img_id//n_img_per_task),
#              "proj_id": proj_id, "requester_name": requester_name}
#             for img_id in range(start_img_id, end_img_id+1)
#         ]
#     db.bulk_insert_mappings(
#         sql_models.Image,
#         image_schemas,
#     )
#     db.commit()
#
#     # inference task(async request to model server) per n_img_per_task
#     bucket = STORAGE_CLIENT.get_bucket(BUCKET_NAME)
#
#     # inference and db update
#     num_failed = 0
#     vecs_dict = {}
#     for q in range(div_vectors+1):
#         vec_id = q+start_vec_id
#         offset_id = q * n_img_per_task
#         if q == div_vectors:
#             _img_list = img_list[offset_id:offset_id + mod_vectors]
#         else:
#             _img_list = img_list[offset_id:(q + 1) * n_img_per_task]
#
#         # image to vector (request to model server)
#         st = time.time()
#         npz = asyncio.run(image_to_vector(_img_list, data_type="local_file"))
#
#         # vector file save to numpy array
#         vec_filename = f"{q:04d}.npz"
#         with open(extract_folder+vec_filename, 'wb') as npz_file:
#             np.save(npz_file, npz[:, 1:])
#
#         # vector file upload to gcs (upload path = 'gs://image_retrieval/zip_id/{zip_id}/{file_name}')
#         vec_blob = f"zip_ids/{zip_id}/{vec_filename}"
#         vec_url = f"gs://{BUCKET_NAME}/{vec_blob}"
#         blob = bucket.blob(vec_blob)
#         blob.upload_from_filename(extract_folder+vec_filename)
#
#         et = time.time()
#         print(et-st)
#
#         # update image objects
#         update_images = [{"img_id": img_list_idx + offset_id + start_img_id,
#                           "vec_id": vec_id,
#                           "zip_id": zip_id,
#                           "img_file": img_file.replace(extract_folder, ""),
#                           "vec_url": vec_url,
#                           "inference_status": True,
#                           "reference_status": True}
#                          if npz[img_list_idx, 0] == 1
#                          else {"img_id": img_list_idx + offset_id + start_img_id,
#                                "vec_id": vec_id,
#                                "zip_id": zip_id,
#                                "img_file": img_file.replace(extract_folder, ""),
#                                "vec_url": vec_url,
#                                "inference_status": False,
#                                "reference_status": False}
#                          for img_list_idx, img_file in enumerate(_img_list)]
#
#         db.bulk_update_mappings(
#             sql_models.Image,
#             update_images,
#         )
#         db.commit()
#
#         # update vecter object
#         vecs_dict[vec_id] = {"vec_url": vec_url, "start_img_id": start_img_id, "end_img_id": end_img_id}
#         vec_model = db.query(sql_models.Vector).filter_by(vec_id=vec_id).update(
#             vecs_dict[vec_id]
#         )
#         db.commit()
#
#         # failed image count
#         num_failed += npz.shape[0] - np.count_nonzero(npz[:, 0])
#
#     # remove files
#     shutil.rmtree(extract_folder)
#
#     # update zip object
#     time_updated = datetime.datetime.now(timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S")
#     zip_schema.time_updated = time_updated
#     zip_schema.request_status = 1
#     zip_model = db.query(sql_models.Zip).filter_by(zip_id=zip_id).update(
#         {"time_updated": time_updated, "request_status": 1}
#     )
#     db.commit()
#
#     # add vectors in faiss memory
#     update_vectors = []
#     for vec_id, vec_dict in vecs_dict.items():
#         resp = requests.post(f"{FAISS_URL}/add_vector", json=vec_dict)
#         print(f"status: {resp.status_code}, vec_id: {vec_id}")
#         update_vector = {"vec_id": vec_id, "reference_status": True} if resp.status_code == 200 \
#             else {"vec_id": vec_id, "reference_status": False}
#         update_vectors.append(update_vector)
#
#     db.bulk_update_mappings(
#         sql_models.Vector,
#         update_vectors,
#     )
#     db.commit()
#
#     resp_json = {"zip_url": zip_url,
#                  "failed_count": num_failed,
#                  "requests_status": True}
#
#     return resp_json


# @app.post("/add_vector")
# async def add_vector(add_vector_input: schemas.AddVectorInput, db: AsyncSession = Depends(get_db)):
#     # vec_url = add_vector_input.vec_url
#     # start_img_id = add_vector_input.start_img_id
#     # end_img_id = add_vector_input.end_img_id
#     # img_id = add_vector_input.img_id
#     img_url = add_vector_input.img_url
#
#     # image_object = db.query(sql_models.Image).filter_by(img_url=img_url).one()
#
#     query = select(sql_models.Image).where(img_url=img_url)
#     res = await db.execute(query)
#     image_object = res.scalars().one()
#
#     img_id = image_object.img_id
#     vec_url = image_object.vec_url
#
#     # vec_dict = {"vec_url": vec_url, "start_img_id": start_img_id, "end_img_id": end_img_id}
#     req_dict = {"vec_url": vec_url, "img_id": img_id}
#     resp = requests.post(f"{FAISS_URL}/add_vector", json=req_dict)
#
#     return resp.json()


@app.post("/add_img_file")
async def add_img_file(add_img_file_input: schemas.AddImgFileInput,
                 db: AsyncSession = Depends(get_db)):

    # parameter setting
    img_url = add_img_file_input.img_url
    requester_name = add_img_file_input.requester_name
    proj_id = add_img_file_input.proj_id

    today = datetime.datetime.now(timezone('Asia/Seoul')).strftime("%Y-%m-%d")
    save_folder = f"/app/images/{today}/"

    if IS_LOCAL:
        if img_url is None or img_url == "string":
            img_url = "gs://mlworks_objdetect2d_dev/results/input/1/CCTV.jpg"
        save_folder = f"./{today}/"

    # check img files
    if await is_exist_img(db=db, img_url=img_url):
        return {"img_url": img_url, "description": "already exist url"}

    # image to vector (request to model server)
    st = time.time()
    npz = await image_to_vector([img_url], data_type="gcs_url")
    # npy = sync_inference(img=img_url, data_type="gcs_url")
    # npz = np.expand_dims(npy, axis=0)
    et = time.time()
    print("inference time :", et - st)

    # check inference status
    inference_status = True
    if npz[0, 0] == 0:
        inference_status = False
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model Inference Error",
        )

    # vector file save to numpy array
    split_path = img_url.split('/')
    bucket_name = split_path[2]
    blob_name = '/'.join(split_path[3:])
    vec_filename = f"{blob_name}.npz"

    if not os.path.exists(os.path.dirname(save_folder + vec_filename)):
        os.makedirs(os.path.dirname(save_folder + vec_filename), exist_ok=True)

    print("npz shape:", npz.shape)

    # vector file upload to gcs (upload path = gs://image_retrieval/zip_id/{zip_id}/{file_name}')
    vec_blob = f"vector_files/{vec_filename}"
    vec_url = f"gs://{BUCKET_NAME}/{vec_blob}"
    bucket = STORAGE_CLIENT.get_bucket(BUCKET_NAME)
    blob = bucket.blob(vec_blob)

    with blob.open('wb') as npz_file:
        np.save(npz_file, npz[:, 1:])

    et = time.time()
    print("save vector time :", et - st)

    # # query vec id
    # max_vec_id = db.query(func.max(sql_models.Vector.vec_id)).scalar()
    # vec_id = int(max_vec_id) + 1 if max_vec_id is not None else 1

    # # insert Vector objects
    # vector_schema = schemas.Vector(
    #     # vec_id=vec_id,
    #     vec_url=vec_url,
    #     proj_id=proj_id,
    #     requester_name=requester_name,
    # )
    # vector_model = sql_models.Vector(**vector_schema.dict())
    # db.add(vector_model)
    # db.commit()
    # # db.refresh(vector_model)
    # vec_id = vector_model.vec_id

    # # query Image object
    # max_img_id = db.query(func.max(sql_models.Image.img_id)).scalar()
    # img_id = int(max_img_id) + 1 if max_img_id is not None else 1

    # insert Image objects
    image_schema = schemas.Image(
        # img_id=img_id,
        img_url=img_url,
        # vec_id=vec_id,
        # img_file=img_url,
        vec_url=vec_url,
        proj_id=proj_id,
        inference_status=inference_status,
        requester_name=requester_name,
    )
    image_model = sql_models.Image(**image_schema.dict())
    db.add(image_model)
    await db.commit()
    # db.refresh(image_model)
    img_id = image_model.img_id

    # add vectors in faiss memory
    req_dict = {"vec_url": vec_url, "img_id": img_id}
    resp = requests.post(f"{FAISS_URL}/add_vector", json=req_dict)

    # update vector object
    reference_status = True
    if resp.status_code != 200:
        reference_status = False

    # vector_model.start_img_id = img_id
    # vector_model.end_img_id = img_id
    # vector_model.request_status = True
    image_model.reference_status = reference_status
    await db.commit()

    resp_json = {"img_id": img_id, "img_url": img_url, "vec_url": vec_url, "requests_status": inference_status,
                 "reference_status": reference_status}
    resp_json = jsonable_encoder(resp_json)
    return JSONResponse(resp_json)


@app.post("/remove_vector")
async def remove_vector(remove_input: schemas.RemoveVectorInput, db: AsyncSession = Depends(get_db)):
    img_ids = remove_input.img_ids
    requester_name = remove_input.requester_name

    remove_req_dict = {"img_ids": img_ids,
                       "requester_name": requester_name}

    resp = requests.post(url=f"{FAISS_URL}/remove_vectors", json=remove_req_dict)
    assert resp.status_code == 200

    # Update DB
    # update_images = [
    #     {"img_id": img_id, "reference_status": False}
    #     for img_id in img_ids
    # ]
    # db.bulk_update_mappings(
    #     sql_models.Image,
    #     update_images,
    # )


    query = delete(sql_models.Image).where(sql_models.Image.img_id.in_(img_ids))
    res = await db.execute(query)
    await db.commit()

    print("response status :", resp.status_code)

    remove_list = [
        {img_id: "removed"}
        for img_id in img_ids
    ]
    resp_json = jsonable_encoder(remove_list)
    return JSONResponse(resp_json)


@app.get("/ping")
async def ping():
    resp = requests.get(url=f"{FAISS_URL}/ping")
    resp_json = resp.json()
    if resp_json["status"] != "healthy":
        return {"status": "faiss unhealthy"}
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # # sqlalchemy
    # from sqlalchemy.orm import Session
    # from sqlalchemy import func
    #
    # # custom module
    # import schemas
    # import sql_models
    # from database import SessionLocal, engine
    #
    # img_url = "gs://mlworks_objdetect2d_dev/results/input/1/CCTV.jpg"
    #
    # sql_models.Base.metadata.drop_all(bind=engine)
    # sql_models.Base.metadata.create_all(bind=engine)
    #
    # db = SessionLocal()
    # image_schema = schemas.Image(img_id=None, img_url=img_url)
    # image_model = sql_models.Image(**image_schema.dict())
    # db.add(image_model)
    # db.commit()
    #
    # image_model.reference_status = False
    # db.commit()
    #
    # image_model.img_id

    img_url_list = ["gs://cwmla/0001/000321b3883cc38ff4698f74dc5fa177.png",
                    "gs://cwmla/0001/000321b3883cc38ff4698f74dc5fa177_draw.png",
                    "gs://cwmla/0001/00099a702231ff4384c334d551f18365.png",
                    "gs://cwmla/0001/0033db034b787c1c804b1001d9401619.png"]
