from Faiss import Faiss
import time
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from schemas import SimilaritySearchInput, AddVectorInput, RemoveVectorInput
import numpy as np

faiss = Faiss(
    dims=256,
    bucket_name="image_retrieval"
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/similarity_search")
def similarity_search(item: SimilaritySearchInput):
    start_time = time.time()
    query = np.array(item.query, dtype=np.float32)
    k = item.k if item.k is not None else 0
    limit = item.limit if item.limit is not None else 0
    radius = item.radius
    filter_ids = item.filter_ids

    distances, distances_img_ids = faiss.search(query=query, k=k, radius=radius, limit=limit, filter_ids=filter_ids)
    print("search time(s) :", time.time()-start_time)
    resp_content = {"distances": distances,
                    "distances_img_ids": distances_img_ids,
                    "description": ""}
    # return resp_content
    resp_json = jsonable_encoder(resp_content)
    return JSONResponse(resp_json)


@app.post("/add_vector")
def add_vector(item: AddVectorInput):
    vec_url = item.vec_url
    img_id = item.img_id
    # start_img_id = item.start_img_id
    # end_img_id = item.end_img_id

    res_dict = faiss.add_vector(vec_url=vec_url, img_id=img_id)
    return res_dict

@app.post("/remove_vectors")
def remove_vectors(item: RemoveVectorInput):
    img_ids = item.img_ids
    requester_name = item.requester_name

    res = faiss.remove_vectors(img_ids=np.array(img_ids))
    return res


@app.get("/ping")
def faiss_api_ping():
    return {"status": faiss.ping()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
