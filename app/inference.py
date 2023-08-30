import os
import time
import requests
import base64
import ujson
import aiohttp
import asyncio
import numpy as np
import json


MODEL_NAME = os.environ["MODEL_NAME"]
SERVICE_HOSTNAME = os.environ["SERVICE_HOSTNAME"]

url = f"http://{SERVICE_HOSTNAME}"
infer_route = f"/v2/models/{MODEL_NAME}/infer"
# headers = {"HOST": f"{SERVICE_HOSTNAME}"}

request_form = {
    "inputs": [
        {
            "name": "image",
            "shape": [1],
            "datatype": "BYTES",
            "parameters": {},
            "data": []
        }
    ]
}


async def inference_task(inference_list: list, data_type: str = "gcs_url"):

    tasks = []
    async with aiohttp.ClientSession(json_serialize=ujson.dumps) as session:
        for img in inference_list:
            tasks.append(asyncio.create_task(inference(session, img=img, data_type=data_type)))

        npy_list = await asyncio.gather(*tasks)

    npz = np.stack(npy_list)
    if npz.ndim < 2:
        npz = np.expand_dims(npz, axis=0)

    return npz


async def inference(session: aiohttp.ClientSession, img: str = None, data_type="gcs_url"):
    st = time.time()
    data = img

    # gcs url
    if data_type == "gcs_url":
        pass

    # local file read
    elif data_type == "local_file":
        with open(img, "rb") as image_file:
            image_binary = image_file.read()
            encoded_string = base64.b64encode(image_binary)
            data = encoded_string.decode('utf-8')

        # # image read
        # img = cv2.imread(filepath)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image_binary = cv2.imencode('.JPEG', img)[1].tobytes()

    elif data_type == "img_as_text":
        pass

    request_data = {
        "inputs": [
            {
                "name": "image",
                "shape": [1],
                "datatype": "BYTES",
                "parameters": {"data_type": data_type},
                "data": [data]
            }
        ]
    }

    async with session.post(url+infer_route, json=request_data, timeout=aiohttp.ClientTimeout(total=60*5)) as resp:
        status = resp.status
        print(resp.status)
        resp_json = await resp.json()

    if int(status) != 200:
        print("inference response error!")
        return np.zeros(256+1, dtype=np.float32)

    try:
        npy = np.array([1]+resp_json["outputs"][0]["data"][0], dtype=np.float32)
    except:
        print("response doesn't have output data")
        if data_type != "img_as_text":
            print("filename :", img)
        print("response :", resp)
        return np.zeros(256+1, dtype=np.float32)

    et = time.time()
    print(et - st)

    return npy


async def image_to_vector(img_list: list, data_type: str = "gcs_url"):
    start = time.time()
    async with aiohttp.ClientSession(json_serialize=ujson.dumps) as session:
        if len(img_list) == 1:
            npy_list = [await inference(session, img=img_list[0], data_type=data_type)]
            print("npy_shape!!! :", npy_list[0].shape)
        else:
            tasks = [asyncio.create_task(inference(session, img=img, data_type=data_type)) for img in img_list]
            npy_list = await asyncio.gather(*tasks)
    npz = np.stack(npy_list)
    print("npz_shape!!! :", npz.shape)
    # print(results)
    end = time.time()
    print(f'Send {len(img_list)} requests, time consuming:{end - start}')
    return npz


def sync_inference(img: str = None, data_type="gcs_url"):
    st = time.time()
    data = img

    # gcs url
    if data_type == "gcs_url":
        pass

    # local file read
    elif data_type == "local_file":
        with open(img, "rb") as image_file:
            image_binary = image_file.read()
            encoded_string = base64.b64encode(image_binary)
            data = encoded_string.decode('utf-8')

        # # image read
        # img = cv2.imread(filepath)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image_binary = cv2.imencode('.JPEG', img)[1].tobytes()

    elif data_type == "img_as_text":
        pass

    request_data = {
        "inputs": [
            {
                "name": "image",
                "shape": [1],
                "datatype": "BYTES",
                "parameters": {"data_type": data_type},
                "data": [data]
            }
        ]
    }

    resp = requests.post(url+infer_route, json=request_data)
    status = resp.status_code
    print(status)
    resp_json = resp.json()

    if int(status) != 200:
        print("inference response error!")
        return np.zeros(256+1, dtype=np.float32)

    try:
        npy = np.array([1]+resp_json["outputs"][0]["data"][0], dtype=np.float32)
    except:
        print("response doesn't have output data")
        if data_type != "img_as_text":
            print("filename :", img)
        print("response :", resp)
        return np.zeros(256+1, dtype=np.float32)

    et = time.time()
    print(et - st)

    return npy



async def search(session: aiohttp.ClientSession, url: str, query: list, filter_ids: list, radius: float):
    st = time.time()

    # similarity search
    search_req_dict = {"query": query,
                       "filter_ids": filter_ids,
                       "radius": radius}

    async with session.post(url=url, json=search_req_dict, timeout=aiohttp.ClientTimeout(total=60*5)) as resp:
        resp_json = await resp.json()

    print("resp_json :", resp_json)

    return resp_json


async def faiss_search(url: str, query: list, filter_ids: list, radius: float):
    start = time.time()
    async with aiohttp.ClientSession(json_serialize=ujson.dumps) as session:
        resp = await search(session, url=url, query=query, filter_ids=filter_ids, radius=radius)

    print(resp)
    end = time.time()
    return resp




if __name__ == "__main__":
    from PIL import Image
    import io

    img_url = 'gs://mlworks_objdetect2d_dev/results/input/1/CCTV.jpg'
    request_form["inputs"][0]["data"] = [img_url]

    # blob name
    split_path = img_url.split('/')
    bucket_name = split_path[2]
    blob_name = '/'.join(split_path[3:])

    from google.cloud import storage
    STORAGE_CLIENT = storage.Client.from_service_account_json("/secrets/service_account.json")
    bucket = STORAGE_CLIENT.get_bucket(bucket_name)
    blob = bucket.blob(blob_name=blob_name)
    image_data = blob.download_as_bytes()
    img = Image.open(io.BytesIO(image_data))

    # with blob.open('rb') as image_file:
    with open("test.jpg", 'rb') as image_file:
        image_binary = image_file.read()
        encoded_string = base64.b64encode(image_binary)
        img_as_text = encoded_string.decode('utf-8')

    request_form["inputs"][0]["data"] = [img_as_text]


    st = time.time()
    res = requests.post(url+infer_route, json=request_form)
    # res = requests.post(url + infer_route, json=request_data)
    print(time.time() - st)
