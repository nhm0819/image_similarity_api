import faiss
import time
from google.cloud import storage
import numpy as np
import os
import requests
import json


class Faiss(object):
    def __init__(self,
                 dims: int = 256,
                 bucket_name: str = "image_retrieval",
                 ):
        self.state = "Not Ready"
        self.dims = dims

        # gcs
        if os.path.exists("/secrets/service_account.json"):
            self.gcp_key = "/secrets/service_account.json"
        elif os.path.exists("mlworks-key.json"):
            self.gcp_key = "mlworks-key.json"
        else:
            raise "Doesn't have gcp service account file"

        self.bucket_name = bucket_name
        self.storage_client = storage.Client.from_service_account_json(self.gcp_key)
        self.bucket = self.storage_client.get_bucket(bucket_name)

        # faiss
        self.index = faiss.IndexFlatL2(self.dims)  # build the index
        self.index_map = faiss.IndexIDMap2(self.index)

        # image retrieval api
        self.image_retrieval_api = os.environ["IMAGE_RETRIEVAL_API"]
        self.image_retrieval_api_url = f"http://{self.image_retrieval_api}"
        _ = self.add_db_vectors()

        self.state = "healthy"
        print("ready on Faiss API")

    def ping(self):
        return self.state

    def add_db_vectors(self):
        # resp = requests.get(self.image_retrieval_api_url+'/get_vector_files')
        # # vectors_dict = json.loads(resp.content.decode('utf-8'))
        # vectors_dict = resp.json()
        # '''
        # vectors_dict = {
        #     vector_id : {
        #         "zip_id": vector.zip_id,
        #         "vec_url": vector.vec_url,
        #         "start_img_id": vector.start_img_id,
        #         "end_img_id": vector.end_img_id,
        #     }
        # }
        # '''
        #
        # res_dict = {}
        # for vector_id, vector_dict in vectors_dict.items():
        #     vec_url = vector_dict["vec_url"]
        #     start_img_id = vector_dict["start_img_id"]
        #     end_img_id = vector_dict["end_img_id"]
        #     res = self.add_vector(vec_url, start_img_id, end_img_id)
        #     res_dict[vector_id] = res

        resp = requests.get(self.image_retrieval_api_url + '/get_images')
        # vectors_dict = json.loads(resp.content.decode('utf-8'))
        images_dict = resp.json()
        '''
        images_dict = {
            [
            {
            'img_url':img_url,
            'img_id': 1,
            'vec_url': 'gs://image_retrieval/vector_files/0001/00099a702231ff4384c334d551f18365.png.npz',
            'requester_name': 'hongmin',
            'reference_status': None,
            'proj_id': -1,
            'inference_status': False
            },
            ...
            ]
        }
        '''

        res_list = []
        for image_dict in images_dict:
            img_id = image_dict["img_id"]
            vec_url = image_dict["vec_url"]
            res_list.append(self.add_vector(vec_url, img_id))

        return json.dumps(res_list)

    def add_vector(self, vec_url: str, img_id: int): # start_img_id: int, end_img_id: int
        start_time = time.time()

        blob_name = vec_url.replace("gs://image_retrieval/", "")

        # read npz files in gcs bucket. (reference dataset)
        blob_ = self.bucket.blob(blob_name=blob_name)
        with blob_.open('rb') as f:
            npz = np.load(f)

        # img_ids = np.arange(start_img_id, end_img_id+1)
        img_id = np.array([img_id])

        # remove vector ids for prevent duplicated
        self.remove_vectors(img_id)

        # add vectors
        self.index_map.add_with_ids(npz, img_id)

        end_time = time.time()
        print("reading new references time :", end_time - start_time)

        return [{id: True} for id in img_id.tolist()]

    def remove_vectors(self, img_ids: np.ndarray):
        n_removed = self.index_map.remove_ids(np.array(img_ids, dtype='int64'))
        return {"n_removed": n_removed}

    def search(self, query: np.ndarray, filter_ids: list = [], k: int = 0, radius: float = 0.3):
        """
        :param query: shape=(N, 256)
        :param filter_ids: filter id range like [start_id_1, end_id_1, start_id_2, end_id_2, ...]
        :param k: num of vectors to search
        :param radius: neighbors range to search
        :return: distances: shape=(N, k)
        :return: distances_img_ids: shape=(N, k)
        """
        if query.ndim < 2:
            query = np.expand_dims(query, axis=0)

        # search in index_map
        if k > 0:
            D, I = self.index_map.search(query, k=k)
            D, I = D.flatten(), I.flatten()
            lims = [i * k for i in range(query.shape[0] + 1)]
        else:
            lims, D, I = self.index_map.range_search(query, radius)

        # filter indexes by filter_ids
        print("filter :", filter_ids)
        if len(filter_ids) > 1:
            filter = np.zeros_like(I)

            # filter[filter_ids] = 1
            filter[np.where(np.isin(I, filter_ids))] = 1

            # while filter_ids:
            #     start_imd_id = filter_ids.pop(0)
            #     end_img_id = filter_ids.pop(0)
            #
            #     filter_array = np.where((I >= start_imd_id) & (I <= end_img_id), 1, 0)
            #     filter += filter_array
            #
            # filter = filter.clip(0, 1)

        else:
            filter = np.ones_like(I)

        I_filtered = I * filter

        print("I :", I)
        print("I_filtered :", I_filtered)
        print("filter_array :", filter)
        distances, distances_img_ids = [], []
        for i in range(len(lims)-1):
            distances_ = D[lims[i]:lims[i+1]]
            distance_ids_ = I_filtered[lims[i]:lims[i + 1]]
            filter_ids = np.where(distance_ids_ != 0)[0]

            distances.append(distances_[filter_ids].tolist())
            distances_img_ids.append(distance_ids_[filter_ids].tolist())

        return distances, distances_img_ids


if __name__ == "__main__":
    # import faiss
    # import os
    # import time
    # import requests
    # from google.cloud import storage
    # import numpy as np

    gcp_key = "mlworks-key.json"
    bucket_name = "image_retrieval"
    storage_client = storage.Client.from_service_account_json(gcp_key)
    bucket = storage_client.get_bucket(bucket_name)

    dims = 256
    index = faiss.IndexFlatL2(dims)  # build the index
    index_map = faiss.IndexIDMap2(index)

    blob_name = "zip_ids/1/0000.npz"
    blob = bucket.blob(blob_name=blob_name)
    with blob.open('rb') as file:
        npz = np.load(file)

    start_img_id = 1
    img_ids = np.arange(start_img_id, start_img_id+npz.shape[0])

    index_map.add_with_ids(npz, img_ids)

    # npz = np.zeros((99999, 256))
    npz = np.random.normal(0, 0.1, (99999,256))
    np_ids = np.arange(1, 1 + 99999)
    st = time.time()
    index_map.add_with_ids(npz, np_ids)
    et = time.time()
    print(et-st)


    # query = vec[:3]
    idselector = faiss.IDSelectorRange(1, 2)
    i = 1
    query = npz[:i]
    lims, D, I = index_map.range_search(query, 10)

    # filter indexes by filter_ids
    filter_ids = [1,10]
    print("filter :", filter_ids)
    if len(filter_ids) > 1:
        filter = np.zeros_like(I)

        while filter_ids:
            start_imd_id = filter_ids.pop(0)
            end_img_id = filter_ids.pop(0)

            filter_array = np.where((I >= start_imd_id) & (I <= end_img_id), 1, 0)
            filter += filter_array

        filter = filter.clip(0, 1)

    else:
        filter = np.ones_like(I)

    I_filtered = I * filter

    print("I :", I)
    print("I_filtered :", I_filtered)
    print("filter_array :", filter)
    distances, distances_img_ids = [], []
    for i in range(len(lims) - 1):
        distances.append(D[lims[i]:lims[i + 1]].tolist())
        distances_img_ids.append(I_filtered[lims[i]:lims[i + 1]].tolist())

    distance = np.array(distances[0])
    distances_img_id = np.array(distances_img_ids[0])

    filter_index = np.where(distances_img_id != 0)[0]
    distance[filter_index]
    distances_img_id[filter_index]




    D, I = index_map.search(query, k=10)
    remove_res = index_map.remove_ids(np.array([3, 5, 7], dtype='int64'))
    distances2, distances_img_ids2 = index_map.search(query, k=10)
    np.all(distances == distances2)
    np.all(distances_img_ids == distances_img_ids2)
    rec = index_map.reconstruct(3)

    import json

    blobs = storage_client.list_blobs(bucket_name, prefix='zip_id')

    image_retrieval_api_host = os.environ['IMAGE_RETRIEVAL_API']
    image_retrieval_api_url = f"http://{image_retrieval_api_host}"
    resp = requests.get(url=image_retrieval_api_url + '/get_images')
    resp.content

    resp = requests.get(url="http://localhost:8000" + '/get_images')
    resp.json()


    vectors_dict = json.loads(resp.content.decode('utf-8'))
