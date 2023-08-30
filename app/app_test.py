from fastapi.testclient import TestClient
from .app import app

client = TestClient(app)


from typing import List, Optional
from pydantic import BaseModel

class Item(BaseModel):
    gcs_url: str
    description: Optional[str] = None


def test_extract_main():
    item = Item
    item.gcs_url = "gs://mlworks_seg2d_dev/upload/406868/1680499579/example5.zip"

    response = client.post("/image-retrieval",
                           json={
                               "gcs_url": "gs://mlworks_seg2d_dev/upload/406868/1680499579/example5.zip",
                               "description": None,
                           })
    print(response)
    # assert response.status_code == 200