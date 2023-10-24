FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11
#FROM gcr.io/crowdworks-aiml-ops/faiss-api:v4

# Langauage setting
#RUN apt-get update && apt-get install -y locales git && \
#    localedef -f UTF-8 -i ko_KR ko_KR.UTF-8
#
#ENV LANG=ko_KR.UTF-8 \
#    LANGUAGE=ko_KR.UTF-8 \
#    LC_ALL=ko_KR.UTF-8 \
#    PYTHONIOENCODING=utf-8

# upgrade pip
RUN pip install --upgrade pip

# install faiss
RUN pip install faiss-cpu==1.7.4

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
# RUN pip install google-cloud-storage
# RUN pip install python-multipart

# remove cache
RUN rm -rf /root/.cache/pip

WORKDIR /faiss-app
COPY ./ ./

# key file
ENV GOOGLE_APPLICATION_CREDENTIALS mlworks-key.json

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# workers, worker-class