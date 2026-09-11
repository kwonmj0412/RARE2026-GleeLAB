FROM --platform=linux/amd64 pytorch/pytorch AS example-algorithm-amd64

ENV PYTHONUNBUFFERED=1

# opencv-python-headless 는 보통 libGL 불필요하지만, 일부 베이스에서 libglib 요구.
# root 로 설치 후 user 로 전환.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# 가중치(fold*.pt) + config.json 은 resources/ 에 둔다. (model/ 은 더 이상 사용 안 함)
# tarball 업로드 방식을 쓰면 GC 가 /opt/ml/model 로 마운트하므로 이 COPY 없이도 동작.
COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user rare_model.py /opt/app/
COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]