FROM nvcr.io/nvidia/pytorch:23.11-py3

COPY requirements.txt /workspace/
RUN pip install -r /workspace/requirements.txt

RUN git config --global --add safe.directory '*'

