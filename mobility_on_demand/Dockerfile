FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "kddcup", "/bin/bash", "-c"]

COPY . /app/

ENTRYPOINT ["conda", "run", "-n", "kddcup", "python", "local_test.py"]

