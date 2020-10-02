FROM tensorflow/tensorflow:1.15.4-gpu-py3


WORKDIR /wd

ADD requirements.txt /wd/requirements.txt

RUN pip install -r requirements.txt

ENV TF_XLA_FLAGS=--tf_xla_auto_jit=2

CMD ['bash']

