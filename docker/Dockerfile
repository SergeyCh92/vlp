FROM python:3.11.1

WORKDIR /usr/app

COPY ./vlp_service /usr/app/vlp_service
COPY ./requirements.txt /usr/app
COPY ./libs /usr/app/libs

RUN pip install --no-cache-dir --upgrade -r /usr/app/requirements.txt
RUN pip install libs/common_rabbit-1.0.0-py3-none-any.whl  \
    /usr/app/libs/common_core-1.0.0-py3-none-any.whl \
    /usr/app/libs/gateway_service-0.1.0-py3-none-any.whl

ENV PYTHONPATH "${PYTHONPATH}:/usr/app"

CMD ["python", "vlp_service/main.py"]