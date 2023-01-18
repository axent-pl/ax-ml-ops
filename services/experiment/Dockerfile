FROM python:3.9.13

COPY requirements.txt /
RUN pip install -r requirements.txt

COPY packages/ /packages/
RUN _cwd=$(pwd); for p in $(find packages -mindepth 1 -maxdepth 1 -type d); do cd $p; pip install .; cd $_cwd; done;

COPY entrypoint.sh /
COPY experiments/ /experiments/

WORKDIR /
ENTRYPOINT "./entrypoint.sh"