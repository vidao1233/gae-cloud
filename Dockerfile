FROM python:3.7
WORKDIR /Home
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8080
COPY . /Home
CMD streamlit run --server.port 8080 Home.py