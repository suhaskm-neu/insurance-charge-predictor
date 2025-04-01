# To use this Dockerfile, first build the Docker image:
#   docker build -t my-streamlit-app .
#
# Then, run the app:
#   docker run -p 8501:8501 my-streamlit-app
#
# Open a web browser and navigate to http://localhost:8501 to view the app.

FROM python:3.12-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
