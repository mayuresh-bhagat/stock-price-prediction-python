FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
EXPOSE 5000
CMD gunicorn --workers=4 --bind 0.0.0.0:5000 app:app