FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py auth_utils.py database.py dbInfo.py ./

RUN mkdir -p uploads

EXPOSE 3001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "3001"]
