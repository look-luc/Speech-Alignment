# Multilingual Force Aligner
## How to run
### Database
```bash
cd database
make build build
make run
```
### RabbitMQ
```bash
cd src/rabbitmq
docker compose up -d rabbitmq
python3 -u consumer.py
python3 -u sender.py
```
### Files in src
```bash
python3 -u caching_data.py
python3 -u test_asr.py
python3 -u test_one.py
```
### Web interface
```bash
make migrate
make run
```
