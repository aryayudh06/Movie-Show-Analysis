from kafka import KafkaProducer
import json
import requests


producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
)

data = {'message': 'Halo dari Python!'}
producer.send('test-topic', value=data)
producer.flush()
