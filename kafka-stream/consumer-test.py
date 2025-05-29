import json
from kafka import KafkaConsumer


consumer = KafkaConsumer(
    'test-topic',
    bootstrap_servers = 'localhost:9092',
    value_deserializer = lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset = "earliest",
    enable_auto_commit = True,
    group_id = "group-1-test",
)

print("waiting message...")
for message in consumer:
    print ("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
                                          message.offset, message.key,
                                          message.value))