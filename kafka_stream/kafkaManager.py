import json
import time
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import logging

class KafkaManager:
  def __init__(self, bootstrap_server: str = 'localhost:9092'):
    self.bootstrap_server = bootstrap_server
    self.logger = logging.getLogger(__name__)
  
  def createProducer(self):
    producer = KafkaProducer(
        bootstrap_servers=self.bootstrap_server,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        retries=3,
        compression_type='gzip'
      )
    return producer

  def createConsumer(self, topic:str, group_id:str):
    consumer = KafkaConsumer(
      topic,
      bootstrap_servers = self.bootstrap_server,
      group_id=group_id,
      auto_offset_reset="earliest",
      enable_auto_commit=True,
      value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    return consumer

  def _on_send_success(self, topic: str, record_metadata):
      """Callback for successful message delivery"""
      self.logger.info(
          f"Message sent to {topic} [Partition: {record_metadata.partition}, "
          f"Offset: {record_metadata.offset}]"
      )
  
  def _on_send_error(self, excp):
      """Callback for failed message delivery"""
      self.logger.error('Error sending message', exc_info=excp)

  def sendMessage(self, producer:KafkaProducer, topic:str, message):
    try:
      future = producer.send(
            topic,
            value=message
      ).add_callback(self.on_send_success).add_errback(self.on_send_error)
      # producer.flush()
      return True
    except Exception as e:
      self.logger.error(f"Failed to send message to {topic}: {str(e)}")
      return False