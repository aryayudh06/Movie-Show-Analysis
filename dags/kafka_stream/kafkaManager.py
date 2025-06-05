import json
import time
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import logging

class KafkaManager:
  def __init__(self, bootstrap_server: str = 'localhost:9092'):
    self.bootstrap_server = bootstrap_server
    self.logger = self._setup_logger()
      
  def _setup_logger(self):
    """Configure and return a logger instance"""
    logger = logging.getLogger('kafka_manager')
    logger.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch.setFormatter(formatter)
    
    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(ch)
    
    return logger

  def createProducer(self):
    """Create Kafka producer with detailed logging"""
    self.logger.info(f"Attempting to create producer connecting to {self.bootstrap_server}")
    try:
        producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_server,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3,
            compression_type='gzip'
        )
        self.logger.info("Successfully created Kafka producer")
        return producer
    except KafkaError as e:
        self.logger.error(f"Failed to create producer: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        self.logger.critical(f"Unexpected error creating producer: {str(e)}", exc_info=True)
        raise

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