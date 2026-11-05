import os
import time

import pika

# Default to 'localhost' if running outside Docker, or use the container name
rabbitmq_host = os.environ.get('RABBITMQ_HOST', 'localhost')

connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host))

channel = connection.channel()

channel.queue_declare(queue='task_queue', durable=True)
print(' [*] Waiting for messages. To exit press CTRL+C')

# Define what to do when a message arrives
def callback(ch, method, properties, body):
    print(f" [x] Received {body.decode()}")
    time.sleep(2) # Simulate processing time
    print(" [x] Done")
    # Acknowledge that the message was processed successfully
    ch.basic_ack(delivery_tag=method.delivery_tag)

# Fair dispatch: Don't give more than 1 message to a worker at a time
channel.basic_qos(prefetch_count=1)

# Tell RabbitMQ to use our callback function for 'task_queue'
channel.basic_consume(queue='task_queue', on_message_callback=callback)

channel.start_consuming()
