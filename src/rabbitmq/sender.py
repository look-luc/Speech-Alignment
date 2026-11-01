import pika

# 1. Establish a connection and open a channel
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# 2. Declare a queue (creates it if it doesn't exist)
channel.queue_declare(queue='task_queue', durable=True)

# 3. Publish a message to the default exchange (routing directly to the queue)
message = "Hello, RabbitMQ!"
channel.basic_publish(
    exchange='', # Empty string uses the default direct exchange
    routing_key='task_queue', # Routes exactly to the queue name
    body=message,
    properties=pika.BasicProperties(
        delivery_mode=pika.DeliveryMode.Persistent # Make message survive server restarts
    )
)

print(f" [x] Sent '{message}'")
connection.close()
