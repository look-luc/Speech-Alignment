FROM rabbitmq:3.13-management

# Optional: Set custom environment variables (default guest/guest is fine for local dev)
# ENV RABBITMQ_DEFAULT_USER=myuser
# ENV RABBITMQ_DEFAULT_PASS=mypassword

# Optional: Copy a custom configuration file into the container
# COPY rabbitmq.conf /etc/rabbitmq/rabbitmq.conf

# Optional: Enable additional plugins if your application requires them
# RUN rabbitmq-plugins enable --offline rabbitmq_shovel rabbitmq_shovel_management

# Expose standard AMQP and Management ports
EXPOSE 5672 15672
