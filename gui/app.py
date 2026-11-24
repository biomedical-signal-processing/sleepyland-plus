"""Start the Flask web interface on the port published by Docker Compose."""

from app import app


if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',  # Bind beyond localhost so Docker can publish the service.
        port=8887
    )
