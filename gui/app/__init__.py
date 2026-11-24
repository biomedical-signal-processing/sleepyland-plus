"""Create the Flask UI and register its upload, results, and history routes."""

from flask import Flask
from app.routes import bp

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static",
)
app.register_blueprint(bp)
