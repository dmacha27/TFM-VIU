from flask import Flask
from flask_compress import Compress

def create_app():
    app = Flask(__name__)
    Compress(app)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'

    from .views import views

    app.register_blueprint(views, url_prefix='/')

    return app
