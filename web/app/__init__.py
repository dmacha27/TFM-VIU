from flask import Flask
from flask_compress import Compress

def create_app():
    app = Flask(__name__)
    Compress(app)
    app.config['SECRET_KEY'] = 'ac6c836efab24ca0a75d485e8b9a9f6e'

    from .views import views

    app.register_blueprint(views, url_prefix='/')

    return app
