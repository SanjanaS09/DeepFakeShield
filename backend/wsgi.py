"""
WSGI configuration for production deployment
Used by Gunicorn, uWSGI, or similar WSGI servers
"""

import os
import sys
from app import create_app
from config import Config, ProductionConfig

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Create application instance
# Use ProductionConfig for production deployment
config_class = ProductionConfig if os.environ.get('FLASK_ENV') == 'production' else Config
application = create_app(config_class)

if __name__ == "__main__":
    application.run()
