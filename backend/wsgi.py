"""
WSGI entrypoint for production servers such as Gunicorn.
"""

import os

from app import app as application


if __name__ == "__main__":
    application.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "5000")),
    )
