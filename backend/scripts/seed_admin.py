"""
backend/scripts/seed_admin.py

Creates initial users in the database.
Run once after setting up the database:

    cd backend
    python scripts/seed_admin.py

Users created
-------------
admin@fakenews.dev   / admin123!      role: admin
checker@fakenews.dev / checker123!    role: fact_checker
publisher@fakenews.dev / publisher123! role: publisher
"""

import sys
import os

# Make backend importable from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from backend.app import create_app
from backend.database.models import db
from backend.database.crud   import create_user, get_user_by_email

SEED_USERS = [
    {
        "email":    "admin@fakenews.dev",
        "username": "admin",
        "password": "admin123!",
        "role":     "admin",
    },
    {
        "email":    "checker@fakenews.dev",
        "username": "factchecker",
        "password": "checker123!",
        "role":     "fact_checker",
    },
    {
        "email":    "publisher@fakenews.dev",
        "username": "publisher",
        "password": "publisher123!",
        "role":     "publisher",
    },
]


def seed():
    app = create_app("development")
    with app.app_context():
        db.create_all()
        created = 0
        skipped = 0

        for u in SEED_USERS:
            if get_user_by_email(u["email"]):
                print(f"  SKIP  {u['email']} (already exists)")
                skipped += 1
            else:
                create_user(u["email"], u["username"], u["password"], u["role"])
                print(f"  OK    {u['email']}  role={u['role']}")
                created += 1

        print(f"\nDone — {created} created, {skipped} skipped.")
        print("\nTest login:")
        print("  curl -X POST http://localhost:5000/api/auth/login \\")
        print('    -H "Content-Type: application/json" \\')
        print('    -d \'{"email":"admin@fakenews.dev","password":"admin123!"}\'')


if __name__ == "__main__":
    seed()