import os
import sqlite3
import logging

try:
    import psycopg2
    import psycopg2.extras
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

import dbInfo

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CebolaDatabase")

class DatabaseManager:
    def __init__(self):
        self.db_type = None  # 'postgres' or 'sqlite'
        self.sqlite_path = "cebola.db"
        self._detect_and_init_db()

    def _detect_and_init_db(self):
        """
        Detects if PostgreSQL credentials are provided and can connect.
        If not, or if psycopg2 is missing, falls back to SQLite.
        """
        # Check if we should attempt Postgres connection
        use_postgres = False
        if HAS_POSTGRES:
            # Check if any credentials are set in dbInfo
            db_config = dbInfo.DB
            if getattr(db_config, "dbname", "") and getattr(db_config, "ip", ""):
                use_postgres = True

        if use_postgres:
            try:
                logger.info("Attempting to connect to PostgreSQL database...")
                # Test connection
                conn = self._get_postgres_connection()
                conn.close()
                self.db_type = 'postgres'
                logger.info("Successfully connected to PostgreSQL database!")
            except Exception as e:
                logger.warning(f"PostgreSQL connection failed: {e}. Falling back to SQLite.")
                self.db_type = 'sqlite'
        else:
            if not HAS_POSTGRES:
                logger.info("psycopg2 is not installed. Using SQLite.")
            else:
                logger.info("PostgreSQL credentials not configured. Using SQLite.")
            self.db_type = 'sqlite'

        # Initialize tables
        self.initialize_tables()

    def _get_postgres_connection(self):
        """Creates a PostgreSQL connection."""
        db_config = dbInfo.DB
        return psycopg2.connect(
            dbname=db_config.dbname,
            user=db_config.username,
            password=db_config.password,
            host=db_config.ip,
            port=getattr(db_config, "port", "5432"),
        )

    def _get_sqlite_connection(self):
        """Creates a SQLite connection and enables foreign keys and row factory."""
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def get_connection(self):
        """Returns a connection based on the active database type."""
        if self.db_type == 'postgres':
            return self._get_postgres_connection()
        else:
            return self._get_sqlite_connection()

    def execute_query(self, query: str, params: tuple = ()) -> list:
        """
        Executes a SELECT query and returns a list of dictionaries representing the rows.
        Automatically converts parameter placeholders from %s to ? for SQLite.
        """
        if self.db_type == 'sqlite':
            query = query.replace('%s', '?')

        conn = self.get_connection()
        try:
            if self.db_type == 'postgres':
                # Use RealDictCursor to get dictionary results
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cursor.execute(query, params)
                results = [dict(row) for row in cursor.fetchall()]
                cursor.close()
            else:
                cursor = conn.cursor()
                cursor.execute(query, params)
                results = [dict(row) for row in cursor.fetchall()]
                cursor.close()
            return results
        finally:
            conn.close()

    def execute_one(self, query: str, params: tuple = ()) -> dict:
        """
        Executes a SELECT query and returns the first row as a dictionary, or None if no row is found.
        """
        results = self.execute_query(query, params)
        return results[0] if results else None

    def execute_write(self, query: str, params: tuple = ()) -> int:
        """
        Executes an INSERT, UPDATE, or DELETE query and returns the number of affected rows.
        """
        if self.db_type == 'sqlite':
            query = query.replace('%s', '?')

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            affected = cursor.rowcount
            conn.commit()
            cursor.close()
            return affected
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def initialize_tables(self):
        """Initializes tables in the active database if they do not exist."""
        # Standard SQL schema compatible with both PostgreSQL and SQLite
        queries = [
            # 1. Users Table
            """
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(255) PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                name VARCHAR(255) NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at VARCHAR(50) NOT NULL
            );
            """,
            # 2. Shops Table
            """
            CREATE TABLE IF NOT EXISTS shops (
                id VARCHAR(255) PRIMARY KEY,
                owner_id VARCHAR(255) NOT NULL,
                name VARCHAR(255) NOT NULL,
                description TEXT NOT NULL,
                logo_url TEXT,
                is_active BOOLEAN NOT NULL,
                created_at VARCHAR(50) NOT NULL,
                FOREIGN KEY (owner_id) REFERENCES users (id) ON DELETE CASCADE
            );
            """,
            # 3. Products Table
            """
            CREATE TABLE IF NOT EXISTS products (
                id VARCHAR(255) PRIMARY KEY,
                shop_id VARCHAR(255) NOT NULL,
                title VARCHAR(255) NOT NULL,
                description TEXT NOT NULL,
                price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                photo_url TEXT,
                is_available BOOLEAN NOT NULL,
                created_at VARCHAR(50) NOT NULL,
                FOREIGN KEY (shop_id) REFERENCES shops (id) ON DELETE CASCADE
            );
            """,
            # 4. Orders Table
            """
            CREATE TABLE IF NOT EXISTS orders (
                id VARCHAR(255) PRIMARY KEY,
                shop_id VARCHAR(255) NOT NULL,
                user_id VARCHAR(255),
                guest_order_id VARCHAR(255),
                customer_name VARCHAR(255),
                customer_email VARCHAR(255),
                customer_phone VARCHAR(255),
                total_amount REAL NOT NULL,
                total_quantity INTEGER NOT NULL,
                status VARCHAR(50) NOT NULL,
                qr_code_data TEXT,
                created_at VARCHAR(50) NOT NULL,
                FOREIGN KEY (shop_id) REFERENCES shops (id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
            );
            """,
            # 5. Order Items Table
            """
            CREATE TABLE IF NOT EXISTS order_items (
                id VARCHAR(255) PRIMARY KEY,
                order_id VARCHAR(255) NOT NULL,
                product_id VARCHAR(255),
                product_title VARCHAR(255) NOT NULL,
                product_logo TEXT,
                quantity INTEGER NOT NULL,
                price_at_time REAL NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders (id) ON DELETE CASCADE,
                FOREIGN KEY (product_id) REFERENCES products (id) ON DELETE SET NULL
            );
            """,
            # 6. Subscription Plans Table
            """
            CREATE TABLE IF NOT EXISTS subscription_plans (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT NOT NULL,
                price REAL NOT NULL,
                interval VARCHAR(50) NOT NULL,
                max_products INTEGER,
                is_active BOOLEAN NOT NULL,
                created_at VARCHAR(50) NOT NULL
            );
            """,
            # 7. Subscriptions Table
            """
            CREATE TABLE IF NOT EXISTS subscriptions (
                id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                plan_id VARCHAR(255) NOT NULL,
                status VARCHAR(50) NOT NULL,
                started_at VARCHAR(50) NOT NULL,
                expires_at VARCHAR(50),
                cancelled_at VARCHAR(50),
                created_at VARCHAR(50) NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (plan_id) REFERENCES subscription_plans (id) ON DELETE RESTRICT
            );
            """,
            # 8. Shop Ratings Table
            """
            CREATE TABLE IF NOT EXISTS shop_ratings (
                id VARCHAR(255) PRIMARY KEY,
                shop_id VARCHAR(255) NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT,
                created_at VARCHAR(50) NOT NULL,
                FOREIGN KEY (shop_id) REFERENCES shops (id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                UNIQUE(shop_id, user_id)
            );
            """
        ]

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            for query in queries:
                cursor.execute(query)
            conn.commit()
            cursor.close()
            logger.info("Database schemas checked/initialized successfully.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error initializing database tables: {e}")
            raise e
        finally:
            conn.close()

        # Seed data if empty
        try:
            self.seed_subscription_plans()
            self.seed_data()
        except Exception as e:
            logger.error(f"Error seeding data: {e}")

    def seed_subscription_plans(self):
        existing = self.execute_one("SELECT COUNT(*) as count FROM subscription_plans")
        count = existing.get("count", 0) if existing else 0
        if count > 0:
            return

        import datetime
        now = datetime.datetime.utcnow().isoformat() + "Z"
        plans = [
            ("plan-free", "Free", "Up to 10 products, basic shop features.", 0.0, "monthly", 10, True, now),
            ("plan-basic", "Basic", "Up to 50 products and priority support.", 9.99, "monthly", 50, True, now),
            ("plan-pro", "Pro", "Unlimited products and advanced analytics.", 29.99, "monthly", None, True, now),
        ]
        for plan in plans:
            self.execute_write(
                "INSERT INTO subscription_plans (id, name, description, price, interval, max_products, is_active, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                plan
            )
        logger.info("Default subscription plans seeded.")

    def seed_data(self):
        enable_demo = os.environ.get(
            "CEBOLA_ENABLE_DEMO_DATA",
            "true" if self.db_type == "sqlite" else "false"
        ).lower() in ("1", "true", "yes")
        if not enable_demo:
            return

        user_exists = self.execute_one("SELECT COUNT(*) as count FROM users")
        count = user_exists.get("count", 0) if user_exists else 0
        if count > 0:
            return

        logger.info("Database is empty. Seeding demo data...")

        import hashlib
        import base64
        import datetime

        salt = os.urandom(16)
        db_hash = hashlib.pbkdf2_hmac('sha256', b"password123", salt, 100000)
        password_hash = f"{base64.b64encode(salt).decode('utf-8')}:{base64.b64encode(db_hash).decode('utf-8')}"
        now = datetime.datetime.utcnow().isoformat() + "Z"
        self.execute_write(
            "INSERT INTO users (id, email, name, password_hash, created_at) VALUES (%s, %s, %s, %s, %s)",
            ("local-demo-user", "merchant@example.com", "Merchant User", password_hash, now)
        )
        
        # 2. Insert Shop
        self.execute_write(
            "INSERT INTO shops (id, owner_id, name, description, logo_url, is_active, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            ("shop-demo", "local-demo-user", "Mercearia Cebola", "Fresh shelf products, daily reservations, and in-store pickup.", None, True, now)
        )
        
        # 3. Insert Products
        products = [
            ("prod-1", "shop-demo", "Organic onions", "Small batch from a nearby farm, packed in 1 kg bags.", 2.4, 32, None, True, now),
            ("prod-2", "shop-demo", "Goat cheese", "Soft cheese with limited daily availability.", 5.9, 8, None, True, now),
            ("prod-3", "shop-demo", "Fig jam", "Seasonal jar, ideal for pickup bundles.", 4.2, 0, None, False, now)
        ]
        for p in products:
            self.execute_write(
                "INSERT INTO products (id, shop_id, title, description, price, quantity, photo_url, is_available, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                p
            )
            
        # 4. Insert Order
        self.execute_write(
            "INSERT INTO orders (id, shop_id, user_id, guest_order_id, customer_name, customer_email, customer_phone, total_amount, total_quantity, status, qr_code_data, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            ("c7ed1d87-58c1-4f76-8c71-97720f22fd38", "shop-demo", None, "48291375", "Maria Silva", "maria@example.com", "+351 900 000 000", 10.7, 3, "paid", "order:c7ed1d87-58c1-4f76-8c71-97720f22fd38", now)
        )
        
        # 5. Insert Order Items
        order_items = [
            ("item-1", "c7ed1d87-58c1-4f76-8c71-97720f22fd38", "prod-1", "Organic onions", None, 2, 2.4),
            ("item-2", "c7ed1d87-58c1-4f76-8c71-97720f22fd38", "prod-2", "Goat cheese", None, 1, 5.9)
        ]
        for oi in order_items:
            self.execute_write(
                "INSERT INTO order_items (id, order_id, product_id, product_title, product_logo, quantity, price_at_time) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                oi
            )
        logger.info("Demo data seeded successfully.")

# Export a single global instance
db = DatabaseManager()
