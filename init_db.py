import psycopg2


create_class_table = """
CREATE TABLE IF NOT EXISTS Class (
    cid SERIAL PRIMARY KEY,
    classname VARCHAR(255) NOT NULL,
    class_index INT UNIQUE NOT NULL
);
"""

create_image_table = """
CREATE TABLE IF NOT EXISTS Image (
    iid SERIAL PRIMARY KEY,
    width INT NOT NULL,
    height INT NOT NULL,
    imagename VARCHAR(255) NOT NULL
);
"""

create_annote_table = """
CREATE TABLE IF NOT EXISTS Annote (
    id SERIAL PRIMARY KEY,
    iid INT NOT NULL,
    class_index INT NOT NULL,
    x1 FLOAT NOT NULL,
    y1 FLOAT NOT NULL,
    x2 FLOAT NOT NULL,
    y2 FLOAT NOT NULL,
    FOREIGN KEY (iid) REFERENCES Image(iid)
);
"""

def init_db(conn):
    try:
        cur = conn.cursor()
        # Create tables
        cur.execute(create_class_table)
        cur.execute(create_image_table)
        cur.execute(create_annote_table)
        print('Tables created successfully.')
        # Commit the changes
        conn.commit()
        cur.close()
        print("Database initialized successfully.")
    except (Exception, psycopg2.Error) as error:
        print("Error while initializing database:", error)
        conn.rollback()
