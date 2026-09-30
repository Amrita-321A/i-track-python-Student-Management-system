import sqlite3

# Connect to your database (creates file if it doesn't exist)
conn = sqlite3.connect('database.db')

# Create a cursor object
c = conn.cursor()

# Create students table
c.execute('''
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL
)
''')

# Create submissions table
c.execute('''
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    answers TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(id)
)
''')

# Commit changes and close connection
conn.commit()
conn.close()

print("Database and tables created successfully!")
