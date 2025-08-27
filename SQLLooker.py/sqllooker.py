import mysql.connector

# --- IMPORTANT: Configure your MySQL connection details here ---
# Replace with your actual MySQL username, password, and database name.
# These details must match what you used to set up 'restaurant_new_db'
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root', # Your MySQL username
    'password': '12345678', # Your MySQL password
    'database': 'restaurant_new_db' # The name of the database created by the Canvas
}
# -------------------------------------------------------------

def get_mysql_connection():
    """Establishes and returns a connection to the MySQL database using DB_CONFIG."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        print("Please ensure your MySQL server is running and connection details are correct.")
        return None

def display_table_contents(conn, table_name):
    """
    Connects to the database and displays the content of a specified table.
    """
    if conn is None:
        print("No database connection available.")
        return

    cursor = conn.cursor()
    try:
        # Execute the query to select all data from the specified table
        query = f"SELECT * FROM {table_name} ;"
        cursor.execute(query)

        # Fetch all the rows
        rows = cursor.fetchall()

        # Get column names from the cursor description
        column_names = [i[0] for i in cursor.description]

        print(f"\n--- Contents of table: '{table_name}' ---")
        if rows:
            print(" | ".join(column_names)) # Print header
            print("-" * len(" | ".join(column_names))) # Separator
            for row in rows:
                print(" | ".join(str(item) for item in row)) # Print each row
        else:
            print(f"The '{table_name}' table is empty or does not exist.")

    except mysql.connector.Error as err:
        print(f"Error retrieving data from table '{table_name}': {err}")
    finally:
        if cursor:
            cursor.close()

if __name__ == '__main__':
    conn = get_mysql_connection()
    if conn:
        try:
            # Example: Display the 'Orders' table
            display_table_contents(conn, 'Order_Items')

            # You can call this function for other tables too:
            display_table_contents(conn, 'Meals')
            display_table_contents(conn, 'Ingredients')
            display_table_contents(conn, 'Suppliers')
            #display_table_contents(conn, 'Recipes')
            display_table_contents(conn, 'Recipe_Ingredients')
        finally:
            conn.close()
            print("\nMySQL connection is closed.")
    else:
        print("Could not establish a database connection.")
