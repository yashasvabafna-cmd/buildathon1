import mysql.connector
import pandas as pd
import json
import os
from datetime import datetime

# --- IMPORTANT: Configure your MySQL connection details here ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',        # Your MySQL username
    'password': '12345678', # Your MySQL password
}
NEW_DB_NAME = 'restaurant_new_db'
# -------------------------------------------------------------
class Item:
    def __init__(self, item_name, quantity, modifiers=None):
        self.item_name = item_name
        self.quantity = quantity
        self.modifiers = modifiers if modifiers is not None else []

def get_mysql_connection(database_name=None):
    """Establishes and returns a connection to the MySQL server or a specific database."""
    config = DB_CONFIG.copy()
    if database_name:
        config['database'] = database_name
    
    try:
        conn = mysql.connector.connect(**config)
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        print("Please ensure your MySQL server is running and connection details are correct.")
        return None

def create_database_if_not_exists(conn, db_name):
    """Creates the specified database if it does not already exist."""
    cursor = conn.cursor()
    try:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        print(f"Database '{db_name}' created or already exists.")
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Error creating database '{db_name}': {err}")
    finally:
        cursor.close()

def drop_all_tables(conn):
    """Drops all tables in the connected database to allow for a clean creation."""
    cursor = conn.cursor()
    try:
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;") # Disable foreign key checks temporarily
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        for table in tables:
            print(f"Dropping table: {table}")
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;") # Re-enable foreign key checks
        print("All existing tables dropped successfully.")
    except mysql.connector.Error as err:
        print(f"Error dropping tables: {err}")
    finally:
        cursor.close()

def create_restaurant_tables(conn):
    """Creates all necessary tables for the restaurant system in MySQL."""
    cursor = conn.cursor()
    try:
        # Create Suppliers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Suppliers (
                Supplier_ID VARCHAR(255) PRIMARY KEY,
                Supplier_Name VARCHAR(255) NOT NULL,
                Contact_Info TEXT,
                Lead_Time INT,
                Payment_Terms VARCHAR(255)
            );
        ''')
        print("Table 'Suppliers' created or already exists.")

        # Create Meals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Meals (
                meal_id INT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                type VARCHAR(255),
                category VARCHAR(255),
                price REAL,
                Chef_chef_id INT
            );
        ''')
        print("Table 'Meals' created or already exists.")

        # Create Ingredients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Ingredients (
                ingredient_id INT PRIMARY KEY AUTO_INCREMENT,
                ingredient_name VARCHAR(255) NOT NULL UNIQUE,
                unit VARCHAR(50),
                current_inventory REAL,
                reorder_point REAL,
                reorder_quantity REAL,
                supplier_id VARCHAR(255),
                FOREIGN KEY(supplier_id) REFERENCES Suppliers(Supplier_ID)
            );
        ''')
        print("Table 'Ingredients' created or already exists.")

        # Updated: Create Recipes table for meal-level recipe details
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Recipes (
                recipe_id INT PRIMARY KEY AUTO_INCREMENT,
                meal_id INT UNIQUE, # Ensures one recipe per meal
                meal_name VARCHAR(255) NOT NULL,
                ingredients JSON, # Stores the raw ingredient list as JSON
                recipe TEXT,      # Stores the preparation instructions
                FOREIGN KEY(meal_id) REFERENCES Meals(meal_id)
            );
        ''')
        print("Table 'Recipes' created or already exists.")

        # New Table: Recipe_Ingredients for linking meals to individual ingredients
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Recipe_Ingredients (
                Recipe_Ingredient_ID INT PRIMARY KEY AUTO_INCREMENT,
                Meal_ID INT,
                Ingredient_ID INT,
                Quantity REAL NOT NULL,
                Recipe_Unit VARCHAR(50) NOT NULL,
                FOREIGN KEY(Meal_ID) REFERENCES Meals(meal_id),
                FOREIGN KEY(Ingredient_ID) REFERENCES Ingredients(ingredient_id)
            );
        ''')
        print("Table 'Recipe_Ingredients' created or already exists.")

        # Create Orders table (from your 'create-orders-table' immersive)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Orders (
            order_id INT PRIMARY KEY AUTO_INCREMENT,
            meal_id INT,
            item_name VARCHAR(255) NOT NULL,
            quantity INT NOT NULL,
            modifiers TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (meal_id) REFERENCES Meals(meal_id)
        );
        """)
        print("Table 'Orders' created or already exists.")

        conn.commit()
        print("All tables created successfully!")
    except mysql.connector.Error as err:
        print(f"An error occurred during table creation: {err}")
    finally:
        cursor.close()

def insert_data_into_tables(conn):
    """
    Inserts data into the tables from external CSV and JSON files.
    """
    cursor = conn.cursor()
    try:
        # --- Insert into Suppliers table (hardcoded for simplicity) ---
        suppliers_data = [
            ('Supplier001', 'City Fresh Produce', '123-456-7890', 2, 'Net 30'),
            ('Supplier002', 'Prime Meats Co.', '987-654-3210', 3, 'Net 15'),
            ('Supplier003', 'Spice Hub Distributors', '555-111-2222', 4, 'Net 45')
        ]
        insert_supplier_query = "INSERT INTO Suppliers (Supplier_ID, Supplier_Name, Contact_Info, Lead_Time, Payment_Terms) VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE Supplier_Name=VALUES(Supplier_Name);"
        cursor.executemany(insert_supplier_query, suppliers_data)
        print("Inserted/Ignored data into Suppliers table.")

        # --- Insert into Meals table from meals.csv ---
        try:
            meals_df = pd.read_csv("meals.csv")
            meals_df.rename(columns={'item_name': 'name'}, inplace=True)
            insert_meal_query = "INSERT INTO Meals (meal_id, name, type, category, price, Chef_chef_id) VALUES (%s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE name=VALUES(name);"
            meal_data = [tuple(row) for row in meals_df[['meal_id', 'name', 'type', 'category', 'price', 'Chef_chef_id']].to_numpy()]
            cursor.executemany(insert_meal_query, meal_data)
            print("Inserted/Ignored data into Meals table from meals.csv.")
        except FileNotFoundError:
            print("Warning: meals.csv not found. Skipping Meals data insertion.")

        # --- Insert into Ingredients table from ingredients_listcsv.csv ---
        ingredient_name_to_id = {} # To store mapping for recipes
        try:
            ingredients_df = pd.read_csv("ingredients_listcsv.csv")
            ingredients_data_tuples = []
            for index, row in ingredients_df.iterrows():
                ingredients_data_tuples.append((
                    row['ingredient_name'],
                    row['unit'],
                    row['current_inventory'],
                    row['reorder_level'], 
                    row['reorder_level'] * 2, # Using reorder_level * 2 as a dummy reorder_quantity
                    row['supplier_id']
                ))

            insert_ingredient_query = """
            INSERT INTO Ingredients (ingredient_name, unit, current_inventory, reorder_point, reorder_quantity, supplier_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                unit=VALUES(unit), current_inventory=VALUES(current_inventory),
                reorder_point=VALUES(reorder_point), reorder_quantity=VALUES(reorder_quantity),
                supplier_id=VALUES(supplier_id);
            """
            cursor.executemany(insert_ingredient_query, ingredients_data_tuples)
            print("Inserted/Ignored data into Ingredients table from ingredients_listcsv.csv.")

            # Fetch ingredient_name to ID mapping for Recipe_Ingredients table
            cursor.execute("SELECT ingredient_name, ingredient_id FROM Ingredients")
            ingredient_name_to_id = {k.lower(): v for k, v in cursor.fetchall()}

        except FileNotFoundError:
            print("Warning: ingredients_listcsv.csv not found. Skipping Ingredients data insertion.")

        # --- Insert into Recipes and Recipe_Ingredients tables from recipes_batch_2.json ---
        try:
            with open('recipes_batch_2.json', 'r') as f:
                recipes_batch_2 = json.load(f)

            recipes_to_insert_main_table = [] # For the 'Recipes' table
            recipes_to_insert_linking_table = [] # For the 'Recipe_Ingredients' table
            
            # Realistic quantities dictionary for a more realistic approach
            realistic_quantities = {
                'salt': (0.01, 'kg'), 'cooking oil': (0.05, 'litres'), 'onion': (0.1, 'kg'), 'tomatoes': (0.15, 'kg'),
                'ginger-garlic paste': (0.02, 'kg'), 'turmeric': (0.005, 'kg'), 'garam masala': (0.005, 'kg'),
                'rice': (0.2, 'kg'), 'paneer': (0.1, 'kg'), 'chicken': (0.2, 'kg'), 'mutton': (0.2, 'kg'),
                'whole wheat flour': (0.1, 'kg'), 'eggplant': (0.2, 'kg'), 'maida': (0.1, 'kg'),
                'potato': (0.15, 'kg'), 'mixed vegetables': (0.2, 'kg'), 'sugar': (0.05, 'kg'),
                'milk': (0.25, 'litres'), 'fish': (0.2, 'kg'), 'prawn': (0.2, 'kg'), 'thick yogurt': (0.1, 'kg'),
                'lemon juice': (0.05, 'litres'), 'bell pepper': (0.1, 'kg'), 'ghee': (0.05, 'kg'),
                'peas': (0.1, 'kg'), 'green chilies': (0.01, 'kg'), 'cauliflower': (0.2, 'kg'),
                'cornflour': (0.05, 'kg'), 'soy sauce': (0.05, 'litres'), 'chili sauce': (0.05, 'litres'),
                'spring onion': (0.02, 'kg'), 'ginger': (0.01, 'kg'), 'garlic': (0.01, 'kg'),
                'curd': (0.1, 'kg'), 'chickpea flour': (0.1, 'kg'), 'whole wheat flour': (0.1, 'kg'),
                'coriander leaves': (0.005, 'kg'), 'jaggery': (0.1, 'kg'), 'semolina': (0.1, 'kg'),
                'black pepper': (0.005, 'kg'), 'cumin seeds': (0.005, 'kg'), 'mustard seeds': (0.005, 'kg'),
                'mint leaves': (0.005, 'kg'), 'fresh cream': (0.1, 'litres'), 'cashew nuts': (0.05, 'kg'),
                'butter': (0.05, 'kg'), 'basmati rice': (0.2, 'kg')
            }

            # Pre-compile common ingredient variations for better mapping
            ingredient_variations = {
                'paneer': 'paneer', 'yogurt': 'thick yogurt', 'ginger-garlic paste': 'ginger-garlic paste',
                'turmeric': 'turmeric', 'red chili powder': 'kashmiri red chili powder', 'garam masala': 'garam masala',
                'chaat masala': 'chaat masala', 'lemon juice': 'lemon juice', 'onion': 'onion',
                'bell pepper': 'bell pepper', 'salt': 'salt', 'oil': 'cooking oil',
                'flour': 'maida', 'ghee': 'ghee', 'potato': 'potato', 'peas': 'peas',
                'green chilies': 'green chilies', 'cauliflower': 'cauliflower', 'cornflour': 'cornflour',
                'soy sauce': 'soy sauce', 'chili sauce': 'chili sauce', 'spring onion': 'spring onion',
                'tomato': 'tomatoes', 'ginger': 'ginger', 'garlic': 'garlic', 'curd': 'curd',
                'chickpea flour': 'chickpea flour', 'whole wheat flour': 'whole wheat flour',
                'fenugreek leaves': 'coriander leaves', 'pickle masala': 'garam masala', 'sattu': 'chickpea flour',
                'eggplant': 'mixed vegetables', 'moong dal': 'peas', 'red lentils': 'peas',
                'coconut milk': 'milk', 'chicken': 'chicken', 'mutton': 'mutton', 'fish': 'fish',
                'prawn': 'prawn', 'rice': 'rice', 'basmati rice': 'basmati rice', 'milk': 'milk',
                'sugar': 'sugar', 'semolina': 'semolina', 'jaggery': 'jaggery', 'black pepper': 'black pepper',
                'cumin seeds': 'cumin seeds', 'mustard seeds': 'mustard seeds', 'mint leaves': 'mint leaves',
                'fresh cream': 'fresh cream', 'cashew nuts': 'cashew nuts', 'butter': 'butter',
                'mixed vegetables': 'mixed vegetables', 'capsicum': 'bell pepper', 'besan': 'chickpea flour',
                'suji': 'semolina', 'coriander': 'coriander leaves', 'methi': 'coriander leaves', # Fenugreek -> Coriander for mapping
                'mutter': 'peas', # Mutter Paneer uses Peas
            }

            unmapped_ingredients_count = 0
            
            for recipe_data in recipes_batch_2:
                meal_id = recipe_data['meal_id']
                meal_name = recipe_data['meal_name']
                ingredients_raw_json = json.dumps(recipe_data['ingredients']) # Store as JSON string
                recipe_instructions = recipe_data['recipe']
                
                # Add to main Recipes table
                recipes_to_insert_main_table.append((meal_id, meal_name, ingredients_raw_json, recipe_instructions))

                ingredients_from_json = recipe_data['ingredients']
                for ingredient_name_raw in ingredients_from_json:
                    ingredient_name_lower = ingredient_name_raw.lower().strip()
                    
                    # Apply lenient mapping for ingredient_name_clean
                    mapped_ingredient_name = None
                    for key, mapped_val in ingredient_variations.items():
                        if key in ingredient_name_lower:
                            mapped_ingredient_name = mapped_val
                            break
                    
                    ingredient_id = None
                    if mapped_ingredient_name:
                        ingredient_id = ingredient_name_to_id.get(mapped_ingredient_name)

                    if ingredient_id:
                        # Use realistic quantities, or a default value if not found
                        quantity, unit = realistic_quantities.get(mapped_ingredient_name, (0.01, 'units'))
                        recipes_to_insert_linking_table.append((meal_id, ingredient_id, quantity, unit))
                    else:
                        unmapped_ingredients_count += 1
                        # Uncomment the next line to debug specific unmapped ingredients
                        # print(f"DEBUG: Could not map ingredient '{ingredient_name_raw}' (cleaned: '{ingredient_name_lower}') for meal ID {meal_id}.")

            if recipes_to_insert_main_table:
                insert_recipe_query = """
                INSERT INTO Recipes (meal_id, meal_name, ingredients, recipe)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    meal_name=VALUES(meal_name), ingredients=VALUES(ingredients), recipe=VALUES(recipe);
                """
                cursor.executemany(insert_recipe_query, recipes_to_insert_main_table)
                print("Inserted/Ignored data into Recipes table from recipes_batch_2.json.")
            else:
                print("No main recipe data processed from recipes_batch_2.json (no meals mapped successfully).")

            if recipes_to_insert_linking_table:
                insert_linking_query = """
                INSERT INTO Recipe_Ingredients (Meal_ID, Ingredient_ID, Quantity, Recipe_Unit)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    Quantity=VALUES(Quantity), Recipe_Unit=VALUES(Recipe_Unit);
                """
                cursor.executemany(insert_linking_query, recipes_to_insert_linking_table)
                print("Inserted/Ignored data into Recipe_Ingredients table from recipes_batch_2.json.")
            else:
                print("No linking recipe data processed from recipes_batch_2.json (no ingredients mapped successfully).")
            
            if unmapped_ingredients_count > 0:
                print(f"Warning: {unmapped_ingredients_count} ingredients from recipes_batch_2.json could not be mapped to existing ingredients. Check console for details.")

        except FileNotFoundError:
            print("Warning: recipes_batch_2.json not found. Skipping Recipes data insertion.")
        except json.JSONDecodeError:
            print("Error: recipes_batch_2.json is not a valid JSON file. Skipping Recipes data insertion.")

        conn.commit()
    except mysql.connector.Error as err:
        print(f"An error occurred during data insertion: {err}")
    finally:
        cursor.close()

# def deplete_inventory_from_order(order_data_items, conn):
#     """
#     Depletes ingredients from the Ingredients table based on the confirmed order items.
    
#     Args:
#         order_data_items (list[Item]): A list of Item objects from the confirmed order.
#         conn (mysql.connector.connection.MySQLConnection): An active MySQL database connection.
#     """
#     if conn is None:
#         print("Error: MySQL connection not established. Cannot deplete inventory.")
#         return False

#     print("\n--- Starting inventory depletion from single order ---")
#     try:
#         with conn.cursor() as cursor:
#             # Step 1: Map meal names to meal_ids from the database
#             meal_name_to_id = {}
#             try:
#                 cursor.execute("SELECT name, meal_id FROM Meals")
#                 meal_name_to_id = {name.lower(): meal_id for name, meal_id in cursor.fetchall()}
#                 print(f"DEBUG: Meal name to ID map: {meal_name_to_id}")
#             except mysql.connector.Error as err:
#                 print(f"Error fetching meal_id mapping: {err}")
#                 return False

#             for order_item in order_data_items:
#                 # --- NEW: Type check for order_item ---
#                 if not isinstance(order_item, Item):
#                     print(f"Error: Expected 'Item' object but received type '{type(order_item).__name__}'. Cannot process order item: {order_item}")
#                     # This error indicates a serious issue in how the function is called.
#                     return False # Return False to indicate failure
#                 # --- END NEW ---

#                 item_name = order_item.item_name
#                 order_quantity = order_item.quantity
#                 meal_id = meal_name_to_id.get(item_name.lower())
#                 print(f"\nDEBUG: Processing order for '{item_name}' (Quantity: {order_quantity})")
#                 print(f"DEBUG: Found Meal ID: {meal_id}")

#                 if meal_id is None:
#                     print(f"Warning: Meal '{item_name}' not found in Meals table. Cannot deplete inventory for this item.")
#                     continue

#                 # Step 2: Get required ingredients and quantities for the ordered meal from Recipe_Ingredients
#                 recipe_query = """
#                 SELECT ri.ingredient_id, ri.quantity, i.ingredient_name, i.current_inventory, i.unit AS recipe_unit
#                 FROM Recipe_Ingredients ri
#                 JOIN Ingredients i ON ri.ingredient_id = i.ingredient_id
#                 WHERE ri.meal_id = %s;
#                 """
#                 cursor.execute(recipe_query, (meal_id,))
#                 ingredients_for_recipe = cursor.fetchall()
#                 print(f"DEBUG: Ingredients for recipe (Meal ID {meal_id}): {ingredients_for_recipe}")


#                 if not ingredients_for_recipe:
#                     print(f"Warning: No recipe ingredients found for meal '{item_name}'. Skipping inventory depletion for this item.")
#                     continue

#                 print(f"Depleting inventory for '{item_name}' (ordered {order_quantity}x):")
#                 for ingredient_id, recipe_quantity_per_meal, ingredient_name, current_inventory, recipe_unit in ingredients_for_recipe:
#                     total_depletion_amount = recipe_quantity_per_meal * order_quantity

#                     print(f"DEBUG:   Ingredient: '{ingredient_name}' (ID: {ingredient_id})")
#                     print(f"DEBUG:     Recipe Qty per meal: {recipe_quantity_per_meal} {recipe_unit}")
#                     print(f"DEBUG:     Current Inventory: {current_inventory} {recipe_unit}")
#                     print(f"DEBUG:     Total Depletion Amount: {total_depletion_amount} {recipe_unit}")

#                     if current_inventory is None:
#                         print(f"  - WARNING: Inventory for '{ingredient_name}' is NULL. Cannot deplete.")
#                         continue
                    
#                     new_inventory = current_inventory - total_depletion_amount

#                     # Step 3: Update the current_inventory in the Ingredients table
#                     update_inventory_query = """
#                     UPDATE Ingredients
#                     SET current_inventory = %s
#                     WHERE ingredient_id = %s;
#                     """
#                     cursor.execute(update_inventory_query, (new_inventory, ingredient_id))
#                     print(f"  - Depleted {total_depletion_amount} {recipe_unit} of '{ingredient_name}'. New inventory: {new_inventory} {recipe_unit}")
            
#             conn.commit()
#             print("Inventory depletion completed successfully.")
#             return True

#     except mysql.connector.Error as err:
#         print(f"An error occurred during inventory depletion: {err}")
#         conn.rollback() # Rollback changes if an error occurs
#         return False
#     except Exception as e:
#         print(f"An unexpected error occurred during inventory depletion: {e}")
#         return False
def main():
    """Main function to establish connection, create DB, create tables, and populate data."""
    # Connect to MySQL server without selecting a specific database first
    conn_server = get_mysql_connection(database_name=None)
    if not conn_server:
        return

    try:
        # Create the new database if it doesn't exist
        create_database_if_not_exists(conn_server, NEW_DB_NAME)
    finally:
        if conn_server:
            conn_server.close()

    # Now connect to the newly created/existing database
    conn_db = get_mysql_connection(database_name=NEW_DB_NAME)
    if not conn_db:
        return

    try:
        # Drop all tables in the specific database for a clean start
        drop_all_tables(conn_db)

        # Create all necessary tables
        create_restaurant_tables(conn_db)

        # Insert all data into the tables
        insert_data_into_tables(conn_db)

        print(f"\nSuccessfully created and populated database '{NEW_DB_NAME}'.")

        # --- Run batch inventory depletion using actual data from 'Orders' table ---
        print("\n--- Running batch inventory depletion using actual data from 'Orders' table ---")
        # success = deplete_inventory_from_order(NEW_DB_NAME,conn_db) # Pass the connection to the DB
        # if success:
        #     print("\nBatch inventory depletion process finished successfully.")
        # else:
        #     print("\nBatch inventory depletion process encountered errors.")

    finally:
        if conn_db:
            conn_db.close()
            print("\nMySQL connection to database closed.")

if __name__ == '__main__':
    # Define a simple Item class for standalone testing
    class Item:
        def __init__(self, item_name, quantity, modifiers=None):
            self.item_name = item_name
            self.quantity = quantity
            self.modifiers = modifiers if modifiers is not None else []

    dummy_order_data = [
        Item(item_name='Paneer Tikka', quantity=1),
        Item(item_name='Butter Chicken', quantity=2),
        Item(item_name='Gulab Jamun', quantity=1) # An item that might not have ingredients
    ]

    # conn = get_mysql_connection()
    # if conn:
    #     try:
    #         success = deplete_inventory_from_order(dummy_order_data, conn)
    #         if success:
    #             print("\nDummy order inventory depletion process finished successfully.")
    #         else:
    #             print("\nDummy order inventory depletion process encountered errors.")
    #     finally:
    #         if conn.is_connected():
    #             conn.close()
    #             print("\nMySQL connection closed.")
    # else:
    #     print("Could not establish database connection for inventory depletion.")
    main()
