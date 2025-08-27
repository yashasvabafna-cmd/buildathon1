import mysql.connector
import pandas as pd
import json
import os
import re
from datetime import datetime
import time 

# --- IMPORTANT: Configure your MySQL connection details here ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',        # Your MySQL username
    'password': '12345678', # Your MySQL password
}
NEW_DB_NAME = 'restaurant_new_db'
# -------------------------------------------------------------

# Unit conversion factors to a standard SI basis (grams and milliliters)
CONVERSION_FACTORS = {
    'kg': 1000,
    'L': 1000,
    'g': 1,
    'ml': 1,
    'piece': 25, # Assumption: average weight for a piece
    'cup': 240,  # Assumption: average weight/volume for a cup
    'tbsp': 15,  # Assumption: 1 tablespoon = 15 ml/g
    'tsp': 5,    # Assumption: 1 teaspoon = 5 ml/g
    'bunch': 100, # Assumption: average weight for a bunch
}

# Mapping of messy/repetitive names to a single, clean name
CLEAN_NAME_MAP = {
    'paneer': 'Paneer',
    'g paneer': 'Paneer',
    'cup paneer': 'Paneer',
    'grated paneer': 'Paneer',
    'g boneless chicken breast': 'boneless chicken',
    'g boneless chicken': 'boneless chicken',
    'g minced mutton or chicken': 'minced meat',
    'g minced mutton': 'minced meat',
    'g chicken or mutton': 'minced meat',
    'g minced meat': 'minced meat',
    'g firm white fish fillets': 'firm white fish',
    'g firm fish': 'firm white fish',
    'g prawns': 'Prawns',
    'g okra': 'Okra',
    'g chicken tikka pieces': 'chicken tikka pieces',
    'g mutton': 'Mutton',
    'g chicken': 'Chicken',
    'ginger-garlic paste': 'ginger-garlic paste',
    'Salt to taste': 'Salt',
    'Cooking Oil': 'Cooking Oil',
    'Oil for deep frying': 'Cooking Oil',
    'Oil for shallow frying': 'Cooking Oil',
    'oil for pan': 'Cooking Oil',
    'tbsp vegetable oil for grilling': 'Cooking Oil',
    'tbsp oil': 'Cooking Oil',
    'Ghee for cooking': 'Ghee',
    'tbsp ghee': 'Ghee',
    'Ghee for shallow frying': 'Ghee',
    'Melted butter for basting': 'Butter',
    'tbsp butter': 'Butter',
    'Melted butter for brushing': 'Butter',
    'tsp turmeric powder': 'turmeric',
    '1/2 tsp turmeric': 'turmeric',
    'tsp Kashmiri red chili powder': 'Kashmiri red chili powder',
    'tbsp red chili powder': 'red chili powder',
    'tsp red chili powder': 'red chili powder',
    '1/2 tsp red chili powder': 'red chili powder',
    'tsp garam masala': 'garam masala',
    '1/2 tsp garam masala': 'garam masala',
    'tsp chaat masala': 'chaat masala',
    '1/2 tsp chaat masala': 'chaat masala',
    'tbsp lemon juice': 'lemon juice',
    'Juice of': 'lemon juice',
    'large onion': 'onion',
    'large onions': 'onion',
    'medium onion': 'onion',
    'small onion': 'onion',
    'large bell pepper': 'bell pepper',
    '1/2 bell pepper': 'bell pepper',
    'Salt and pepper to taste': 'Salt',
    'Water as needed': 'Water',
    'cups hot water': 'Water',
    'Warm water to knead': 'Water',
    'cups water or vegetable stock': 'Water',
    'all-purpose flour': 'all-purpose flour',
    '1/4 cup all-purpose flour': 'all-purpose flour',
    'cups all': 'all-purpose flour',
    'gram flour': 'gram flour',
    'cup gram flour': 'gram flour',
    '1/2 cup gram flour (besan)': 'gram flour',
    '1/4 cup roasted gram flour (besan)': 'gram flour',
    'whole wheat flour': 'whole wheat flour',
    'cups whole wheat flour': 'whole wheat flour',
    'cup whole wheat flour': 'whole wheat flour',
    'ghee': 'Ghee',
    'butter': 'Butter',
    'Sugar': 'Sugar',
    'tsp sugar': 'Sugar',
    '1/2 cup yogurt': 'yogurt',
    'yogurt': 'yogurt',
    'thick yogurt': 'yogurt',
    'cup thick yogurt': 'yogurt',
    'curd': 'yogurt',
    'cups hung curd': 'yogurt',
    '1/4 cup yogurt': 'yogurt',
    'cream': 'cream',
    '1/4 cup heavy cream': 'cream',
    'tbsp cream': 'cream',
    'Ginger': 'ginger',
    'inch ginger': 'ginger',
    'tsp ginger': 'ginger',
    'Ginger garlic paste': 'ginger-garlic paste',
    'garlic': 'garlic',
    'cloves garlic': 'garlic',
    'garlic paste': 'garlic',
    'Tomatoes': 'Tomatoes',
    'large tomatoes': 'Tomatoes',
    'Tomato puree': 'Tomato puree',
    'medium potatoes': 'potatoes',
    'large potatoes': 'potatoes',
    'peppers': 'bell pepper',
    'spring onion': 'spring onion',
    'coriander leaves': 'coriander',
    'tbsp chopped coriander leaves': 'coriander',
    '1/4 cup chopped coriander': 'coriander',
    'tbsp chopped coriander': 'coriander',
    'cornflour': 'cornflour',
    '1/4 cup cornflour': 'cornflour',
    'tbsp cornflour': 'cornflour',
    'red chili powder': 'red chili powder',
    'mustard oil': 'mustard oil',
    'tbsp mustard oil': 'mustard oil',
    'paneer cubes': 'Paneer',
    'chicken tikka pieces': 'chicken tikka pieces',
    'cooked rice': 'basmati rice',
    'basmati rice': 'basmati rice',
    'cups basmati rice': 'basmati rice',
    'Milk to knead': 'milk',
    '1/2 cup warm milk': 'milk',
    'litre full': 'milk'
}

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
        print(f"Error creating database: {err}")
    finally:
        cursor.close()

def drop_all_tables():
    """Drops all tables in the database to ensure a clean start."""
    conn = None
    try:
        conn = get_mysql_connection(database_name=NEW_DB_NAME)
        if not conn:
            print("Failed to get connection for dropping tables.")
            return

        cursor = conn.cursor()
        # Disable foreign key checks temporarily
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
        
        # Get a list of all tables in the current database
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        if tables:
            print("\n--- Dropping existing tables ---")
            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
            print("All tables dropped.")
            conn.commit()
        else:
            print("No existing tables to drop.")
            
        # Re-enable foreign key checks
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
    except mysql.connector.Error as err:
        print(f"Error dropping tables: {err}")
        if conn: conn.rollback()
    finally:
        if conn and conn.is_connected():
            conn.close()

def create_restaurant_tables():
    """Creates the necessary tables for the restaurant database."""
    conn = None
    try:
        conn = get_mysql_connection(database_name=NEW_DB_NAME)
        if not conn:
            print("Failed to get connection for creating tables.")
            return

        cursor = conn.cursor()
        
        # SQL to create the tables
        tables = {
            'Suppliers': """
                CREATE TABLE IF NOT EXISTS Suppliers (
                    supplier_id VARCHAR(255) PRIMARY KEY,
                    supplier_name VARCHAR(255),
                    contact_info VARCHAR(255)
                );
            """,
            'Ingredients': """
                CREATE TABLE IF NOT EXISTS Ingredients (
                    ingredient_id INT AUTO_INCREMENT PRIMARY KEY,
                    ingredient_name VARCHAR(255) UNIQUE NOT NULL,
                    unit VARCHAR(50),
                    current_inventory DOUBLE,
                    reorder_point DOUBLE,
                    reorder_quantity DOUBLE,
                    supplier_id VARCHAR(255),
                    FOREIGN KEY (supplier_id) REFERENCES Suppliers(supplier_id)
                );
            """,
            'Meals': """
                CREATE TABLE IF NOT EXISTS Meals (
                    meal_id INT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    type VARCHAR(255),
                    category VARCHAR(255),
                    price REAL,
                    Chef_chef_id INT,
                    available BOOLEAN DEFAULT TRUE
                );
            """,
            'Recipe_Ingredients': """
                CREATE TABLE IF NOT EXISTS Recipe_Ingredients (
                    recipe_ingredient_id INT AUTO_INCREMENT PRIMARY KEY,
                    Meal_ID INT,
                    Ingredient_ID INT,
                    Quantity DOUBLE,
                    FOREIGN KEY (Meal_ID) REFERENCES Meals(meal_id),
                    FOREIGN KEY (Ingredient_ID) REFERENCES Ingredients(ingredient_id),
                    UNIQUE (Meal_ID, Ingredient_ID)
                );
            """,
            'Purchase_Orders': """
                CREATE TABLE IF NOT EXISTS Purchase_Orders (
                    po_id INT AUTO_INCREMENT PRIMARY KEY,
                    ingredient_id INT,
                    ingredient_name VARCHAR(255),
                    ordered_quantity DOUBLE,
                    status VARCHAR(50) DEFAULT 'Placed',
                    order_placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    order_delivered_at TIMESTAMP NULL,
                    FOREIGN KEY (ingredient_id) REFERENCES Ingredients(ingredient_id)
                );
            """,
            'Order_Items': """
                CREATE TABLE IF NOT EXISTS Order_Items (
                    order_item_id INT AUTO_INCREMENT PRIMARY KEY,
                    order_id VARCHAR(255) NOT NULL,
                    meal_id INT,
                    quantity INT,
                    FOREIGN KEY (meal_id) REFERENCES Meals(meal_id)
                );
            """
        }
        
        print("\n--- Creating Tables ---")
        for table_name, table_sql in tables.items():
            print(f"Creating table '{table_name}'...")
            cursor.execute(table_sql)
            print(f"Table '{table_name}' created.")
        conn.commit()
        print("All tables created successfully!")
    except mysql.connector.Error as err:
        print(f"Error creating tables: {err}")
        if conn: conn.rollback()
    finally:
        if conn and conn.is_connected():
            conn.close()

def insert_data_into_tables():
    """Populates the database tables with data from CSV and JSON files."""
    conn = None
    try:
        conn = get_mysql_connection(database_name=NEW_DB_NAME)
        if not conn:
            print("Failed to get connection for inserting data.")
            return

        cursor = conn.cursor()
        # --- Insert Suppliers (Dummy Data) ---
        suppliers_data = [
            ('Supplier001', 'Green Groceries Inc.', 'contact1@example.com'),
            ('Supplier002', 'Spice Route Co.', 'contact2@example.com'),
            ('Supplier003', 'Dairy Farms Ltd.', 'contact3@example.com')
        ]
        supplier_insert_query = "INSERT INTO Suppliers (supplier_id, supplier_name, contact_info) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE supplier_name=VALUES(supplier_name);"
        cursor.executemany(supplier_insert_query, suppliers_data)
        
        # --- Insert Meals from meals.csv ---
        print("\n--- Inserting Meals from CSV ---")
        try:
            meals_df = pd.read_csv("sqldatafiles/meals.csv")
            insert_meal_query = """
                INSERT INTO Meals (meal_id, name, type, category, price, Chef_chef_id, available)
                VALUES (%s, %s, %s, %s, %s, %s, TRUE)
                ON DUPLICATE KEY UPDATE name=VALUES(name);
            """
            meal_data = [tuple(row) for row in meals_df.to_numpy()]
            cursor.executemany(insert_meal_query, meal_data)
            print(f"  - {cursor.rowcount} meals from CSV inserted/updated.")
        except FileNotFoundError:
            print("Warning: sqldatafiles/meals.csv not found. Skipping Meals data insertion.")

        # --- Insert Ingredients from ingredients_listcsv.csv ---
        print("\n--- Inserting Ingredients from CSV ---")
        ingredient_name_to_id = {}
        try:
            df_ingredients = pd.read_csv('sqldatafiles/ingredients_listcsv.csv')
            
            ingredient_insert_query = """
                INSERT INTO Ingredients (ingredient_name, unit, current_inventory, reorder_point, reorder_quantity, supplier_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    unit = VALUES(unit),
                    current_inventory = VALUES(current_inventory),
                    reorder_point = VALUES(reorder_point),
                    reorder_quantity = VALUES(reorder_quantity),
                    supplier_id = VALUES(supplier_id);
            """
            ingredients_to_insert = [
                (row['ingredient_name'], row['unit'], row['current_inventory'], row['reorder_level'], 
                 row['reorder_quantity'] if 'reorder_quantity' in row else 25.0, row['supplier_id'])
                for index, row in df_ingredients.iterrows()
            ]
            
            cursor.executemany(ingredient_insert_query, ingredients_to_insert)
            print(f"  - {cursor.rowcount} ingredients from CSV inserted/updated.")
            
            cursor.execute("SELECT ingredient_id, ingredient_name FROM Ingredients")
            ingredient_name_to_id = {name.lower(): id for id, name in cursor.fetchall()}
        except FileNotFoundError:
            print("Warning: sqldatafiles/ingredients_listcsv.csv not found. Skipping Ingredients data insertion.")

        # --- Insert Meals and Recipe Ingredients from JSON ---
        print("\n--- Inserting Recipes from JSON ---")
        try:
            with open('sqldatafiles/recipes_batch_2.json', 'r') as f:
                recipes_data = json.load(f)

            unmapped_count = 0
            for recipe in recipes_data:
                # Get the meal_id for the current recipe from the Meals table
                cursor.execute("SELECT meal_id FROM Meals WHERE name = %s", (recipe['meal_name'],))
                meal_id = cursor.fetchone()
                if not meal_id:
                    print(f"  - Skipping recipe for '{recipe['meal_name']}' as meal was not found in meals.csv.")
                    continue
                meal_id = meal_id[0]

                # Now, handle Recipe_Ingredients
                for ingredient_str in recipe['ingredients']:
                    # Improved parsing logic
                    match = re.search(r'^\s*([\d.]+)?\s*([a-zA-Z\s,]+)', ingredient_str)
                    
                    if match:
                        quantity_str = match.group(1)
                        quantity = float(quantity_str) if quantity_str else 0.0
                        ingredient_name_raw = match.group(2).strip()
                        ingredient_name = ingredient_name_raw.split(',')[0].strip()
                    else:
                        quantity = 0.0
                        ingredient_name = ingredient_str.split(',')[0].strip()

                    ing_id = ingredient_name_to_id.get(ingredient_name.lower())

                    if not ing_id:
                        unmapped_count += 1
                        insert_unmapped_query = """
                            INSERT INTO Ingredients (ingredient_name, unit, current_inventory, reorder_point, reorder_quantity, supplier_id)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE ingredient_name = ingredient_name;
                        """
                        cursor.execute(insert_unmapped_query, (ingredient_name, 'piece', 0.0, 1.0, 25.0, 'Supplier001'))
                        ing_id = cursor.lastrowid
                        ingredient_name_to_id[ingredient_name.lower()] = ing_id

                    # Ensure quantity is capped at 20.0
                    capped_quantity = min(quantity, 20.0)

                    recipe_ing_insert_query = """
                        INSERT INTO Recipe_Ingredients (Meal_ID, Ingredient_ID, Quantity)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE Quantity = VALUES(Quantity);
                    """
                    cursor.execute(recipe_ing_insert_query, (meal_id, ing_id, capped_quantity))

            print(f"  - {unmapped_count} ingredients were newly added to the main inventory.")
        except FileNotFoundError:
            print("Warning: sqldatafiles/recipes_batch_2.json not found. Skipping recipe data insertion.")

        conn.commit()
        print("All data inserted successfully!")
    except mysql.connector.Error as err:
        print(f"Error inserting data: {err}")
        if conn: conn.rollback()
    finally:
        if conn and conn.is_connected():
            conn.close()

def clean_and_standardize_ingredients():
    """
    Standardizes ingredient units and consolidates duplicate ingredients into a single entry.
    """
    conn = None
    try:
        conn = get_mysql_connection(database_name=NEW_DB_NAME)
        if not conn:
            print("Failed to get connection for cleaning ingredients.")
            return

        cursor = conn.cursor(dictionary=True)
        print("\n--- Cleaning and Standardizing Ingredients ---")

        cursor.execute("SELECT ingredient_id, ingredient_name, unit, current_inventory, reorder_point, reorder_quantity FROM Ingredients")
        ingredients = cursor.fetchall()
        
        # A map to hold consolidated ingredient data: clean_name -> {id, inventory, ...}
        consolidated_ingredients = {}

        for ing in ingredients:
            original_name = ing['ingredient_name']
            clean_name = CLEAN_NAME_MAP.get(original_name.lower(), original_name)
            
            # Get the conversion factor, defaulting to 1 if not found
            conversion_factor = CONVERSION_FACTORS.get(ing['unit'].lower(), 1)
            
            # Convert quantities to a standard unit (grams or ml)
            ing['current_inventory'] *= conversion_factor
            ing['reorder_point'] *= conversion_factor
            ing['reorder_quantity'] *= conversion_factor
            ing['unit'] = 'g' if ing['unit'] in ['kg', 'g', 'piece', 'cup', 'tbsp', 'tsp', 'bunch'] else 'ml'

            if clean_name not in consolidated_ingredients:
                consolidated_ingredients[clean_name] = {
                    'primary_id': ing['ingredient_id'],
                    'inventory': ing['current_inventory'],
                    'reorder_point': ing['reorder_point'],
                    'reorder_quantity': ing['reorder_quantity'],
                    'unit': ing['unit']
                }
            else:
                # Add to existing consolidated entry
                consolidated_ingredients[clean_name]['inventory'] += ing['current_inventory']
                # Take the max reorder point/quantity to be safe
                consolidated_ingredients[clean_name]['reorder_point'] = max(consolidated_ingredients[clean_name]['reorder_point'], ing['reorder_point'])
                consolidated_ingredients[clean_name]['reorder_quantity'] = max(consolidated_ingredients[clean_name]['reorder_quantity'], ing['reorder_quantity'])

        # Update the main ingredient table and delete duplicates
        ingredients_to_delete = []
        for ing in ingredients:
            original_name = ing['ingredient_name']
            clean_name = CLEAN_NAME_MAP.get(original_name.lower(), original_name)
            
            if clean_name != original_name or ing['unit'] not in ['g', 'ml']:
                # This is a duplicate or needs updating
                if ing['ingredient_id'] != consolidated_ingredients[clean_name]['primary_id']:
                    ingredients_to_delete.append(ing['ingredient_id'])
                    # Update Recipe_Ingredients to point to the new primary ID
                    cursor.execute("UPDATE Recipe_Ingredients SET Ingredient_ID = %s WHERE Ingredient_ID = %s", (consolidated_ingredients[clean_name]['primary_id'], ing['ingredient_id']))

        # Delete old duplicates
        if ingredients_to_delete:
            delete_query = "DELETE FROM Ingredients WHERE ingredient_id IN ({})".format(','.join(['%s'] * len(ingredients_to_delete)))
            cursor.execute(delete_query, ingredients_to_delete)
        
        # Update the primary ingredient entries with consolidated values
        for clean_name, data in consolidated_ingredients.items():
            update_query = """
                UPDATE Ingredients
                SET ingredient_name = %s, unit = %s, current_inventory = %s, reorder_point = %s, reorder_quantity = %s
                WHERE ingredient_id = %s
            """
            cursor.execute(update_query, (
                clean_name,
                data['unit'],
                data['inventory'],
                data['reorder_point'],
                data['reorder_quantity'],
                data['primary_id']
            ))

        conn.commit()
        print("  - Ingredients table has been standardized and duplicates removed.")

    except mysql.connector.Error as err:
        print(f"Error during ingredient cleaning: {err}")
        if conn: conn.rollback()
    finally:
        if conn and conn.is_connected():
            conn.close()

def set_reorder_point_from_recipes():
    """
    Sets the reorder point for each ingredient based on the maximum quantity required
    by any recipe that uses it.
    """
    conn = None
    try:
        conn = get_mysql_connection(database_name=NEW_DB_NAME)
        if not conn:
            print("Failed to get connection for setting reorder points from recipes.")
            return

        print("\n--- Setting Reorder Points Based on Recipes ---")
        cursor = conn.cursor()

        # Find the maximum quantity required for each ingredient
        max_qty_query = """
            SELECT Ingredient_ID, MAX(Quantity)
            FROM Recipe_Ingredients
            GROUP BY Ingredient_ID;
        """
        cursor.execute(max_qty_query)
        max_quantities = cursor.fetchall()

        if max_quantities:
            for ing_id, max_qty in max_quantities:
                # Update the reorder_point to be the max quantity required, ensuring it's not greater than the existing reorder point
                update_query = """
                    UPDATE Ingredients
                    SET reorder_point = GREATEST(%s, reorder_point)
                    WHERE ingredient_id = %s;
                """
                cursor.execute(update_query, (max_qty, ing_id))
            conn.commit()
            print("  - Reorder points updated to be at least the maximum quantity required by any recipe.")
    except mysql.connector.Error as err:
        print(f"Error setting reorder points: {err}")
        if conn: conn.rollback()
    finally:
        if conn and conn.is_connected():
            conn.close()

def update_meal_availability():
    """
    Updates the 'available' status of meals based on ingredient inventory.
    This function now manages its own database connection.
    """
    conn = None
    try:
        conn = get_mysql_connection(database_name=NEW_DB_NAME)
        if not conn:
            print("Error: Could not establish database connection for updating meal availability.")
            return

        print("\n--- Updating Meal Availability ---")
        cursor = conn.cursor()
        
        # First, assume all meals are available
        cursor.execute("UPDATE Meals SET available = TRUE")
        conn.commit()

        # Find meals that are NOT available due to insufficient ingredients
        unavailable_meals_query = """
            SELECT DISTINCT m.meal_id, m.name
            FROM Meals m
            JOIN Recipe_Ingredients ri ON m.meal_id = ri.Meal_ID
            JOIN Ingredients i ON ri.ingredient_id = i.ingredient_id
            WHERE i.current_inventory < ri.Quantity;
        """
        cursor.execute(unavailable_meals_query)
        unavailable_meals = cursor.fetchall()

        if unavailable_meals:
            print("The following meals are marked as UNAVAILABLE due to insufficient ingredients:")
            for meal_id, meal_name in unavailable_meals:
                print(f"  - {meal_name} (Meal ID: {meal_id})")
                # Set AVAILABLE to FALSE for these meals
                update_query = "UPDATE Meals SET available = FALSE WHERE meal_id = %s"
                cursor.execute(update_query, (meal_id,))
            conn.commit()
        else:
            print("All meals have sufficient ingredients and are marked as AVAILABLE.")

    except mysql.connector.Error as err:
        print(f"Error updating meal availability: {err}")
        if conn: conn.rollback()
    except Exception as e:
        print(f"An unexpected error occurred during update_meal_availability: {e}")
        if conn: conn.rollback()
    finally:
        if conn and conn.is_connected():
            conn.close()

def set_initial_inventory():
    """
    Sets a reasonable starting inventory value for ingredients if their initial value is zero
    or below their reorder point, ensuring they are initially well-stocked.
    This function now manages its own database connection.
    """
    conn = None
    try:
        conn = get_mysql_connection(database_name=NEW_DB_NAME)
        if not conn:
            print("Failed to get connection for setting initial inventory.")
            return

        print("\n--- Setting Initial Inventory ---")
        cursor = conn.cursor()

        # Update ingredients that are at or below their reorder point or have zero/null inventory
        update_query = """
            UPDATE Ingredients
            SET current_inventory = reorder_point + reorder_quantity -- Bring it above reorder point
            WHERE current_inventory <= reorder_point OR current_inventory IS NULL OR current_inventory <= 0.0;
        """
        cursor.execute(update_query)
        conn.commit()
        print(f"  - Updated {cursor.rowcount} ingredient inventories to ensure sufficient stock initially.")
    except Exception as err:
        print(f"Error setting initial inventory: {err}")
        if conn: conn.rollback()
    finally:
        if conn and conn.is_connected():
            conn.close()

def find_missing_ingredients_for_meal(meal_name):
    """
    Finds and returns a list of ingredients for a given meal that are below the required quantity.
    """
    conn = None
    try:
        conn = get_mysql_connection(database_name=NEW_DB_NAME)
        if not conn:
            print(f"  - Could not establish connection to check ingredients for '{meal_name}'.")
            return []

        cursor = conn.cursor()
        query = """
            SELECT i.ingredient_name, i.current_inventory, ri.Quantity
            FROM Meals m
            JOIN Recipe_Ingredients ri ON m.meal_id = ri.Meal_ID
            JOIN Ingredients i ON ri.ingredient_id = i.ingredient_id
            WHERE m.name = %s AND i.current_inventory < ri.Quantity;
        """
        cursor.execute(query, (meal_name,))
        return cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Error checking ingredients for {meal_name}: {err}")
        return []
    finally:
        if conn and conn.is_connected():
            conn.close()

def debug_availability_issue():
    """
    Prints a detailed debug report for meals that are marked as unavailable.
    """
    conn = None
    try:
        conn = get_mysql_connection(database_name=NEW_DB_NAME)
        if not conn:
            print("Error: Could not connect for debugging.")
            return

        print("\n--- Debugging Meal Availability Issues ---")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM Meals WHERE available = FALSE;")
        unavailable_meals = [row[0] for row in cursor.fetchall()]

        if unavailable_meals:
            print("The following meals are still marked as unavailable. Here's why:")
            for meal_name in unavailable_meals:
                missing_ingredients = find_missing_ingredients_for_meal(meal_name)
                if missing_ingredients:
                    print(f"  - '{meal_name}' is unavailable because:")
                    for ing_name, current_inv, required_qty in missing_ingredients:
                        print(f"    - '{ing_name}' has {current_inv} but requires {required_qty}.")
        else:
            print("All meals appear to have sufficient ingredients.")
    except Exception as e:
        print(f"An error occurred during debugging: {e}")
    finally:
        if conn and conn.is_connected():
            conn.close()

def check_and_order_ingredients():
    """
    Checks for ingredients that have fallen below their reorder point and simulates
    a purchase order to replenish stock.
    This function now manages its own database connection.
    """
    conn = None
    try:
        conn = get_mysql_connection(database_name=NEW_DB_NAME)
        if not conn:
            print("Failed to get connection for checking and ordering ingredients.")
            return

        cursor = conn.cursor()
        print("\n--- Checking and Ordering Ingredients ---")

        # Find ingredients below reorder point
        reorder_query = """
            SELECT ingredient_id, ingredient_name, reorder_quantity, supplier_id, current_inventory
            FROM Ingredients
            WHERE current_inventory < reorder_point AND reorder_point > 0;
        """
        cursor.execute(reorder_query)
        to_order = cursor.fetchall()

        if to_order:
            for ing_id, ing_name, qty, supp_id, current_inv in to_order:
                print(f"  - Placing order for {qty} of '{ing_name}' from supplier '{supp_id}'.")
                
                # Insert a new purchase order
                po_insert_query = """
                    INSERT INTO Purchase_Orders (ingredient_id, ingredient_name, ordered_quantity, status)
                    VALUES (%s, %s, %s, 'Placed');
                """
                cursor.execute(po_insert_query, (ing_id, ing_name, qty))
                po_id = cursor.lastrowid
                conn.commit()

                # Simulate delivery time
                print("  - Simulating a 10-second delivery delay...")
                time.sleep(10) # Simulating a delay of 10 seconds
                
                # Update inventory and order status to 'Delivered'
                update_status_query = "UPDATE Purchase_Orders SET status = 'Delivered', order_delivered_at = NOW() WHERE po_id = %s;"
                cursor.execute(update_status_query, (po_id,))
                
                update_inventory_query = "UPDATE Ingredients SET current_inventory = current_inventory + %s WHERE ingredient_id = %s;"
                cursor.execute(update_inventory_query, (qty, ing_id))
                
                conn.commit()
        else:
            print("  - No ingredients need to be reordered.")
    except mysql.connector.Error as err:
        print(f"Error checking/ordering ingredients: {err}")
        if conn: conn.rollback()
    except Exception as e:
        print(f"An unexpected error occurred during check_and_order_ingredients: {e}")
        if conn: conn.rollback()
    finally:
        if conn and conn.is_connected():
            conn.close()

def deplete_inventory_from_order(order_data_items):
    """
    Simulates a customer order and depletes inventory for the meals ordered.
    This function now manages its own database connection.
    """
    conn = None
    try:
        conn = get_mysql_connection(database_name=NEW_DB_NAME)
        if not conn:
            print("Error: MySQL connection not established. Cannot deplete inventory.")
            return False

        print("\n--- Depleting Inventory for Customer Order ---")
        cursor = conn.cursor()
        
        # A dictionary to sum up total ingredient usage for this order
        ingredient_usage = {}
        for item in order_data_items:
            # Get meal ID
            cursor.execute("SELECT meal_id FROM Meals WHERE name = %s", (item.item_name,))
            meal_id = cursor.fetchone()
            if meal_id:
                meal_id = meal_id[0]
                # Get all ingredients for this meal
                cursor.execute("SELECT Ingredient_ID, Quantity FROM Recipe_Ingredients WHERE Meal_ID = %s", (meal_id,))
                recipe_ingredients = cursor.fetchall()
                for ing_id, quantity in recipe_ingredients:
                    if ing_id in ingredient_usage:
                        ingredient_usage[ing_id] += quantity * item.quantity
                    else:
                        ingredient_usage[ing_id] = quantity * item.quantity
        
        if not ingredient_usage:
            print("  - No meals were found in the order to deplete inventory.")
            return True # No depletion, but not an error

        # Update inventory based on total usage
        for ing_id, used_qty in ingredient_usage.items():
            update_query = "UPDATE Ingredients SET current_inventory = GREATEST(0, current_inventory - %s) WHERE ingredient_id = %s;"
            cursor.execute(update_query, (used_qty, ing_id))
        
        conn.commit()
        print("  - Inventory successfully depleted for the order.")
        return True

    except mysql.connector.Error as err:
        print(f"Error depleting inventory: {err}")
        if conn: conn.rollback()
        return False
    except Exception as e:
        print(f"An unexpected error occurred during deplete_inventory_from_order: {e}")
        if conn: conn.rollback()
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()

def verify_purchase_orders():
    """
    Verifies that purchase orders were created and shows the most recent ones.
    This function now manages its own database connection.
    """
    conn = None
    try:
        conn = get_mysql_connection(database_name=NEW_DB_NAME)
        if not conn:
            print("Failed to get connection for verifying purchase orders.")
            return

        cursor = conn.cursor()
        print("\n--- Verifying Purchase Orders ---")
        cursor.execute("SELECT po_id, ingredient_name, ordered_quantity, status, order_placed_at, order_delivered_at FROM Purchase_Orders ORDER BY order_placed_at DESC LIMIT 5")
        po_results = cursor.fetchall()
        if po_results:
            print("Recent Purchase Orders:")
            for po_id, ing_name, qty, status, placed_at, delivered_at in po_results:
                print(f"  PO ID: {po_id}, Ingredient: {ing_name}, Quantity: {qty}, Status: {status}, Placed: {placed_at}, Delivered: {delivered_at}")
        else:
            print("  - No purchase orders found.")
    except mysql.connector.Error as err:
        print(f"Error during Purchase Order verification: {err}")
        if conn: conn.rollback()
    finally:
        if conn and conn.is_connected():
            conn.close()

def fetch_order_data_from_db():
    """Fetches the latest order data from the Order_Items table and returns as a list of Item objects."""
    conn = None
    try:
        conn = get_mysql_connection(database_name=NEW_DB_NAME)
        if not conn:
            print("Error: Could not connect to fetch order data from DB.")
            return []

        cursor = conn.cursor()
        print("\n--- Fetching Latest Order from Database ---")
        
        # Find the most recent order_id
        cursor.execute("SELECT DISTINCT order_id FROM Order_Items ORDER BY order_item_id DESC LIMIT 1")
        latest_order_id = cursor.fetchone()
        
        if not latest_order_id:
            print("  - No orders found in the database.")
            return []

        latest_order_id = latest_order_id[0]
        
        # Fetch all items for that order_id
        order_query = """
            SELECT m.name, oi.quantity
            FROM Order_Items oi
            JOIN Meals m ON oi.meal_id = m.meal_id
            WHERE oi.order_id = %s;
        """
        cursor.execute(order_query, (latest_order_id,))
        order_items_data = cursor.fetchall()
        
        # Convert the fetched data into a list of Item objects
        customer_order_items = [Item(meal_name, quantity) for meal_name, quantity in order_items_data]
        
        print(f"  - Fetched {len(customer_order_items)} items for Order ID: {latest_order_id}")
        return customer_order_items

    except mysql.connector.Error as err:
        print(f"Error fetching order data from DB: {err}")
        return []
    finally:
        if conn and conn.is_connected():
            conn.close()

def main():
    """Main function to establish connection, create DB, create tables, and populate data."""
    conn_server = get_mysql_connection(database_name=None)
    if not conn_server:
        return

    try:
        create_database_if_not_exists(conn_server, NEW_DB_NAME)
    finally:
        if conn_server and conn_server.is_connected():
            conn_server.close()

    # No need to pass conn_db around; each function will get its own connection.
    drop_all_tables()
    create_restaurant_tables()
    insert_data_into_tables()
    
    # This is the new, crucial step
    clean_and_standardize_ingredients()
    set_reorder_point_from_recipes()
    
    set_initial_inventory() # This now intelligently brings stock above reorder point
    update_meal_availability() # This will reflect initial stock

    print(f"\nSuccessfully created and populated database '{NEW_DB_NAME}'.")

    # --- Simulating a customer order by fetching from the Order_Items table ---
    customer_order = fetch_order_data_from_db()
    
    if customer_order:
        deplete_inventory_from_order(customer_order)
    else:
        print("\nNo order data to process for inventory depletion.")
    
    update_meal_availability() # Reflect depletion

    check_and_order_ingredients() # Should now correctly detect and order

    verify_purchase_orders() # Verify the orders placed

    # After ordering and delivery, update availability again to reflect new stock
    update_meal_availability()
    
    # Run the debugger to identify which ingredients are still missing
    debug_availability_issue()
    
if __name__ == '__main__':
    main()