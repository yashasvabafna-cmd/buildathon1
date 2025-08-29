import mysql.connector
import json
from datetime import datetime
from dotenv import load_dotenv
import os
from decimal import Decimal # Import Decimal to handle database types correctly
load_dotenv()
# Corrected: Import Item from Classes.py
from Classes import Item # Assuming Item class is defined in Classes.py

# --- IMPORTANT: MySQL DB_CONFIG for inventory_depletion.py ---
# Ensure these details match your 'restaurant_new_db' setup.
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',        # Your MySQL username
    'password': '12345678', # Your MySQL password
    'database': os.getenv('DB_NAME') # The database where 'Ingredients' and 'Recipe_Ingredients' tables are
}
# ----------------------------------------------------

def get_mysql_connection():
    """Establishes and returns a connection to the MySQL database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        print("Please ensure your MySQL server is running and connection details are correct.")
        return None

def deplete_inventory_from_order(order_data_items, conn):
    """
    Depletes ingredients from the Ingredients table based on the confirmed order items.
    
    Args:
        order_data_items (list[Item]): A list of Item objects from the confirmed order.
        conn (mysql.connector.connection.MySQLConnection): An active MySQL database connection.
    """
    if conn is None:
        print("Error: MySQL connection not established. Cannot deplete inventory.")
        return False

    try:
        with conn.cursor(dictionary=True) as cursor: # Use dictionary=True for easier access to column names
            print("\n--- Starting inventory depletion from single order ---")
            
            # Step 1: Get meal-ingredient relationships and current inventory for ordered items
            # This complex query aims to get all necessary info in one go
            
            # First, collect all item_names from order_data_items
            item_names_in_order = [item.item_name for item in order_data_items]
            
            if not item_names_in_order:
                print("No items in the order to deplete inventory.")
                return False

            # Fetch meal_ids for the ordered item_names
            meal_name_to_id_query = "SELECT name, meal_id FROM Meals WHERE name IN (%s);"
            placeholders = ', '.join(['%s'] * len(item_names_in_order))
            cursor.execute(meal_name_to_id_query % placeholders, tuple(item_names_in_order))
            meal_name_to_id_map = {row['name']: row['meal_id'] for row in cursor.fetchall()}
            
            meal_ids_in_order_set = set()
            for item in order_data_items:
                meal_id = meal_name_to_id_map.get(item.item_name)
                if meal_id:
                    # Assign meal_id to item if it exists, for later use if needed, but primarily collect for query
                    item.item_id = meal_id # This modifies the Item object in the list
                    meal_ids_in_order_set.add(meal_id)
            
            if not meal_ids_in_order_set:
                print("No valid meal IDs found in the order to deplete inventory.")
                return False

            meal_ids_in_order_tuple = tuple(meal_ids_in_order_set)

            # Fetch detailed ingredient info for recipes of meals in the order
            query_recipes = """
                SELECT 
                    ri.Ingredient_ID,
                    ri.quantity AS recipe_quantity_per_meal,
                    i.ingredient_name,
                    i.current_inventory,
                    i.unit AS recipe_unit,
                    m.name AS meal_name,
                    m.meal_id
                FROM Recipe_Ingredients ri
                JOIN Ingredients i ON ri.Ingredient_ID = i.ingredient_id
                JOIN Meals m ON ri.Meal_ID = m.meal_id
                WHERE m.meal_id IN (%s);
            """ % ','.join(['%s'] * len(meal_ids_in_order_tuple))
            
            cursor.execute(query_recipes, meal_ids_in_order_tuple)
            recipe_details = cursor.fetchall()

            # Prepare for updates
            ingredients_to_update = {} # {ingredient_id: {'name': name, 'current_inventory': qty, 'unit': unit, 'total_depletion': qty}}

            # Calculate total depletion for each ingredient across all ordered meals
            for order_item in order_data_items:
                meal_name = order_item.item_name
                ordered_quantity = order_item.quantity
                meal_id_for_item = meal_name_to_id_map.get(meal_name) # Get meal_id using the map

                if not meal_id_for_item:
                    print(f"DEBUG: Could not find meal_id for '{meal_name}'. Skipping depletion for this item.")
                    continue

                print(f"DEBUG: Processing order for '{meal_name}' (Quantity: {ordered_quantity})")

                # Filter recipe_details for the current meal
                meal_recipe_ingredients = [
                    detail for detail in recipe_details
                    if detail['meal_id'] == meal_id_for_item
                ]
                
                if not meal_recipe_ingredients:
                    print(f"DEBUG: No recipe ingredients found for Meal ID {meal_id_for_item} ('{meal_name}'). Skipping.")
                    continue

                for ingredient_detail in meal_recipe_ingredients:
                    ingredient_id = ingredient_detail['Ingredient_ID']
                    recipe_qty_per_meal = ingredient_detail['recipe_quantity_per_meal']
                    ingredient_name = ingredient_detail['ingredient_name']
                    current_inventory = ingredient_detail['current_inventory']
                    recipe_unit = ingredient_detail['recipe_unit']

                    # Ensure we handle Decimal types correctly for arithmetic
                    recipe_qty_per_meal = float(recipe_qty_per_meal) if isinstance(recipe_qty_per_meal, Decimal) else recipe_qty_per_meal
                    current_inventory = float(current_inventory) if isinstance(current_inventory, Decimal) else current_inventory
                    
                    # Calculate depletion amount for this ingredient for this item
                    depletion_amount = ordered_quantity * recipe_qty_per_meal

                    if ingredient_id not in ingredients_to_update:
                        ingredients_to_update[ingredient_id] = {
                            'name': ingredient_name,
                            'current_inventory': current_inventory,
                            'unit': recipe_unit,
                            'total_depletion': 0.0
                        }
                    ingredients_to_update[ingredient_id]['total_depletion'] += depletion_amount
            
            # Step 2: Update inventory for each ingredient
            for ingredient_id, data in ingredients_to_update.items():
                ingredient_name = data['name']
                current_inventory = data['current_inventory']
                total_depletion_amount = data['total_depletion']
                recipe_unit = data['unit']

                print(f"Depleting inventory for '{ingredient_name}':")
                print(f"DEBUG:     Current Inventory: {current_inventory:.2f} {recipe_unit}") # Formatted for clarity
                print(f"DEBUG:     Total Depletion Amount: {total_depletion_amount:.2f} {recipe_unit}") # Formatted for clarity

                # Handle cases where current_inventory might be None (newly added ingredient not initialized)
                if current_inventory is None:
                    print(f"  - WARNING: Inventory for '{ingredient_name}' is NULL. Treating as 0 for depletion.")
                    current_inventory = 0.0
                
                new_inventory = current_inventory - total_depletion_amount

                # Step 3: Update the current_inventory in the Ingredients table
                update_inventory_query = """
                UPDATE Ingredients
                SET current_inventory = %s
                WHERE ingredient_id = %s;
                """
                cursor.execute(update_inventory_query, (new_inventory, ingredient_id))
                print(f"  - Depleted {total_depletion_amount:.2f} {recipe_unit} of '{ingredient_name}'. New inventory: {new_inventory:.2f} {recipe_unit}")
        
        conn.commit()
        print("Inventory depletion completed successfully.")
        return True

    except mysql.connector.Error as err:
        print(f"An error occurred during inventory depletion: {err}")
        conn.rollback() # Rollback changes if an error occurs
        return False
    except Exception as e:
        print(f"An unexpected error occurred during inventory depletion: {e}")
        return False
