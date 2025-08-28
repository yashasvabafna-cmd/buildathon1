from datetime import datetime
import json
import mysql.connector
import time
from inventory_depletion import deplete_inventory_from_order

# --- Helper function to get current inventory for debugging ---
def get_ingredient_current_inventory(ingredient_id, conn):
    """Fetches the current_inventory for a given ingredient_id."""
    if conn is None:
        return None
    try:
        with conn.cursor(dictionary=True) as cursor: # Use dictionary=True for easier access to column names
            query = "SELECT ingredient_name, current_inventory, unit FROM Ingredients WHERE ingredient_id = %s;"
            cursor.execute(query, (ingredient_id,))
            result = cursor.fetchone()
            if result:
                return {"name": result["ingredient_name"], "inventory": result["current_inventory"], "unit": result["unit"]}
            return None
    except mysql.connector.Error as err:
        print(f"Error fetching inventory for ingredient ID {ingredient_id}: {err}")
        return None

def get_unavailable_meals(conn):
    """
    Identifies and returns a list of meals that cannot be made due to insufficient ingredient inventory.
    
    Args:
        conn (mysql.connector.connection.MySQLConnection): An active MySQL database connection.
        
    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents an unavailable meal
                    and includes its name and the missing ingredients.
    """
    if conn is None:
        print("Error: MySQL connection not established. Cannot check for unavailable meals.")
        return []

    unavailable_meals = []
    try:
        with conn.cursor(dictionary=True) as cursor:
            # Get all meals and their required ingredients
            cursor.execute("""
                SELECT 
                    m.meal_id,
                    m.name AS meal_name,
                    ri.quantity AS required_quantity,
                    i.ingredient_name,
                    i.current_inventory,
                    i.unit
                FROM Meals m
                JOIN Recipe_Ingredients ri ON m.meal_id = ri.Meal_ID
                JOIN Ingredients i ON ri.Ingredient_ID = i.ingredient_id
            """)
            
            meal_ingredients_data = cursor.fetchall()
            
            # Group ingredients by meal
            meals_breakdown = {}
            for row in meal_ingredients_data:
                meal_id = row['meal_id']
                if meal_id not in meals_breakdown:
                    meals_breakdown[meal_id] = {
                        'meal_name': row['meal_name'],
                        'ingredients': []
                    }
                meals_breakdown[meal_id]['ingredients'].append({
                    'ingredient_name': row['ingredient_name'],
                    'required_quantity': row['required_quantity'],
                    'current_inventory': row['current_inventory'],
                    'unit': row['unit']
                })

            # Check availability for each meal
            for meal_id, meal_info in meals_breakdown.items():
                missing_ingredients = []
                for ingredient in meal_info['ingredients']:
                    if ingredient['current_inventory'] is None or ingredient['current_inventory'] < ingredient['required_quantity']: # Added None check
                        missing_ingredients.append({
                            'ingredient_name': ingredient['ingredient_name'],
                            'needed': (ingredient['required_quantity'] - (ingredient['current_inventory'] or 0)), # Handle None inventory as 0
                            'unit': ingredient['unit']
                        })
                
                if missing_ingredients:
                    unavailable_meals.append({
                        'meal_name': meal_info['meal_name'],
                        'missing_ingredients': missing_ingredients
                    })
                    
        return unavailable_meals

    except mysql.connector.Error as err:
        print(f"An error occurred while checking for unavailable meals: {err}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

def get_available_menu_meals(conn):
    """
    Fetches all meals and filters out those that are currently unavailable due to insufficient ingredients.

    Args:
        conn (mysql.connector.connection.MySQLConnection): An active MySQL database connection.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents an available meal
                    and includes its ID and name.
    """
    if conn is None:
        print("Error: MySQL connection not established. Cannot fetch available menu meals.")
        return []

    try:
        with conn.cursor(dictionary=True) as cursor:
            # Get all meals
            cursor.execute("SELECT meal_id, name AS meal_name FROM Meals;")
            all_meals = cursor.fetchall()

            # Get unavailable meals
            unavailable_meals_info = get_unavailable_meals(conn)
            unavailable_meal_names = {meal['meal_name'].lower() for meal in unavailable_meals_info}

            # Filter out unavailable meals
            available_meals = [
                meal for meal in all_meals
                if meal['meal_name'].lower() not in unavailable_meal_names
            ]
            
            return available_meals

    except mysql.connector.Error as err:
        print(f"An error occurred while fetching available menu meals: {err}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def insert_orders_from_bot(order_data, conn, deplete_inventory_func):
    """
    Saves order data from the bot's 'cart' list directly to the MySQL 'Order_Items' table.
    Then triggers inventory depletion and prints before/after inventory levels.
    Also, it will now display meals that are unavailable after depletion.
    
    Args:
        order_data (list): A list of Item objects from the confirmed order.
        conn (mysql.connector.connection.MySQLConnection): An active MySQL database connection.
        deplete_inventory_func (function): The function to call for inventory depletion.

    Returns:
        dict: A dictionary with "success" (bool), "unavailable_meals" (list[dict]), and "error" (str, if any).
    """
    if conn is None:
        print("Error: MySQL connection not established. Cannot save order.")
        return {"success": False, "error": "MySQL connection not established."}

    try:
        with conn.cursor() as cursor:
            meal_name_to_id = {}
            try:
                cursor.execute("SELECT name, meal_id FROM Meals")
                meal_name_to_id = {name.lower(): meal_id for name, meal_id in cursor.fetchall()}
            except mysql.connector.Error as err:
                print(f"Error fetching meal_id mapping: {err}")
                return {"success": False, "error": f"Error fetching meal_id mapping: {err}"}

            orders_to_insert = []
            order_id = f"ORDER_{datetime.now().strftime('%Y%m%d%H%M%S%f')}" # Generate a unique order_id
            
            for item in order_data:
                item_name = item.item_name
                quantity = item.quantity
                meal_id = meal_name_to_id.get(item_name.lower())
                
                if meal_id:
                    orders_to_insert.append((order_id, meal_id, quantity))
                else:
                    print(f"Warning: Meal '{item_name}' not found in the database. Skipping.")

            if orders_to_insert:
                # Insert into Order_Items table
                insert_query = "INSERT INTO Order_Items (order_id, meal_id, quantity) VALUES (%s, %s, %s);"
                cursor.executemany(insert_query, orders_to_insert)
                conn.commit()
                print(f"\n--- Order '{order_id}' saved to 'Order_Items' table. ---")

                # --- Pre-depletion Inventory Check (for debugging) ---
                print("\n--- Pre-depletion Inventory Check ---")
                # Get unique ingredient IDs involved in the order for pre-depletion check
                ingredients_to_check_query = """
                    SELECT DISTINCT ri.Ingredient_ID, i.ingredient_name, i.unit
                    FROM Order_Items oi
                    JOIN Recipe_Ingredients ri ON oi.meal_id = ri.Meal_ID
                    JOIN Ingredients i ON ri.Ingredient_ID = i.ingredient_id
                    WHERE oi.order_id = %s;
                """
                cursor.execute(ingredients_to_check_query, (order_id,))
                ingredients_to_check = {row[0]: {"name": row[1], "unit": row[2]} for row in cursor.fetchall()}

                ingredients_before_depletion = {} # {ingredient_id: {name, inventory, unit}}
                for ing_id, ing_data in ingredients_to_check.items():
                    inv = get_ingredient_current_inventory(ing_id, conn)
                    if inv:
                        ingredients_before_depletion[ing_id] = inv

                for ing_id, inv_data in ingredients_before_depletion.items():
                    print(f"  - BEFORE: {inv_data['name']} (ID: {ing_id}): {inv_data['inventory']} {inv_data['unit']}")


                # --- Call inventory depletion from the separate module ---
                # The deplete_inventory_from_order function itself will print detailed DEBUG messages
                deplete_inventory_func(order_data, conn) # Use the passed function

                # --- Post-depletion Inventory Check ---
                print("\n--- Post-depletion Inventory Check ---")
                for ing_id, _ in ingredients_before_depletion.items(): # Use the same IDs checked before
                    inv = get_ingredient_current_inventory(ing_id, conn)
                    if inv:
                        print(f"  - AFTER: {inv['name']} (ID: {ing_id}): {inv['inventory']} {inv['unit']}")

                # --- Check and display unavailable meals ---
                print("\n--- Checking for Unavailable Meals Post-Depletion ---")
                unavailable_meals = get_unavailable_meals(conn)
                if unavailable_meals:
                    print("\nWARNING: The following meals are now unavailable due to insufficient ingredients:")
                    for meal in unavailable_meals:
                        print(f"- {meal['meal_name']}:")
                        for missing_ing in meal['missing_ingredients']:
                            print(f"    - Missing {missing_ing['needed']:.2f} {missing_ing['unit']} of {missing_ing['ingredient_name']}")
                else:
                    print("All meals remain available based on current inventory.")
                
                return {"success": True, "unavailable_meals": unavailable_meals}

            else:
                print("\nNo valid order items to save to the 'Orders' table.")
                return {"success": False, "error": "No valid order items to save."}

    except mysql.connector.Error as err:
        print(f"An error occurred while saving orders to MySQL: {err}")
        return {"success": False, "error": f"MySQL Error: {err}"}
    except Exception as e:
        print(f"An unexpected error occurred while saving orders: {e}")
        return {"success": False, "error": f"Unexpected Error: {e}"}

