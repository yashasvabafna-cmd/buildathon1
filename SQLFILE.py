import sqlite3
import pandas as pd
import json
import numpy as np
import os # Import os module to check for file existence
from datetime import datetime # Import datetime for timestamp

def create_restaurant_tables(db_name='restaurant.db'):
    """
    Connects to an SQLite database and creates all necessary tables for
    a restaurant inventory and recipe management system.

    Args:
        db_name (str): The name of the SQLite database file.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        c = conn.cursor()

        # Create Suppliers table
        c.execute('''
            CREATE TABLE IF NOT EXISTS Suppliers (
                Supplier_ID TEXT PRIMARY KEY,
                Supplier_Name TEXT NOT NULL,
                Contact_Info TEXT,
                Lead_Time INTEGER,
                Payment_Terms TEXT
            );
        ''')
        print("Table 'Suppliers' created or already exists.")

        # Create Meals table
        c.execute('''
            CREATE TABLE IF NOT EXISTS Meals (
                meal_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT,
                category TEXT,
                price REAL,
                Chef_chef_id INTEGER
            );
        ''')
        print("Table 'Meals' created or already exists.")

        # Create Ingredients table
        c.execute('''
            CREATE TABLE IF NOT EXISTS Ingredients (
                ingredient_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ingredient_name TEXT NOT NULL UNIQUE,
                unit TEXT,
                current_inventory REAL,
                reorder_level REAL,
                supplier_id TEXT,
                FOREIGN KEY (supplier_id) REFERENCES Suppliers(Supplier_ID)
            );
        ''')
        print("Table 'Ingredients' created or already exists.")

        # Create Recipes table
        # This table stores the recipe details including a JSON string for ingredients
        c.execute('''
            CREATE TABLE IF NOT EXISTS Recipes (
                meal_id INTEGER PRIMARY KEY,
                meal_name TEXT NOT NULL,
                ingredients TEXT, -- Stored as a JSON string
                recipe TEXT,
                FOREIGN KEY (meal_id) REFERENCES Meals(meal_id)
            );
        ''')
        print("Table 'Recipes' created or already exists.")

        # Create Recipe_Ingredients linking meals and specific ingredient IDs
        c.execute('''
            CREATE TABLE IF NOT EXISTS Recipe_Ingredients (
                Recipe_Ingredient_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Meal_ID INTEGER NOT NULL,
                Ingredient_ID INTEGER NOT NULL,
                Quantity REAL,
                FOREIGN KEY (Meal_ID) REFERENCES Meals(meal_id),
                FOREIGN KEY (Ingredient_ID) REFERENCES Ingredients(ingredient_id)
            );
        ''')
        print("Table 'Recipe_Ingredients' created or already exists.")

        # Create Purchase_Orders table
        c.execute('''
            CREATE TABLE IF NOT EXISTS Purchase_Orders (
                PO_ID INTEGER PRIMARY KEY,
                Supplier_ID TEXT NOT NULL,
                Order_Date TEXT,
                Expected_Delivery_Date TEXT,
                Actual_Delivery_Date TEXT,
                Status TEXT,
                Total_Cost REAL,
                FOREIGN KEY (Supplier_ID) REFERENCES Suppliers(Supplier_ID)
            );
        ''')
        print("Table 'Purchase_Orders' created or already exists.")

        # Create PO_Items table
        c.execute('''
            CREATE TABLE IF NOT EXISTS PO_Items (
                PO_Item_ID INTEGER PRIMARY KEY,
                PO_ID INTEGER NOT NULL,
                Ingredient_ID INTEGER NOT NULL,
                Quantity_Ordered REAL,
                Cost_at_Purchase REAL,
                FOREIGN KEY (PO_ID) REFERENCES Purchase_Orders(PO_ID),
                FOREIGN KEY (Ingredient_ID) REFERENCES Ingredients(ingredient_id)
            );
        ''')
        print("Table 'PO_Items' created or already exists.")

        # Create Waste table
        c.execute('''
            CREATE TABLE IF NOT EXISTS Waste (
                Waste_ID INTEGER PRIMARY KEY,
                Ingredient_ID INTEGER NOT NULL,
                Quantity REAL,
                Date TEXT,
                Reason TEXT,
                Cost_of_Waste REAL,
                FOREIGN KEY (Ingredient_ID) REFERENCES Ingredients(ingredient_id)
            );
        ''')
        print("Table 'Waste' created or already exists.")

        # Create Orders table (New table for this request)
        c.execute('''
            CREATE TABLE IF NOT EXISTS Orders (
                Order_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                item_name TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                modifiers TEXT
            );
        ''')
        print("Table 'Orders' created or already exists.")


        conn.commit()
        print("\nAll tables created successfully!")

        # Verify table creation
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = c.fetchall()
        print("\nTables in the database:")
        for table in tables:
            print(table[0])

    except sqlite3.Error as e:
        print(f"An error occurred during table creation: {e}")
    finally:
        if conn:
            conn.close()

def insert_data_into_tables(db_name='restaurant.db'):
    """
    Inserts data into the created tables from CSV and JSON files.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        c = conn.cursor()

        # --- Insert into Suppliers table (Hardcoded as per previous outputs) ---
        suppliers_data = [
            ('Supplier001', 'City Fresh Produce', '123-456-7890', 2, 'Net 30'),
            ('Supplier002', 'Prime Meats Co.', '987-654-3210', 3, 'Net 15'),
            ('Supplier003', 'Spice Hub Distributors', '555-111-2222', 4, 'Net 45')
        ]
        c.executemany("INSERT OR IGNORE INTO Suppliers (Supplier_ID, Supplier_Name, Contact_Info, Lead_Time, Payment_Terms) VALUES (?, ?, ?, ?, ?)", suppliers_data)
        print("\nInserted/Ignored data into Suppliers table.")

        # --- Insert into Meals table from meals.csv ---
        meals_df = None
        if os.path.exists('meals.csv'):
            meals_df = pd.read_csv('meals.csv')
            meals_data = meals_df[['meal_id', 'name', 'type', 'category', 'price', 'Chef_chef_id']].values.tolist()
            c.executemany("INSERT OR IGNORE INTO Meals (meal_id, name, type, category, price, Chef_chef_id) VALUES (?, ?, ?, ?, ?, ?)", meals_data)
            print("Inserted/Ignored data into Meals table.")
        else:
            print("Warning: meals.csv not found. Skipping Meals data insertion.")

        # --- Insert into Ingredients table from ingredients_listcsv.csv ---
        ingredients_df = None
        ingredient_name_to_id = {}
        if os.path.exists('ingredients_listcsv.csv'):
            ingredients_df = pd.read_csv('ingredients_listcsv.csv')
            ingredients_data = ingredients_df[['ingredient_name', 'unit', 'current_inventory', 'reorder_level', 'supplier_id']].values.tolist()
            # SQLite auto-increments ingredient_id, so we don't include it in the insert statement
            c.executemany("INSERT OR IGNORE INTO Ingredients (ingredient_name, unit, current_inventory, reorder_level, supplier_id) VALUES (?, ?, ?, ?, ?)", ingredients_data)
            print("Inserted/Ignored data into Ingredients table.")

            # Re-fetch ingredient_name to ID mapping for Recipe_Ingredients table
            ingredient_name_to_id = dict(c.execute("SELECT ingredient_name, ingredient_id FROM Ingredients").fetchall())
            ingredient_name_to_id = {k.lower(): v for k, v in ingredient_name_to_id.items()} # Ensure lower case for matching

            
           

        else:
            print("Warning: ingredients_listcsv.csv not found. Skipping Ingredients data insertion and related recipe linking.")


        # --- Insert into Recipes table from recipes_batch_2.json ---
        if os.path.exists('recipes_batch_2.json'):
            with open('recipes_batch_2.json', 'r') as f:
                recipes_batch_2 = json.load(f)

            recipes_to_insert = []
            for recipe_data in recipes_batch_2:
                meal_id = recipe_data['meal_id']
                meal_name = recipe_data['meal_name']
                # Escape single quotes in ingredients and recipe strings for SQL
                ingredients_str = json.dumps(recipe_data['ingredients']).replace("'", "''")
                recipe_str = recipe_data['recipe'].replace("'", "''")
                recipes_to_insert.append((meal_id, meal_name, ingredients_str, recipe_str))

            c.executemany("INSERT OR IGNORE INTO Recipes (meal_id, meal_name, ingredients, recipe) VALUES (?, ?, ?, ?)", recipes_to_insert)
            print("Inserted/Ignored data into Recipes table.")

            # --- Insert into Recipe_Ingredients table based on recipes_batch_2.json ---
            new_recipe_ingredient_inserts = []
            # Find the maximum existing Recipe_Ingredient_ID to avoid conflicts
            c.execute("SELECT MAX(Recipe_Ingredient_ID) FROM Recipe_Ingredients;")
            max_id = c.fetchone()[0]
            current_recipe_ingredient_id = (max_id if max_id is not None else 0) + 1
            conn = sqlite3.connect('restaurant.db')
            c = conn.cursor()

            try:
                # Add the new column to the table
                c.execute("ALTER TABLE Recipe_Ingredients ADD COLUMN Recipe_Unit TEXT;")
                conn.commit()
                print("Column 'Recipe_Unit' added successfully to 'Recipe_Ingredients'.")

                
            except sqlite3.Error as e:
                print(f"An error occurred: {e}")

            finally:
                if conn:
                    conn.close()
            for recipe_data in recipes_batch_2:
                meal_id_from_json = recipe_data['meal_id']
                ingredients_from_json = recipe_data['ingredients']

                for ingredient_name_raw in ingredients_from_json:
                    ingredient_name_clean = ingredient_name_raw.split(',')[0].split('(')[0].strip().lower()
                    # Handle specific cleaning cases and lenient matching
                    if 'for litti:' in ingredient_name_clean:
                        ingredient_name_clean = 'whole wheat flour'
                    elif 'for chokha:' in ingredient_name_clean:
                        ingredient_name_clean = 'eggplant'
                    elif 'spices:' in ingredient_name_clean:
                        ingredient_name_clean = 'garam masala'
                    elif 'oil for cooking' in ingredient_name_clean:
                        ingredient_name_clean = 'cooking oil'

                    ingredient_id = ingredient_name_to_id.get(ingredient_name_clean)

                    # More lenient matching for common variations
                    if ingredient_id is None:
                        if 'paneer' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('paneer')
                        elif 'yogurt' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('thick yogurt')
                        elif 'ginger-garlic' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('ginger-garlic paste')
                        elif 'turmeric' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('turmeric')
                        elif 'chili powder' in ingredient_name_clean or 'red chili' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('kashmiri red chili powder')
                        elif 'garam masala' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('garam masala')
                        elif 'chaat masala' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('chaat masala')
                        elif 'lemon juice' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('lemon juice')
                        elif 'onion' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('onion')
                        elif 'bell pepper' in ingredient_name_clean or 'capsicum' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('bell pepper')
                        elif 'salt' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('salt')
                        elif 'vegetable oil' in ingredient_name_clean or 'oil' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('cooking oil')
                        elif 'all-purpose flour' in ingredient_name_clean or 'flour' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('maida')
                        elif 'ghee' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('ghee')
                        elif 'potato' in ingredient_name_clean or 'potatoes' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('potato')
                        elif 'peas' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('peas')
                        elif 'green chilies' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('green chilies')
                        elif 'cabbage' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('mixed vegetables')
                        elif 'carrot' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('mixed vegetables')
                        elif 'french beans' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('mixed vegetables')
                        elif 'gobi' in ingredient_name_clean or 'cauliflower' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('cauliflower')
                        elif 'cornflour' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('cornflour')
                        elif 'soy sauce' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('soy sauce')
                        elif 'chili sauce' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('chili sauce')
                        elif 'spring onion' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('spring onion')
                        elif 'tomato' in ingredient_name_clean or 'tomatoes' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('tomatoes')
                        elif 'ginger' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('ginger')
                        elif 'garlic' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('garlic')
                        elif 'curd' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('curd')
                        elif 'chickpea flour' in ingredient_name_clean or 'besan' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('chickpea flour')
                        elif 'whole wheat flour' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('whole wheat flour')
                        elif 'fenugreek leaves' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('coriander leaves')
                        elif 'pickle masala' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('garam masala')
                        elif 'sattu' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('chickpea flour')
                        elif 'eggplant' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('mixed vegetables')
                        elif 'moong dal' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('peas')
                        elif 'red lentils' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('peas')
                        elif 'coconut milk' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('milk')
                        elif 'chicken' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('chicken')
                        elif 'mutton' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('mutton')
                        elif 'fish' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('fish')
                        elif 'prawn' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('prawn')
                        elif 'rice' in ingredient_name_clean and 'basmati' not in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('rice')
                        elif 'basmati rice' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('basmati rice')
                        elif 'milk' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('milk')
                        elif 'sugar' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('sugar')
                        elif 'semolina' in ingredient_name_clean or 'suji' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('semolina')
                        elif 'jaggery' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('jaggery')
                        elif 'black pepper' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('black pepper')
                        elif 'cumin seeds' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('cumin seeds')
                        elif 'mustard seeds' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('mustard seeds')
                        elif 'coriander leaves' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('coriander leaves')
                        elif 'mint leaves' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('mint leaves')
                        elif 'fresh cream' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('fresh cream')
                        elif 'cashew nuts' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('cashew nuts')
                        elif 'butter' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('butter')
                        elif 'mixed vegetables' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('mixed vegetables')
                        elif 'chickpea flour' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('chickpea flour')
                        elif 'maida' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('maida')
                        elif 'whole wheat flour' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('whole wheat flour')
                        elif 'jaggery' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('jaggery')
                        elif 'semolina' in ingredient_name_clean:
                            ingredient_id = ingredient_name_to_id.get('semolina')

                    if ingredient_id is None:
                        # print(f"Warning: Ingredient '{ingredient_name_raw}' (cleaned: '{ingredient_name_clean}') for Meal ID {meal_id_from_json} not found in ingredients. Skipping.")
                        continue # Skip if no matching ingredient is found

                    quantity = round(np.random.uniform(0.01, 0.5), 3)
                    new_recipe_ingredient_inserts.append((current_recipe_ingredient_id, meal_id_from_json, ingredient_id, quantity))
                    current_recipe_ingredient_id += 1

            # Ensure all 100 meals have at least one recipe entry in Recipe_Ingredients
            existing_meal_ids_in_recipes_ingredients = {item[1] for item in new_recipe_ingredient_inserts}

            # If meals_df was loaded, iterate through its meal_ids
            if meals_df is not None:
                common_ingredient_ids = [
                    ingredient_name_to_id.get('salt'),
                    ingredient_name_to_id.get('cooking oil'),
                    ingredient_name_to_id.get('onion'),
                    ingredient_name_to_id.get('tomatoes'),
                    ingredient_name_to_id.get('ginger-garlic paste'),
                    ingredient_name_to_id.get('turmeric'),
                    ingredient_name_to_id.get('garam masala'),
                    ingredient_name_to_id.get('rice'),
                    ingredient_name_to_id.get('paneer'),
                    ingredient_name_to_id.get('chicken'),
                    ingredient_name_to_id.get('mutton')
                ]
                common_ingredient_ids = [id for id in common_ingredient_ids if id is not None]

                for meal_id in meals_df['meal_id'].unique():
                    if meal_id not in existing_meal_ids_in_recipes_ingredients:
                        if common_ingredient_ids:
                            num_ingredients_to_add = min(len(common_ingredient_ids), np.random.randint(2, 4))
                            selected_placeholder_ingredients = np.random.choice(common_ingredient_ids, num_ingredients_to_add, replace=False)
                            for ingredient_id in selected_placeholder_ingredients:
                                quantity = round(np.random.uniform(0.01, 0.2), 3)
                                new_recipe_ingredient_inserts.append((current_recipe_ingredient_id, meal_id, ingredient_id, quantity))
                                current_recipe_ingredient_id += 1
                        else:
                            print(f"Warning: Meal ID {meal_id} not in recipes_batch_2.json and no suitable common ingredients found for placeholder. Skipping recipe creation for this meal.")
            # A dictionary to map cleaned ingredient names to realistic quantities and units
            realistic_quantities = {
                'salt': (0.01, 'kg'),
                'cooking oil': (0.05, 'litres'),
                'onion': (0.1, 'kg'),
                'tomatoes': (0.15, 'kg'),
                'ginger-garlic paste': (0.02, 'kg'),
                'turmeric': (0.005, 'kg'),
                'garam masala': (0.005, 'kg'),
                'rice': (0.2, 'kg'),
                'paneer': (0.1, 'kg'),
                'chicken': (0.2, 'kg'),
                'mutton': (0.2, 'kg'),
                'whole wheat flour': (0.1, 'kg'),
                'eggplant': (0.2, 'kg'),
                'maida': (0.1, 'kg'),
                'potato': (0.15, 'kg'),
                'mixed vegetables': (0.2, 'kg'),
                'sugar': (0.05, 'kg'),
                'milk': (0.25, 'litres'),
                'fish': (0.2, 'kg'),
                'prawn': (0.2, 'kg'),
                'thick yogurt': (0.1, 'kg'),
                'lemon juice': (0.05, 'litres'),
                'bell pepper': (0.1, 'kg'),
                'ghee': (0.05, 'kg'),
                'peas': (0.1, 'kg'),
                'green chilies': (0.01, 'kg'),
                'cauliflower': (0.2, 'kg'),
                'cornflour': (0.05, 'kg'),
                'soy sauce': (0.05, 'litres'),
                'chili sauce': (0.05, 'litres'),
                'spring onion': (0.02, 'kg'),
                'ginger': (0.01, 'kg'),
                'garlic': (0.01, 'kg'),
                'curd': (0.1, 'kg'),
                'chickpea flour': (0.1, 'kg'),
                'whole wheat flour': (0.1, 'kg'),
                'coriander leaves': (0.005, 'kg'),
                'jaggery': (0.1, 'kg'),
                'semolina': (0.1, 'kg'),
                'black pepper': (0.005, 'kg'),
                'cumin seeds': (0.005, 'kg'),
                'mustard seeds': (0.005, 'kg'),
                'mint leaves': (0.005, 'kg'),
                'fresh cream': (0.1, 'litres'),
                'cashew nuts': (0.05, 'kg'),
                'butter': (0.05, 'kg'),
                'basmati rice': (0.2, 'kg')
            }
            c.executemany("INSERT OR IGNORE INTO Recipe_Ingredients (Recipe_Ingredient_ID, Meal_ID, Ingredient_ID, Quantity, Recipe_Unit) VALUES (?, ?, ?, ?, ?)", new_recipe_ingredient_inserts)
            print("Inserted/Ignored data into Recipe_Ingredients table.")

        else:
            print("Warning: recipes_batch_2.json not found. Skipping Recipes and Recipe_Ingredients data insertion.")


        # --- Insert into Purchase_Orders table (Hardcoded as per previous outputs) ---
        purchase_orders_data = [
            (1, 'Supplier003', '2025-08-05', '2025-08-09', '2025-08-08', 'Received', 6127.76),
            (2, 'Supplier001', '2025-08-10', '2025-08-12', '2025-08-13', 'Received', 7327.0),
            (3, 'Supplier001', '2025-08-12', '2025-08-14', '2025-08-16', 'Received', 64.69),
            (4, 'Supplier003', '2025-08-09', '2025-08-13', '2025-08-14', 'Received', 572.5),
            (5, 'Supplier002', '2025-08-15', '2025-08-18', '2025-08-18', 'Received', 14537.0),
            (6, 'Supplier001', '2025-08-11', '2025-08-13', '2025-08-15', 'Received', 7965.78),
            (7, 'Supplier002', '2025-08-26', '2025-08-29', '2025-08-29', 'Shipped', 14521.91),
            (8, 'Supplier003', '2025-08-11', '2025-08-15', '2025-08-10', 'Received', 2341.6),
            (9, 'Supplier003', '2025-08-20', '2025-08-24', '2025-08-22', 'Shipped', 10738.5),
            (10, 'Supplier001', '2025-08-01', '2025-08-03', '2025-08-04', 'Received', 11090.86),
            (11, 'Supplier002', '2025-08-20', '2025-08-23', '2025-08-22', 'Shipped', 13328.75),
            (12, 'Supplier003', '2025-08-13', '2025-08-17', '2025-08-16', 'Received', 255.45),
            (13, 'Supplier001', '2025-08-01', '2025-08-03', '2025-08-02', 'Received', 10174.5),
            (14, 'Supplier003', '2025-08-07', '2025-08-11', '2025-08-10', 'Received', 6055.0),
            (15, 'Supplier001', '2025-08-01', '2025-08-03', '2025-08-01', 'Received', 5472.0),
            (16, 'Supplier002', '2025-08-21', '2025-08-24', '2025-08-21', 'Shipped', 15155.0),
            (17, 'Supplier002', '2025-08-11', '2025-08-14', '2025-08-15', 'Received', 2465.0),
            (18, 'Supplier003', '2025-08-03', '2025-08-07', '2025-08-06', 'Received', 6636.5),
            (19, 'Supplier001', '2025-08-17', '2025-08-19', '2025-08-19', 'Received', 9314.95),
            (20, 'Supplier003', '2025-08-04', '2025-08-08', '2025-08-08', 'Received', 5962.0)
        ]
        c.executemany("INSERT OR IGNORE INTO Purchase_Orders (PO_ID, Supplier_ID, Order_Date, Expected_Delivery_Date, Actual_Delivery_Date, Status, Total_Cost) VALUES (?, ?, ?, ?, ?, ?, ?)", purchase_orders_data)
        print("Inserted/Ignored data into Purchase_Orders table.")

        # --- Insert into PO_Items table (Hardcoded as per previous outputs) ---
        po_items_data = [
            (1, 1, 48, 9.51, 951.0), (2, 1, 33, 73.64, 3682.0), (3, 1, 40, 7.32, 732.0),
            (4, 1, 5, 35.52, 17.76), (5, 1, 46, 7.45, 745.0), (6, 2, 27, 3.7, 370.0),
            (7, 2, 1, 69.57, 6957.0), (8, 3, 3, 58.39, 29.2), (9, 3, 36, 70.99, 35.49),
            (10, 4, 15, 11.45, 572.5), (11, 5, 33, 94.16, 4708.0), (12, 5, 35, 68.16, 6816.0),
            (13, 5, 29, 28.62, 1431.0), (14, 5, 15, 31.64, 1582.0), (15, 6, 5, 28.55, 14.28),
            (16, 6, 13, 62.63, 6263.0), (17, 6, 33, 33.77, 1688.5), (18, 7, 44, 81.71, 8171.0),
            (19, 7, 45, 63.59, 31.8), (20, 7, 30, 23.98, 1199.0), (21, 7, 10, 50.8, 508.0),
            (22, 7, 1, 33.7, 3370.0), (23, 8, 48, 23.41, 2341.0), (24, 8, 33, 1.0, 0.5),
            (25, 8, 39, 0.05, 5.0), (26, 9, 28, 97.46, 9746.0), (27, 9, 39, 9.92, 992.0),
            (28, 10, 48, 11.0, 1100.0), (29, 10, 2, 63.07, 3153.5), (30, 10, 17, 34.34, 17.17),
            (31, 10, 1, 68.27, 6827.0), (32, 11, 48, 6.78, 678.0), (33, 11, 33, 73.18, 3659.0),
            (34, 11, 27, 89.92, 8992.0), (35, 12, 14, 51.09, 25.54), (36, 12, 33, 45.98, 22.99)
        ]
        c.executemany("INSERT OR IGNORE INTO PO_Items (PO_Item_ID, PO_ID, Ingredient_ID, Quantity_Ordered, Cost_at_Purchase) VALUES (?, ?, ?, ?, ?)", po_items_data)
        print("Inserted/Ignored data into PO_Items table.")

        # --- Insert into Waste table (Hardcoded as per previous outputs) ---
        waste_data = [
            (1, 29, 0.49, '2025-08-19', 'Returned by Customer', 24.5), (2, 32, 0.41, '2025-08-19', 'Cooking Error', 41.0),
            (3, 31, 0.29, '2025-08-17', 'Spoiled', 29.0), (4, 35, 0.32, '2025-08-19', 'Dropped', 32.0),
            (5, 16, 0.31, '2025-08-15', 'Returned by Customer', 31.0), (6, 48, 0.43, '2025-08-17', 'Cooking Error', 43.0),
            (7, 11, 0.31, '2025-08-18', 'Cooking Error', 31.0), (8, 24, 0.49, '2025-08-19', 'Returned by Customer', 49.0),
            (9, 4, 21.89, '2025-08-15', 'Expired', 10.95), (10, 5, 74.71, '2025-08-17', 'Returned by Customer', 37.35),
            (11, 19, 0.46, '2025-08-17', 'Cooking Error', 46.0), (12, 12, 0.25, '2025-08-16', 'Returned by Customer', 12.5),
            (13, 36, 19.56, '2025-08-19', 'Spoiled', 9.78), (14, 34, 0.21, '2025-08-17', 'Dropped', 21.0),
            (15, 40, 0.34, '2025-08-15', 'Expired', 34.0), (16, 25, 0.25, '2025-08-19', 'Cooking Error', 25.0),
            (17, 7, 51.6, '2025-08-19', 'Expired', 25.8), (18, 35, 0.11, '2025-08-19', 'Expired', 11.0),
            (19, 33, 0.24, '2025-08-16', 'Returned by Customer', 12.0), (20, 25, 0.19, '2025-08-16', 'Dropped', 19.0),
            (21, 42, 0.08, '2025-08-19', 'Expired', 4.0), (22, 19, 0.08, '2025-08-17', 'Returned by Customer', 8.0),
            (23, 42, 0.16, '2025-08-20', 'Expired', 8.0), (24, 26, 0.05, '2025-08-16', 'Spoiled', 5.0),
            (25, 29, 0.42, '2025-08-15', 'Dropped', 21.0), (26, 21, 0.02, '2025-08-19', 'Cooking Error', 1.0),
            (27, 47, 0.26, '2025-08-16', 'Expired', 13.0), (28, 26, 0.02, '2025-08-19', 'Returned by Customer', 2.0),
            (29, 29, 0.44, '2025-08-18', 'Spoiled', 22.0), (30, 27, 0.43, '2025-08-17', 'Cooking Error', 43.0),
            (31, 23, 0.28, '2025-08-16', 'Expired', 14.0), (32, 1, 0.17, '2025-08-15', 'Dropped', 17.0),
            (33, 41, 0.17, '2025-08-19', 'Returned by Customer', 8.5), (34, 43, 0.43, '2025-08-15', 'Cooking Error', 21.5),
            (35, 17, 30.65, '2025-08-19', 'Expired', 15.32), (36, 45, 0.22, '2025-08-19', 'Returned by Customer', 11.0),
            (37, 2, 0.03, '2025-08-15', 'Spoiled', 1.5), (38, 26, 0.32, '2025-08-19', 'Dropped', 32.0),
            (39, 43, 0.33, '2025-08-18', 'Cooking Error', 16.5), (40, 48, 0.47, '2025-08-19', 'Expired', 47.0),
            (41, 35, 0.33, '2025-08-16', 'Returned by Customer', 33.0), (42, 28, 0.05, '2025-08-17', 'Spoiled', 5.0),
            (43, 33, 0.41, '2025-08-15', 'Dropped', 20.5), (44, 15, 77.26, '2025-08-18', 'Cooking Error', 38.63),
            (45, 41, 0.25, '2025-08-16', 'Expired', 12.5), (46, 17, 95.0, '2025-08-15', 'Returned by Customer', 47.5),
            (47, 33, 0.03, '2025-08-17', 'Spoiled', 1.5), (48, 43, 0.37, '2025-08-19', 'Dropped', 18.5),
            (49, 39, 0.48, '2025-08-18', 'Cooking Error', 48.0), (50, 46, 0.05, '2025-08-16', 'Expired', 5.0)
        ]
        c.executemany("INSERT OR IGNORE INTO Waste (Waste_ID, Ingredient_ID, Quantity, Date, Reason, Cost_of_Waste) VALUES (?, ?, ?, ?, ?, ?)", waste_data)
        print("Inserted/Ignored data into Waste table.")

        # --- Insert into Orders table from orders.csv (New for this request) ---
        if os.path.exists('orders.csv'):
            orders_df = pd.read_csv('orders.csv')
            # orders_df['timestamp'] might be strings, ensure correct format if needed
            orders_data = orders_df[['timestamp', 'item_name', 'quantity', 'modifiers']].values.tolist()
            c.executemany("INSERT INTO Orders (timestamp, item_name, quantity, modifiers) VALUES (?, ?, ?, ?)", orders_data)
            print("Inserted data into Orders table.")
        else:
            print("Warning: orders.csv not found. Skipping Orders data insertion.")


        conn.commit()
        print("\nAll data inserted successfully!")

        # --- Inventory Depletion Logic (New for this request) ---
        deplete_inventory_from_orders(conn) # Pass the connection to the new function

    except sqlite3.Error as e:
        print(f"An error occurred during data insertion: {e}")
    finally:
        if conn:
            conn.close()

def deplete_inventory_with_units(conn):
    """
    Depletes ingredient inventory based on processed orders,
    correctly handling different units of measure.
    """
    c = conn.cursor()
    print("\n--- Starting Inventory Depletion (with units) ---")
    
    # Define a conversion dictionary for standardizing units to a base unit (e.g., grams)
    unit_conversions = {
        'kg': 1000.0,
        'gram': 1.0,
        'litres': 1000.0,
        'ml': 1.0,
        'pieces': 1.0,
        'units': 1.0,
    }

    try:
        # Step 1: Read all necessary data into Pandas DataFrames for easier manipulation.
        meals_df = pd.read_sql_query("SELECT meal_id, name FROM Meals;", conn)
        recipes_df = pd.read_sql_query("SELECT Meal_ID, Ingredient_ID, Quantity FROM Recipe_Ingredients;", conn)
        ingredients_df = pd.read_sql_query("SELECT ingredient_id, current_inventory, unit FROM Ingredients;", conn)
        orders_df = pd.read_sql_query("SELECT item_name, quantity FROM Orders;", conn)

        # Step 2: Join the tables to get a complete view of orders and their ingredients.
        # Use a case-insensitive join on meal names.
        depletion_data = pd.merge(orders_df, meals_df, left_on=orders_df['item_name'].str.lower(), right_on=meals_df['name'].str.lower())
        depletion_data = pd.merge(depletion_data, recipes_df, left_on='meal_id', right_on='Meal_ID')
        depletion_data = pd.merge(depletion_data, ingredients_df, left_on='Ingredient_ID', right_on='ingredient_id')

        # Step 3: Calculate the total quantity needed for each ingredient, with unit conversion.
        depletion_data['conversion_factor'] = depletion_data['unit'].map(unit_conversions)
        depletion_data['total_needed'] = depletion_data['quantity'] * depletion_data['Quantity'] * depletion_data['conversion_factor']

        # Step 4: Group by ingredient and sum the total needed for each.
        final_depletion = depletion_data.groupby('ingredient_id')['total_needed'].sum().reset_index()

        # Step 5: Update the database with the new inventory levels.
        updates = []
        for index, row in final_depletion.iterrows():
            updates.append((row['total_needed'], row['ingredient_id']))
        
        c.executemany("UPDATE Ingredients SET current_inventory = current_inventory - ? WHERE ingredient_id = ?", updates)
        
        conn.commit()
        print("Inventory depletion completed successfully based on orders.")

    except Exception as e:
        print(f"An unexpected error occurred during inventory depletion: {e}")

# Call the functions at the end of your script
if __name__ == '__main__':
    # Assume your other functions create and populate the database first.
    # create_restaurant_tables()
    # insert_data_into_tables()
    
    conn = sqlite3.connect('restaurant.db')
    deplete_inventory_with_units(conn)
    conn.close()


# Call the functions to create tables and insert data
create_restaurant_tables()
insert_data_into_tables() # This now calls deplete_inventory_from_orders internally



# Open a single connection to the database
conn = sqlite3.connect('restaurant.db')

try:
    # Pass the open connection to the insertion function
    insert_data_into_tables(conn)

    # Pass the same open connection to the depletion function
    deplete_inventory_with_units(conn)

except sqlite3.Error as e:
    print(f"An error occurred: {e}")

finally:
    # Close the connection only once, after all operations are complete
    if conn:
        conn.close()
        print("\nDatabase connection closed.")
