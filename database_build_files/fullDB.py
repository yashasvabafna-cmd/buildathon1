import pandas as pd
import random
import numpy as np
from IPython.display import display
# Re-creating the DataFrames as they were in the latest 'pandas_database_representation'
# This ensures the code is self-contained and runnable.

# Chef1 Table
chefs_data = {
    'chef_id': [101, 102, 103],
    'chef_name': ['Gordon Ramsay', 'Julia Child', 'Wolfgang Puck'],
    'salary': [90000.00, 75000.00, 82000.00]
}
df_chefs = pd.DataFrame(chefs_data)

# Waiter1 Table
waiters_data = {
    'waiter_id': [201, 202, 203],
    'waiter_name': ['Alice Smith', 'Bob Johnson', 'Charlie Brown'],
    'salary': [35000.00, 32000.00, 30000.00],
    'phone': [1234567890, 9876543210, 5551234567]
}
df_waiters = pd.DataFrame(waiters_data)

# Customer1 Table
customers_data = {
    'customer_id': [301, 302, 303],
    'customer_name': ['John Doe', 'Jane Miller', 'Peter Jones'],
    'address': ['123 Main St', '456 Oak Ave', '789 Pine Ln'],
    'phone': [1112223333, 4445556666, 7778889999],
    'Waiter_waiter_id': [201, 201, 202]
}
df_customers = pd.DataFrame(customers_data)

# Supplier1 Table
suppliers_data = {
    'supplier_id': [401, 402, 403],
    'supplier_city': ['New York', 'Los Angeles', 'Chicago'],
    'supplier_name': ['Fresh Foods Inc.', 'Organic Greens Co.', 'Meat Masters Ltd.']
}
df_suppliers = pd.DataFrame(suppliers_data)

# supplier_address Table
supplier_addresses_data = {
    'supplier_id': [401, 402, 403],
    'address': ['100 Broadway', '200 Sunset Blvd', '300 Wacker Dr']
}
df_supplier_addresses = pd.DataFrame(supplier_addresses_data)


# Meal1 Table (Expanded for Indian Restaurant Menu) - Replicated from previous immersive
starters_veg = ["Paneer Tikka", "Vegetable Samosa", "Onion Bhaji", "Hara Bhara Kebab",
    "Gobi Manchurian", "Crispy Corn", "Dahi Ke Kebab", "Aloo Tikki"]
starters_non_veg = ["Chicken Tikka", "Seekh Kebab", "Fish Amritsari", "Prawn Fry",
    "Mutton Shammi Kebab", "Chicken 65", "Chilli Chicken Dry"]
main_course_veg = ["Paneer Butter Masala", "Dal Makhani", "Malai Kofta", "Mix Vegetable Curry",
    "Chana Masala", "Aloo Gobi", "Palak Paneer", "Kadai Paneer",
    "Navratan Korma", "Baingan Bharta", "Veg Kofta Curry", "Mushroom Do Pyaza"]
main_course_non_veg = ["Butter Chicken", "Chicken Tikka Masala", "Mutton Rogan Josh", "Fish Curry",
    "Prawn Malai Curry", "Chicken Korma", "Laal Maas", "Goan Fish Curry",
    "Hydrabadi Chicken Biryani", "Mutton Biryani", "Chicken Chettinad"]
specialties_veg = ["Dum Aloo Kashmiri", "Bharwan Bhindi", "Sarson Ka Saag", "Dal Bukhara",
    "Veg Kolhapuri", "Shahi Paneer"]
specialties_non_veg = ["Nalli Nihari", "Raan Musallam", "Amritsari Kulcha with Chole (Non-Veg option)",
    "Kerala Fish Moilee", "Awadhi Mutton Korma"]
desserts = ["Gulab Jamun", "Rasgulla", "Gajar Halwa", "Moong Dal Halwa",
    "Kulfi (Pistachio)", "Jalebi with Rabri", "Phirni", "Shahi Tukda"]

all_meal_details = []
meal_id_counter = 1

def add_meals(meal_list, meal_type, category, count_per_item=1):
    global meal_id_counter
    for base_name in meal_list:
        for _ in range(count_per_item):
            meal_name = f"{base_name} {random.randint(1, 999)}"
            price = round(random.uniform(150.00, 800.00), 2)
            chef_id = random.choice(df_chefs['chef_id'].tolist())
            all_meal_details.append({
                'meal_id': meal_id_counter,
                'name': meal_name,
                'type': meal_type,
                'category': category,
                'price': price,
                'Chef_chef_id': chef_id
            })
            meal_id_counter += 1

add_meals(starters_veg, 'veg', 'Starter', count_per_item=1)
add_meals(starters_non_veg, 'non-veg', 'Starter', count_per_item=1)
add_meals(main_course_veg, 'veg', 'Main Course', count_per_item=3)
add_meals(main_course_non_veg, 'non-veg', 'Main Course', count_per_item=2)
add_meals(specialties_veg, 'veg', 'Specialty', count_per_item=1)
add_meals(specialties_non_veg, 'non-veg', 'Specialty', count_per_item=1)
add_meals(desserts, 'veg', 'Dessert', count_per_item=2)

df_meals = pd.DataFrame(all_meal_details)


# -----------------------------------------------------
# Ingredient1 Table (Expanded for Indian Cuisine with detailed descriptions)
# -----------------------------------------------------
# Dictionary mapping dishes to realistic ingredients
dish_ingredients = {
    "Paneer Tikka": ["Paneer", "Yogurt", "Ginger", "Garlic", "Red Chilli Powder", "Turmeric Powder", "Garam Masala", "Mustard Oil", "Bell Peppers", "Onions", "Tomatoes"],
    "Vegetable Samosa": ["Potatoes", "Peas", "Onions", "Ginger", "Green Chillies", "Cumin Seeds", "Coriander Powder", "Turmeric Powder", "Garam Masala", "Salt", "Wheat Flour", "Oil"],
    "Onion Bhaji": ["Onions", "Besan (Gram Flour)", "Rice Flour", "Ginger", "Green Chillies", "Cumin Seeds", "Coriander Powder", "Turmeric Powder", "Red Chilli Powder", "Salt", "Oil"],
    "Hara Bhara Kebab": ["Spinach", "Peas", "Potatoes", "Paneer", "Ginger", "Green Chillies", "Garam Masala", "Salt", "Bread Crumbs", "Oil"],
    "Gobi Manchurian": ["Cauliflower", "Maida", "Cornflour", "Ginger", "Garlic", "Green Chillies", "Onions", "Bell Peppers", "Soy Sauce", "Vinegar", "Sugar", "Salt", "Oil"],
    "Crispy Corn": ["Corn", "Maida", "Cornflour", "Ginger", "Garlic", "Green Chillies", "Onions", "Bell Peppers", "Salt", "Pepper", "Oil"],
    "Dahi Ke Kebab": ["Curd (Yogurt)", "Paneer", "Besan (Gram Flour)", "Onions", "Ginger", "Green Chillies", "Coriander Powder", "Garam Masala", "Salt", "Oil"],
    "Aloo Tikki": ["Potatoes", "Peas", "Coriander Powder", "Cumin Seeds", "Garam Masala", "Salt", "Oil"],
    "Chicken Tikka": ["Chicken (Boneless)", "Yogurt", "Ginger", "Garlic", "Red Chilli Powder", "Turmeric Powder", "Garam Masala", "Mustard Oil", "Salt"],
    "Seekh Kebab": ["Minced Meat (Keema)", "Onions", "Ginger", "Garlic", "Green Chillies", "Coriander Powder", "Cumin Seeds", "Garam Masala", "Salt", "Besan (Gram Flour)"],
    "Fish Amritsari": ["Fish Fillet", "Besan (Gram Flour)", "Rice Flour", "Ginger", "Garlic", "Green Chillies", "Coriander Powder", "Cumin Seeds", "Ajwain", "Red Chilli Powder", "Turmeric Powder", "Salt", "Lemon", "Oil"],
    "Prawn Fry": ["Prawns", "Ginger", "Garlic", "Red Chilli Powder", "Turmeric Powder", "Salt", "Lemon", "Oil"],
    "Mutton Shammi Kebab": ["Mutton", "Chana Dal", "Onions", "Ginger", "Garlic", "Green Chillies", "Coriander Powder", "Cumin Seeds", "Garam Masala", "Salt", "Eggs"],
    "Chicken 65": ["Chicken (Boneless)", "Yogurt", "Ginger", "Garlic", "Red Chilli Powder", "Turmeric Powder", "Curry Leaves", "Mustard Seeds", "Green Chillies", "Oil"],
    "Chilli Chicken Dry": ["Chicken (Boneless)", "Maida", "Cornflour", "Ginger", "Garlic", "Green Chillies", "Onions", "Bell Peppers", "Soy Sauce", "Vinegar", "Sugar", "Salt", "Oil"],
    "Paneer Butter Masala": ["Paneer", "Tomatoes", "Onions", "Cashews", "Cream", "Butter", "Ginger", "Garlic", "Cumin Seeds", "Coriander Powder", "Turmeric Powder", "Red Chilli Powder", "Garam Masala", "Salt", "Sugar", "Kasuri Methi"],
    "Dal Makhani": ["Urad Dal", "Rajma", "Butter", "Cream", "Tomatoes", "Onions", "Ginger", "Garlic", "Green Chillies", "Cumin Seeds", "Coriander Powder", "Turmeric Powder", "Red Chilli Powder", "Garam Masala", "Salt"],
    "Malai Kofta": ["Paneer", "Potatoes", "Cashews", "Cream", "Onions", "Tomatoes", "Ginger", "Garlic", "Cumin Seeds", "Coriander Powder", "Turmeric Powder", "Red Chilli Powder", "Garam Masala", "Salt", "Sugar"],
    "Mix Vegetable Curry": ["Mixed Vegetables", "Onions", "Tomatoes", "Ginger", "Garlic", "Green Chillies", "Cumin Seeds", "Coriander Powder", "Turmeric Powder", "Red Chilli Powder", "Garam Masala", "Salt", "Oil"],
    "Chana Masala": ["Chickpeas", "Onions", "Tomatoes", "Ginger", "Garlic", "Green Chillies", "Cumin Seeds", "Coriander Powder", "Turmeric Powder", "Red Chilli Powder", "Garam Masala", "Amchur Powder", "Salt", "Oil"],
    "Aloo Gobi": ["Potatoes", "Cauliflower", "Onions", "Tomatoes", "Ginger", "Garlic", "Green Chillies", "Cumin Seeds", "Coriander Powder", "Turmeric Powder", "Red Chilli Powder", "Salt", "Oil"],
    "Palak Paneer": ["Spinach", "Paneer", "Onions", "Tomatoes", "Ginger", "Garlic", "Green Chillies", "Cumin Seeds", "Coriander Powder", "Turmeric Powder", "Garam Masala", "Salt", "Cream"],
    "Kadai Paneer": ["Paneer", "Bell Peppers", "Onions", "Tomatoes", "Ginger", "Garlic", "Green Chillies", "Coriander Seeds", "Cumin Seeds", "Red Chilli Powder", "Salt", "Oil"],
    "Navratan Korma": ["Mixed Vegetables", "Paneer", "Dry Fruits", "Cream", "Cashews", "Onions", "Ginger", "Garlic", "Cumin Seeds", "Coriander Powder", "Garam Masala", "Salt", "Sugar"],
    "Baingan Bharta": ["Baingan", "Onions", "Tomatoes", "Ginger", "Garlic", "Green Chillies", "Cumin Seeds", "Coriander Powder", "Turmeric Powder", "Red Chilli Powder", "Salt", "Mustard Oil"],
    "Veg Kofta Curry": ["Mixed Vegetables", "Besan (Gram Flour)", "Onions", "Ginger", "Garlic", "Green Chillies", "Tomatoes", "Cream", "Cumin Seeds", "Coriander Powder", "Garam Masala", "Salt", "Oil"],
    "Mushroom Do Pyaza": ["Mushrooms", "Onions", "Tomatoes", "Ginger", "Garlic", "Green Chillies", "Cumin Seeds", "Coriander Powder", "Turmeric Powder", "Red Chilli Powder", "Garam Masala", "Salt", "Oil"],
    "Butter Chicken": ["Chicken (with Bone)", "Tomatoes", "Cashews", "Cream", "Butter", "Ginger", "Garlic", "Red Chilli Powder", "Turmeric Powder", "Garam Masala", "Salt", "Sugar", "Kasuri Methi"],
    "Chicken Tikka Masala": ["Chicken (Boneless)", "Yogurt", "Ginger", "Garlic", "Red Chilli Powder", "Turmeric Powder", "Garam Masala", "Onions", "Tomatoes", "Cream", "Salt"],
    "Mutton Rogan Josh": ["Mutton", "Yogurt", "Ginger", "Garlic", "Kashmiri Chilli Powder", "Fennel Seeds", "Ginger Powder", "Salt", "Oil"],
    "Fish Curry": ["Fish Fillet", "Onions", "Tomatoes", "Ginger", "Garlic", "Green Chillies", "Cumin Seeds", "Coriander Powder", "Turmeric Powder", "Red Chilli Powder", "Garam Masala", "Coconut Milk", "Tamarind", "Salt", "Oil"],
    "Prawn Malai Curry": ["Prawns", "Coconut Milk", "Onions", "Ginger", "Garlic", "Green Chillies", "Cumin Seeds", "Coriander Powder", "Turmeric Powder", "Salt", "Garam Masala", "Bay Leaf", "Cardamom", "Cloves", "Cinnamon"],
    "Chicken Korma": ["Chicken (with Bone)", "Yogurt", "Onions", "Ginger", "Garlic", "Cashews", "Cream", "Garam Masala", "Coriander Powder", "Cumin Seeds", "Turmeric Powder", "Salt", "Oil"],
    "Laal Maas": ["Mutton", "Mathania Chillies", "Onions", "Garlic", "Ginger", "Yogurt", "Ghee", "Cumin Seeds", "Coriander Powder", "Turmeric Powder", "Garam Masala", "Salt"],
    "Goan Fish Curry": ["Fish Fillet", "Coconut", "Red Chillies", "Turmeric Powder", "Coriander Seeds", "Cumin Seeds", "Fenugreek Seeds", "Tamarind", "Kokum", "Salt", "Oil"],
    "Hydrabadi Chicken Biryani": ["Chicken (with Bone)", "Basmati Rice", "Yogurt", "Onions", "Tomatoes", "Ginger", "Garlic", "Green Chillies", "Mint", "Fresh Coriander", "Garam Masala", "Biryani Masala", "Saffron", "Ghee", "Oil", "Salt"],
    "Mutton Biryani": ["Mutton", "Basmati Rice", "Yogurt", "Onions", "Tomatoes", "Ginger", "Garlic", "Green Chillies", "Mint", "Fresh Coriander", "Garam Masala", "Biryani Masala", "Saffron", "Ghee", "Oil", "Salt"],
    "Chicken Chettinad": ["Chicken (with Bone)", "Onions", "Tomatoes", "Ginger", "Garlic", "Green Chillies", "Curry Leaves", "Red Chillies", "Coriander Seeds", "Cumin Seeds", "Fennel Seeds", "Black Pepper Corns", "Cloves", "Cardamom", "Cinnamon", "Star Anise", "Coconut", "Salt", "Oil"],
    "Dum Aloo Kashmiri": ["Potatoes", "Yogurt", "Ginger Powder", "Fennel Powder", "Kashmiri Chilli Powder", "Garam Masala", "Salt", "Oil"],
    "Bharwan Bhindi": ["Bhindi", "Besan (Gram Flour)", "Coriander Powder", "Cumin Powder", "Turmeric Powder", "Red Chilli Powder", "Amchur Powder", "Garam Masala", "Salt", "Oil"],
    "Sarson Ka Saag": ["Mustard Greens", "Spinach", "Bathua", "Ginger", "Garlic", "Green Chillies", "Maize Flour", "Ghee", "Salt"],
    "Dal Bukhara": ["Whole Black Lentils", "Butter", "Cream", "Tomatoes", "Ginger", "Garlic", "Salt"],
    "Veg Kolhapuri": ["Mixed Vegetables", "Onions", "Tomatoes", "Ginger", "Garlic", "Dry Coconut", "Sesame Seeds", "Coriander Seeds", "Cumin Seeds", "Red Chillies", "Garam Masala", "Salt", "Oil"],
    "Shahi Paneer": ["Paneer", "Cashews", "Cream", "Tomatoes", "Onions", "Ginger", "Garlic", "Garam Masala", "Cardamom", "Salt", "Sugar", "Ghee"],
    "Nalli Nihari": ["Lamb Shanks", "Onions", "Ginger", "Garlic", "Wheat Flour", "Nihari Masala", "Ghee", "Salt"],
    "Raan Musallam": ["Lamb Leg", "Yogurt", "Ginger", "Garlic", "Cashews", "Onions", "Tomatoes", "Garam Masala", "Red Chilli Powder", "Turmeric Powder", "Salt", "Oil"],
    "Amritsari Kulcha with Chole (Non-Veg option)": ["Wheat Flour", "Yogurt", "Butter", "Potatoes", "Onions", "Chickpeas", "Tomatoes", "Ginger", "Garlic", "Green Chillies", "Cumin Seeds", "Coriander Powder", "Turmeric Powder", "Red Chilli Powder", "Garam Masala", "Amchur Powder", "Salt", "Oil", "Chicken (optional)"],
    "Kerala Fish Moilee": ["Fish Fillet", "Coconut Milk", "Onions", "Ginger", "Garlic", "Green Chillies", "Curry Leaves", "Mustard Seeds", "Turmeric Powder", "Salt", "Oil"],
    "Awadhi Mutton Korma": ["Mutton", "Yogurt", "Onions", "Ginger", "Garlic", "Cashews", "Cream", "Garam Masala", "Coriander Powder", "Cumin Seeds", "Turmeric Powder", "Salt", "Oil", "Kewra Water"],
    "Gulab Jamun": ["Khoya", "Maida", "Sugar", "Cardamom", "Ghee", "Oil"],
    "Rasgulla": ["Chenna", "Sugar", "Cardamom"],
    "Gajar Halwa": ["Carrots", "Milk", "Sugar", "Ghee", "Cardamom", "Cashews", "Almonds"],
    "Moong Dal Halwa": ["Moong Dal", "Ghee", "Sugar", "Milk", "Cardamom", "Cashews", "Almonds"],
    "Kulfi (Pistachio)": ["Milk", "Sugar", "Pistachios", "Cardamom"],
    "Jalebi with Rabri": ["Maida", "Yogurt", "Sugar", "Cardamom", "Saffron", "Milk"],
    "Phirni": ["Rice Flour", "Milk", "Sugar", "Cardamom", "Saffron", "Almonds", "Pistachios"],
    "Shahi Tukda": ["Bread", "Milk", "Sugar", "Cardamom", "Saffron", "Ghee", "Cashews", "Almonds"]
}


all_ingredients_data = []
ingredient_id_counter = 5001

# Populate df_ingredients based on dish_ingredients
for index, meal_row in df_meals.iterrows():
    meal_id = meal_row['meal_id']
    base_meal_name = meal_row['name'].split(" ")[0]

    realistic_ingredients = dish_ingredients.get(base_meal_name, [])

    ingredient_descriptions = {
        "Paneer": "Cubed", "Yogurt": "Plain", "Ginger": "Grated", "Garlic": "Minced",
        "Red Chilli Powder": "Spicy", "Turmeric Powder": "Pure", "Garam Masala": "Homemade",
        "Mustard Oil": "Refined", "Bell Peppers": "Diced", "Onions": "Finely chopped",
        "Tomatoes": "Pureed", "Potatoes": "Boiled", "Peas": "Fresh", "Cumin Seeds": "Whole",
        "Coriander Powder": "Ground", "Salt": "Table", "Wheat Flour": "All-purpose", "Oil": "Refined",
        "Besan (Gram Flour)": "Fine", "Rice Flour": "Fine", "Spinach": "Chopped", "Bread Crumbs": "Dry",
        "Cauliflower": "Florets", "Maida": "All-purpose", "Cornflour": "Fine", "Soy Sauce": "Light",
        "Vinegar": "White", "Sugar": "Granulated", "Pepper": "Ground", "Curd (Yogurt)": "Whisked",
        "Chicken (Boneless)": "Cubed", "Minced Meat (Keema)": "Lamb", "Fish Fillet": "Cubed",
        "Prawns": "Shelled and deveined", "Mutton": "Curry cut", "Chana Dal": "Soaked",
        "Eggs": "Beaten", "Curry Leaves": "Fresh", "Mustard Seeds": "Whole", "Butter": "Unsalted",
        "Cream": "Fresh", "Cashews": "Whole", "Kasuri Methi": "Dried leaves", "Urad Dal": "Whole",
        "Rajma": "Soaked", "Mixed Vegetables": "Chopped", "Amchur Powder": "Dry Mango",
        "Baingan": "Roasted and mashed", "Dry Fruits": "Mixed", "Coriander Seeds": "Roasted and ground",
        "Dry Coconut": "Grated", "Sesame Seeds": "Roasted", "Ajwain": "Whole", "Rice": "Cooked",
        "Almonds": "Sliced", "Khoya": "Unsweetened", "Ghee": "Melted", "Chenna": "Freshly made",
        "Milk": "Full cream", "Pistachios": "Chopped", "Saffron": "Strands", "Bhindi": "Washed and dried",
        "Bhindi": "Washed and dried", "Mustard Greens": "Chopped", "Bathua": "Chopped", "Maize Flour": "Fine",
        "Lamb Shanks": "Bone-in", "Nihari Masala": "Store-bought", "Lamb Leg": "Bone-in",
        "Kashmiri Chilli Powder": "Mild", "Fennel Powder": "Ground", "Ginger Powder": "Dry",
        "Mathania Chillies": "Dried", "Black Pepper Corns": "Whole", "Cloves": "Whole",
        "Cardamom": "Green pods", "Cinnamon": "Stick", "Star Anise": "Whole", "Kewra Water": "Essence",
        "Kokum": "Dried", "Biryani Masala": "Store-bought"
        # Add more ingredient descriptions as needed
    }

    for ingredient_name in realistic_ingredients:
        description = ingredient_descriptions.get(ingredient_name, "Fresh")
        all_ingredients_data.append({
            'ingredient_id': ingredient_id_counter,
            'ingredient_name': ingredient_name,
            'description': description,
            'Meal_meal_id': meal_id  # Ensure meal_id is correctly assigned
        })
        ingredient_id_counter += 1

df_ingredients = pd.DataFrame(all_ingredients_data)
print("Chefs Table:")
display(df_chefs.head())

print("\nWaiters Table:")
display(df_waiters.head())

print("\nCustomers Table:")
display(df_customers.head())

print("\nSuppliers Table:")
display(df_suppliers.head())

print("\nSupplier Addresses Table:")
display(df_supplier_addresses.head())

print("\nMeals Table:")
display(df_meals.head())

print("\nIngredients Table:")
display(df_ingredients.head())
df_chefs.to_csv('chefs.csv', index=False)
df_waiters.to_csv('waiters.csv', index=False)
df_customers.to_csv('customers.csv', index=False)
df_suppliers.to_csv('suppliers.csv', index=False)
df_supplier_addresses.to_csv('supplier_addresses.csv', index=False)
df_meals.to_csv('meals.csv', index=False)
df_ingredients.to_csv('ingredients.csv', index=False)

