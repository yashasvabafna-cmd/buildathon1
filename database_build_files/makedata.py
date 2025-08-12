import pandas as pd

df = pd.DataFrame({
    "id": list(range(1, 101)),
    "item_name": [
        "Classic Burger", "Cheese Burger", "Bacon Burger", "Veggie Burger", "Mushroom Swiss Burger",
        "BBQ Burger", "Spicy Chicken Burger", "Grilled Chicken Burger", "Double Patty Burger", "Black Bean Burger",
        "French Fries", "Curly Fries", "Sweet Potato Fries", "Cheese Fries", "Loaded Fries",
        "Garlic Fries", "Truffle Fries", "Waffle Fries", "Cajun Fries", "Chili Cheese Fries",
        "Caesar Salad", "Greek Salad", "Garden Salad", "Cobb Salad", "Spinach Salad",
        "Kale Salad", "Southwest Salad", "Quinoa Salad", "Caprese Salad", "Asian Sesame Salad",
        "Margherita Pizza", "Pepperoni Pizza", "BBQ Chicken Pizza", "Veggie Pizza", "Hawaiian Pizza",
        "Meat Lovers Pizza", "Four Cheese Pizza", "Mushroom Pizza", "White Pizza", "Buffalo Chicken Pizza",
        "Chicken Tenders", "Mozzarella Sticks", "Onion Rings", "Jalapeño Poppers", "Garlic Bread",
        "Spinach Artichoke Dip", "Stuffed Mushrooms", "Bruschetta", "Nachos", "Deviled Eggs",
        "Cola", "Diet Cola", "Lemonade", "Iced Tea", "Sweet Tea",
        "Orange Juice", "Apple Juice", "Bottled Water", "Sparkling Water", "Root Beer",
        "Chocolate Cake", "Cheesecake", "Brownie", "Ice Cream", "Apple Pie",
        "Banana Split", "Tiramisu", "Pecan Pie", "Chocolate Chip Cookie", "Strawberry Shortcake",
        "Mac & Cheese", "Baked Beans", "Mashed Potatoes", "Coleslaw", "Cornbread",
        "Potato Wedges", "Steamed Broccoli", "Grilled Corn", "Side Salad", "Fruit Cup",
        "Iced Coffee", "Hot Coffee", "Latte", "Espresso", "Cappuccino",
        "Milkshake - Vanilla", "Milkshake - Chocolate", "Milkshake - Strawberry", "Smoothie - Berry", "Smoothie - Mango",
        "Chicken Alfredo", "Beef Lasagna", "Grilled Salmon", "Steak Frites", "Vegetable Stir Fry",
        "Shrimp Scampi", "Tofu Bowl", "Pulled Pork Sandwich", "Philly Cheesesteak", "Chicken Parmesan"
    ],
    "price": [
        6.99, 7.49, 7.99, 6.49, 7.59,
        7.79, 7.29, 7.49, 8.49, 6.99,
        2.49, 2.99, 3.29, 3.49, 4.49,
        3.59, 4.99, 3.99, 3.79, 4.59,
        5.99, 6.49, 5.49, 6.99, 6.29,
        6.49, 6.79, 6.59, 5.99, 6.39,
        9.99, 10.49, 10.99, 9.49, 10.29,
        11.49, 10.19, 9.99, 10.69, 11.29,
        5.49, 4.99, 4.79, 5.29, 3.99,
        6.49, 5.99, 4.59, 6.29, 4.49,
        1.99, 1.99, 2.49, 2.29, 2.29,
        2.79, 2.79, 1.49, 1.99, 2.49,
        4.99, 5.49, 4.49, 3.99, 4.99,
        5.99, 6.29, 4.29, 2.99, 5.49,
        3.99, 2.99, 3.49, 2.49, 2.99,
        3.49, 3.29, 3.79, 3.99, 3.69,
        2.99, 2.49, 3.99, 2.29, 3.49,
        4.99, 5.49, 5.49, 4.99, 4.99,
        10.49, 11.29, 12.49, 13.99, 9.49,
        12.99, 9.99, 8.49, 10.99, 11.49
    ],
    "category": [
        "main", "main", "main", "main", "main",
        "main", "main", "main", "main", "main",
        "side", "side", "side", "side", "side",
        "side", "side", "side", "side", "side",
        "main", "main", "main", "main", "main",
        "main", "main", "main", "main", "main",
        "main", "main", "main", "main", "main",
        "main", "main", "main", "main", "main",
        "appetizer", "appetizer", "appetizer", "appetizer", "appetizer",
        "appetizer", "appetizer", "appetizer", "appetizer", "appetizer",
        "beverage", "beverage", "beverage", "beverage", "beverage",
        "beverage", "beverage", "beverage", "beverage", "beverage",
        "dessert", "dessert", "dessert", "dessert", "dessert",
        "dessert", "dessert", "dessert", "dessert", "dessert",
        "side", "side", "side", "side", "side",
        "side", "side", "side", "side", "side",
        "beverage", "beverage", "beverage", "beverage", "beverage",
        "beverage", "beverage", "beverage", "beverage", "beverage",
        "main", "main", "main", "main", "main",
        "main", "main", "main", "main", "main"
    ]
})

print(df.head())

df.to_csv('testmenu100.csv', index=False)