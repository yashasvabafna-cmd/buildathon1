import pandas as pd
orders = pd.read_csv('order_table.csv', dtype=pd.StringDtype)
ingredients = pd.read_csv('menu_100_items_with_ingredients.csv', dtype=pd.StringDtype)
# print(ingredients)
# Merge on item_name
merged = pd.merge(orders, ingredients, on='item_name', how='left')

# Save to CSV
merged.to_csv('kitchen_ingredients_orders.csv', index=False)

# Print merged DataFrame
# print(orders.dtypes)
# print(ingredients.dtypes)
print(merged)