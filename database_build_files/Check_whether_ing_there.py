import pandas as pd
import re
import io

df = pd.read_csv(('kitchen_ingredients_orders.csv'), dtype=pd.StringDtype())

# Function to break down ingredients into individual unique words
def breakdown_ingredients(ingredients_str):
    if pd.isna(ingredients_str):
        return []
    # Convert to lowercase and find all word characters, then return unique words
    words = re.findall(r'\b\w+\b', ingredients_str.lower())
    return list(set(words))

# Apply the function to the 'ingredients' column
df['ingredient_words'] = df['ingredients'].apply(breakdown_ingredients)

# Combine all lists of words into a single set to get all unique words
all_unique_words = set()
for word_list in df['ingredient_words']:
    all_unique_words.update(word_list)

# Convert the set to a sorted list for a clean, ordered output
sorted_unique_words = sorted(list(all_unique_words))
sorted_unique_words = pd.DataFrame(sorted_unique_words, columns=['unique_words'])
print("Breakdown of all unique words in the 'ingredients' column:")
sorted_unique_words.to_csv('inventory.csv', index=False)

merged = pd.merge(orders, ingredients, on='item_name', how='left')
print(sorted_unique_words)
