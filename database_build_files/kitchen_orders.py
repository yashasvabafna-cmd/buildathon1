import pandas as pd
from Classes import Item, Order
from oldbots.testfile_IGNORE import result


    
order_number = getattr(result, 'order_id', 1)
df = pd.DataFrame({
    'order_id': [order_number for _ in result.items],
    'item_name': [item.item_name for item in result.items],
    'quantity': [item.quantity for item in result.items],
    'modifiers': [', '.join(item.modifiers) if item.modifiers else '' for item in result.items]
})



print('--- Kitchen Order Summary ---')
for idx, row in df.iterrows():
    print(f"Order #{row['order_id']}: {row['item_name']} | {row['quantity']} | Modifiers: {row['modifiers']}")

# Save Kitchen Order Summary as Order Table
order_table_file = 'order_table.csv'
df.to_csv(order_table_file, index=False)
print('\n--- Kitchen Order Table ---')
print(order_table_file)
# --- Bill Table ---
import numpy as np
import pandas as pd
menu = pd.read_csv('testmenu100.csv')

# Merge order items with menu to get prices
bill_df = pd.merge(df, menu[['item_name', 'price']], on='item_name', how='left')
bill_df['total'] = bill_df['quantity'] * bill_df['price']


# Save bill table to orders.csv
bill_table = bill_df[['order_id', 'item_name', 'quantity', 'price', 'total']]
import os
orders_file = 'orders.csv'
if not os.path.isfile(orders_file):
    bill_table.to_csv(orders_file, index=False)
else:
    bill_table.to_csv(orders_file, mode='a', header=False, index=False)

print('\n--- Bill Table ---')
print(bill_table)
