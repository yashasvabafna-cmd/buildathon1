import pandas as pd 
from oldbots.testfile_IGNORE import Item
from oldbots.testfile_IGNORE import OrderUpdate
order_data={
    'order_id': [1, 2, 3],
    'item_name': ['burger', 'fries', 'soda'],
    'quantity': [2, 1, 3],
    'modifiers': [['no cheese'], [], ['extra ice']]}
order_df = pd.DataFrame(order_data)

