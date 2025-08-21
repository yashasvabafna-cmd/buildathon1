from typing import List
from pydantic import BaseModel
import numpy as np
import pandas as pd

class Item(BaseModel):
    item_name: str
    quantity: int
    modifiers: List[str] = []

class Order(BaseModel):
    items: List[Item]
    delete: List[Item]

class OrderUpdate(BaseModel):
    def count_ordered_items(self, order: Order, menu: pd.DataFrame) -> pd.DataFrame:
        """
        Count how many orders of each menu item are asked for in the given order.
        Returns a DataFrame with menu items, total quantities ordered, and separate columns for each modifier.
        """
        menu_names = menu['item_name'].str.lower().tolist()
        # Collect all unique modifiers
        unique_modifiers = set()
        for item in order.items:
            for mod in item.modifiers:
                unique_modifiers.add(mod.lower())
        modifier_list = sorted(list(unique_modifiers))

        # Prepare count structure
        counts = {name: {'total_ordered': 0, **{mod: 0 for mod in modifier_list}} for name in menu_names}

        for item in order.items:
            item_name = item.item_name.lower()
            if item_name in counts:
                counts[item_name]['total_ordered'] += item.quantity
                # Count each modifier for this item
                for mod in item.modifiers:
                    mod_lower = mod.lower()
                    if mod_lower in modifier_list:
                        counts[item_name][mod_lower] += item.quantity

        # Build DataFrame
        data = {'item_name': [], 'total_ordered': []}
        for mod in modifier_list:
            data[mod] = []
        for name in menu_names:
            data['item_name'].append(name)
            data['total_ordered'].append(counts[name]['total_ordered'])
            for mod in modifier_list:
                data[mod].append(counts[name][mod])
        return pd.DataFrame(data)

