# this file is for Association mining using the Apriori algorithm

from typing import Dict, Any
import pandas as pd

# this function runs the association rule mining using the Apriori algorithm
# it converts a tabular dataset into a transactional one-hot encoded format. then it extracts frequent itemsets based on support and confidence thresholds
def run_association(
    df: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.3,
) -> Dict[str, Any]:

    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except Exception as e:
        raise RuntimeError(
            "mlxtend is required for association rules. Install with: pip install mlxtend"
        ) from e

    # we are converting each row into item tokens and then one hot encoding 
    tokens = []
    for idx, row in df.iterrows():
        items = []
        for c, v in row.items():
            if pd.isna(v):
                continue
            items.append(f"{c}={str(v)}")
        tokens.append(items)

    all_items = sorted({it for row in tokens for it in row})
    if len(all_items) == 0:
        raise ValueError("No valid items found for association mining (data may be empty).")

    onehot = pd.DataFrame(False, index=range(len(tokens)), columns=all_items)
    for i, items in enumerate(tokens):
        for it in items:
            onehot.at[i, it] = True

    itemsets = apriori(onehot, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        return {
            "params": {"task": "association", "algo": "apriori", "min_support": min_support, "min_confidence": min_confidence},
            "metrics": {"n_itemsets": 0.0, "n_rules": 0.0},
            "itemsets": itemsets,
            "rules": pd.DataFrame(),
        }

    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)   #we have used confidence as main filtering 
 
    keep_cols = [c for c in ["antecedents", "consequents", "support", "confidence", "lift"] if c in rules.columns]
    rules = rules[keep_cols].sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)

    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(list(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(list(s)))

    itemsets = itemsets.sort_values("support", ascending=False).reset_index(drop=True)

    return {
        "params": {"task": "association", "algo": "apriori", "min_support": min_support, "min_confidence": min_confidence},
        "metrics": {"n_itemsets": float(len(itemsets)), "n_rules": float(len(rules))},
        "itemsets": itemsets,
        "rules": rules,
    }
