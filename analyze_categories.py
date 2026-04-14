import json
from collections import defaultdict

with open("eval/results/bge_m3_topk3_enriched.json") as f:
    results = json.load(f)

category_stats = defaultdict(lambda: {"hits": 0, "total": 0})
for row in results["per_row"]:
    cat = row["question"]
    category_stats[cat]["total"] += 1
    if row["hit_at_3"]:
        category_stats[cat]["hits"] += 1

# Sort by recall ascending — worst categories first
sorted_cats = sorted(
    category_stats.items(),
    key=lambda x: x[1]["hits"] / x[1]["total"]
)

for cat, stats in sorted_cats:
    recall = stats["hits"] / stats["total"]
    print(f"{recall:.0%}  {stats['hits']}/{stats['total']}  {cat}")