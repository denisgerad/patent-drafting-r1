# Domain Reference Questions

These files contain expert-validated reference questions used to rate
the model's generated questions before sending for domain review.

## File naming convention
  {product_type_key}.json  — matches the key in product_type_checklists.json

## How to add a new domain's reference questions
1. Create a new JSON file named after the product type key
2. Structure it as shown in the existing files:
   - "product_type": matching key from product_type_checklists.json
   - "domain": display name
   - "categories": list of category objects, each with:
     - "name": must exactly match a category name in the checklist
     - "reference_questions": list of expert-validated questions
     - "key_terms": list of technical terms that MUST appear in coverage
     - "depth_markers": list of specific depth indicators required
   - "readiness_threshold": minimum overall score (0.0-1.0) to pass
   - "category_threshold": minimum per-category score to pass

## Getting reference questions for other domains
Ask domain experts to answer: "What questions would you always ask
an inventor of this type of patent before accepting it for drafting?"
Those answers become the reference_questions entries.
