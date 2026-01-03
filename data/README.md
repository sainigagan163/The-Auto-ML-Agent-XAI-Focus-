# Data Directory

Place your dataset files in this folder. The pipeline expects a **CSV** file.

## Expected format

- **File type:** CSV (comma-separated values)
- **Header row:** Required
- **Target column:** Must exist and be named to match `config/default.yaml` (`target_column`).

## Example

```text
feature_1,feature_2,target
1.0,0.5,yes
2.0,0.1,no
```
