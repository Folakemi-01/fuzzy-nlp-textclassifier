import pandas as pd
from mlxtend.evaluate import mcnemar_table, mcnemar


INPUT_FILE = 'mcnemar_input.csv'

# 1. Loading the data from the CSV file
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"ERROR: The file '{INPUT_FILE}' was not found.")
    print("Please create it according to the instructions.")
    exit()

# 2. Creating the contingency table
tb = mcnemar_table(y_target=df['true_label'],
                   y_model1=df['baseline_prediction'],
                   y_model2=df['fuzzy_prediction'])

print("McNemar's Contingency Table:")
print(tb)

# 3. Perform the test and get the results
chi2, p = mcnemar(ary=tb, corrected=True)

print("\n--- McNemar's Test Results ---")
print(f"Chi-squared statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")

# 4. Interpret the results
alpha = 0.05
print(f"\nAlpha level: {alpha}")
if p > alpha:
    print("Interpretation: The p-value is greater than alpha.")
    print("We fail to reject the null hypothesis; there is NO statistically significant difference between the models.")
else:
    print("Interpretation: The p-value is less than or equal to alpha.")
    print("We reject the null hypothesis; there IS a statistically significant difference between the models.")