# UI Verification Guide (Synthetic Data)

This guide explains how to verify a transaction in the UI using the synthetic dataset.

## Prerequisites

1. Generate the synthetic dataset:
   - Run: python train_model.py
   - This creates models/synthetic_transactions.csv
2. Start the backend API:
   - Run: python simple_backend_api.py
   - The UI uses the endpoint /api/synthetic-verify
3. Open the UI:
   - Use the existing frontend entry point (index.html) or your dev server.

## What The UI Checks

The UI sends the following fields to the backend for an exact match in the synthetic dataset:

- Bank Book Name
- Transaction ID
- Amount

A transaction is considered verified only if all three fields match a row in the synthetic CSV.

## How To Verify In The UI

1. Open the Verify section in the UI.
2. Enter values that exist in the synthetic dataset:
   - Bank Book Name: one of the generated bank names
   - Transaction ID: a TXN id from the dataset
   - Amount: the exact amount from the dataset
3. Click Verify Transaction.

## Expected Results

- If a match is found:
  - Status: VERIFIED
  - Message: Transaction Successful: Details Verified against synthetic dataset.
- If no match is found:
  - Status: FAILED
  - Message: Transaction Failed: No matching record found in synthetic dataset.

## Quick Dataset Preview

Use the API to see sample synthetic rows:

- GET http://localhost:8000/api/synthetic-samples?limit=10

You can copy a row's values into the UI to verify successfully.

## Sample Synthetic Rows (From Latest Dataset)

Use any of these rows in the UI verify form (Bank Book Name, Transaction ID, Amount):

| Bank Book Name | Transaction ID | Amount |
| --- | --- | --- |
| KOTAK SAVINGS | TXN4404572 | 6266 |
| KOTAK SAVINGS | TXN-3444769 | 0 |
| HDFC SAVINGS | TXN4821339 | 65726 |
| ICICI CURRENT | TXN5599720 | 10628 |
| AXIS SAVINGS | TXN5097895 | 80078 |
| AXIS SAVINGS | TXN3684702 | 71212 |
| AXIS SAVINGS | TXN4874183 | 24301 |
| SBI SAVINGS | TXN5273036 | 40775 |
| ICICI CURRENT | TXN2209818 | 26855 |
| KOTAK SAVINGS | TXN3702279 | 70593 |

## Troubleshooting

- 404 from /api/synthetic-verify:
  - Run python train_model.py to generate models/synthetic_transactions.csv
- Verification always fails:
  - Ensure the values exactly match a row in the dataset
  - Bank Book Name and Transaction ID are case-insensitive, but must match exactly
