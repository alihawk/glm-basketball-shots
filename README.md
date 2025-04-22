# GLM Basketball Shots

**Generalized Linear Models on Basketball Shot Data**  
Implements Multinomial & Ordinal Logistic Regression, plus bootstrapping for uncertainty, demonstrated on a real basketball‐shot dataset and a synthetic DGP.

## Repository
`https://github.com/your‑username/GLM‑BasketballShots.git`

## Requirements
- Python 3.8+
- `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit‑learn`

Install with:
```bash
pip install -r requirements.txt


Project Structure

.
├── solution.py         # Part 1 implementations & tests
├── part2.py            # Part 2 application & bootstrapping
├── test_lr.py          # Unit tests for Part 1
├── dataset.csv         # Basketball shots data
├── multinominal_coefficients.svg
├── README.md
└── .gitignore
Usage
Run the unit‑tests (Part 1):


python test_lr.py
Apply to real & synthetic data (Part 2):


python part2.py
Open multinomial_coefficients.svg to see coefficient plots.
