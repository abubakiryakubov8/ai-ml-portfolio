# AI & ML Portfolio

Projects by **Baki**, Computer Engineering student at Gachon University.
Python-based machine learning, forecasting, and data-structure work, each with its own README, tests, and run instructions.

| Project | What it shows | Stack |
|---|---|---|
| [ad-sales-prediction](ad-sales-prediction/) | End-to-end ML pipeline: data prep, training, evaluation, reporting | pandas, scikit-learn |
| [demand-forecasting](demand-forecasting/) | Time-series forecasting to automated reorder plan, with error handling and tests | Prophet, pandas |
| [hashmap-search-engine](hashmap-search-engine/) | Custom hash map and TF-IDF search engine built from scratch | Pure Python |
| [bigdata-coursework](bigdata-coursework/) | Deep learning / NLP coursework (CNN, RNN, Korean NLP) | see folder |

## Quick start

```bash
git clone https://github.com/abubakiryakubov8/ai-ml-portfolio.git
cd ai-ml-portfolio/<project-folder>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then follow the README inside the project folder.

## Testing

Continuous integration (GitHub Actions) runs the unit tests and the ML pipeline on every push. See `.github/workflows/ci.yml`.

## Contact

GitHub: [abubakiryakubov8](https://github.com/abubakiryakubov8)
