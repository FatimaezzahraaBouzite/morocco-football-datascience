## Morocco Football - Data science project
A complete data science project analysing the Morocco national football team (Atlas Lions) using historical match data, World Cup statistics, and player attributes. From their historic 1986 World Cup run to becoming the first African team to reach a World Cup semi-final in 2022, explored through data.

# Project goals
Explore Morocco's full match history and identify performance trends over time
Build a machine learning model to predict match outcomes (Win / Draw / Loss)
Analyze the 2022 FIFA World Cup campaign in depth using player-level stats
Generate insights on squad composition, coaching impact, and tournament performance
Deploy an interactive dashboard to predict results against any upcoming opponent

# Datasets used
DatasetSourceDescriptionInternational Football Results 1872–2026Kaggle47,000+ international match resultsFIFA World Cup 1930–2022KaggleAll World Cup matches, squads, lineupsFIFA World Cup 2022 Complete DatasetKagglePlayer stats, xG, shots, possession per matchEuropean Soccer DatabaseKaggleFIFA player attributes for Moroccan squad members

Datasets are not included in this repo due to file size. Download each one from Kaggle and place them in the data/ folder.

# Key analyses
* Win rate by decade - how Morocco's performance evolved from the 1970s to today
* Home vs away breakdown - goals scored, conceded, and clean sheet rates
* Tournament performance - friendlies vs AFCON vs World Cup qualifiers vs finals
* The Regragui effect - statistical comparison of results before and after coach Walid Regragui's appointment in September 2022
* 2022 World Cup deep dive - xG vs actual goals, defensive metrics, penalty shootout analysis
* Player clustering - grouping squad members by FIFA attributes to identify position archetypes
* Match outcome prediction - ELO-based XGBoost model to predict W/D/L for upcoming matches

# Machine learning model
- Task: Predict whether Morocco will Win, Draw, or Lose a given match
Features used:
* ELO rating difference between Morocco and opponent
* Home / away / neutral ground
* Recent form (last 5 matches)
* Tournament type
* Opponent historical strength

Models compared: Logistic Regression → Random Forest → XGBoost

# Sample insight
Morocco's win rate jumped from 41% in the 2010s to 58% in the 2020s — the highest of any decade in their history — coinciding with the rise of European-based players and the appointment of Walid Regragui.

# About the Atlas Lions
Morocco is Africa's most successful national team of the modern era. They made history at the 2022 FIFA World Cup in Qatar by becoming the first African and Arab nation to reach the semi-finals, defeating Spain and Portugal along the way. They are ranked 8th in the world as of 2026 and will co-host the 2030 FIFA World Cup alongside Spain and Portugal.

# Contact
Feel free to open an issue or reach out if you have suggestions, questions, or want to collaborate.
