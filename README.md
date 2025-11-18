# Energy-Vs-Air-Quality-Heatmaps

# Link to Research Document:
https://docs.google.com/document/d/1J-q5261wliOcfurpZg2qW-AIVwDNmbao7ngpFIbH3OQ/edit?usp=sharing

## Forecasting

Pass the `--forecast-year-offset` flag (for example `--forecast-year-offset 5`) when running
`src/entry_point.py` to generate a trend-based projection of the selected metric that many
years beyond the visualization year. The regression now leverages monthly aggregates
when available, giving the models many more data points even if you only have a single
calendar year of raw data. The generated CSV/visuals are written next to the current-year
artifacts inside the `visualizations/` directory.

