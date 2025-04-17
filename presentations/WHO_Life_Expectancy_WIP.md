# WHO Life Expectancy Data Story
## Work-in-Progress Report

**NAME:** [Your Name]  
**SECTION:** [Your Section]  
**TITLE:** Global Health Disparities: The Gap Between Developed and Developing Nations

## Section 1: Dataset and Story

### Dataset Description
This project uses the WHO Life Expectancy dataset, which contains health, economic, and social factors affecting life expectancy across 193 countries from 2000 to 2015. The dataset includes approximately 2,938 observations (193 countries × 15 years) with 22 columns of variables. These variables include life expectancy, country status (developed vs. developing), GDP, education, alcohol consumption, health expenditure, immunization coverage, and various disease prevalence rates. The dataset was sourced from the World Health Organization's Global Health Observatory data repository.

### Story
This data story is a **quest** to understand what factors create the significant gap in life expectancy between developed and developing nations. It examines how economic prosperity (GDP), healthcare investment, and preventive measures (immunization) interact to shape life expectancy outcomes globally. 

This is a quest story because it follows the journey of identifying the magnitude of global health disparities and seeks to uncover the underlying causes. The central visualization (bubble plot) shows how three critical factors (immunization, GDP, and life expectancy) interact, revealing patterns that aren't visible when examining each factor in isolation. Supporting visualizations further break down these relationships, showing how developed nations consistently outperform developing ones across key health indicators. The story concludes with evidence that investment in basic public health measures like immunization can significantly improve life expectancy even in countries with lower GDP.

## Section 2: Poster Layout Sketch

```
+----------------------------------------------------------------------+
|                                                                      |
|  [TITLE] Global Health Disparities: The Gap Between                  |
|          Developed and Developing Nations                            |
|                                                                      |
|  [INTRO TEXT]                     [CENTRAL VISUALIZATION]            |
|  Brief overview of               Bubble Plot showing                 |
|  global health                  Immunization vs Life                 |
|  disparities and                 Expectancy with GDP                 |
|  the importance of               as bubble size and                  |
|  understanding the               colored by country                  |
|  factors that create             status                              |
|  these gaps                                                          |
|                                                                      |
|    ↓                                     ↓                           |
|                                                                      |
|  [SUPPORT VIZ 1]                 [SUPPORT VIZ 2]     [SUPPORT VIZ 3] |
|  Histogram of Life              Bar Chart of Avg.    Scatter Plot of |
|  Expectancy                    Life Expectancy       GDP vs Life     |
|  Distribution                  by Country Status     Expectancy with |
|                                                      Health          |
|                                                      Expenditure     |
|    ↓                                     ↓                ↓          |
|                                                                      |
|  [KEY INSIGHTS]                                     [WOW ELEMENT]    |
|  Bulleted list of key                              Interactive       |
|  findings and implications                          time slider      |
|  for global health policy                           showing changes  |
|                                                     from 2000-2015   |
|                                                                      |
+----------------------------------------------------------------------+
```

### Visual Path
The visual path (indicated by arrows) guides the viewer from the introduction text to the central bubble plot visualization, which shows the three-way relationship between immunization coverage, life expectancy, and GDP. From there, the viewer explores the supporting visualizations that break down these relationships further. Finally, the path leads to key insights and the interactive element showing changes over time.

### WOW Element
The poster will include an interactive time slider allowing viewers to see how life expectancy, GDP, and immunization coverage have changed over the 15-year period (2000-2015). This will highlight countries that have made significant improvements despite economic limitations.

## Section 3: Visualizations

### Visualization 1: Distribution of Life Expectancy (Histogram)
![Histogram of Life Expectancy](/Users/Apexr/Documents/Poser 721/histogram_life_expectancy.pdf)

This histogram shows the overall distribution of life expectancy across all countries and years. It establishes the bimodal nature of global life expectancy, with distinct peaks for developing and developed nations. This visualization is important to our story because it visually demonstrates the gap in life expectancy that we're investigating and provides context for the magnitude of global health disparities.

### Visualization 2: Average Life Expectancy by Country Status (Bar Chart)
![Bar Chart of Life Expectancy by Status](/Users/Apexr/Documents/Poser 721/bar_status_life_expectancy.pdf)

This bar chart quantifies the average difference in life expectancy between developed and developing nations. It directly supports our story by showing the magnitude of the disparity (approximately 9-10 years) and serves as a clear, easy-to-understand visualization for viewers who may not be familiar with global health metrics.

### Visualization 3: GDP per Capita by Country Status (Boxplot)
![Boxplot of GDP by Status](/Users/Apexr/Documents/Poser 721/boxplot_gdp_status.pdf)

This boxplot shows the distribution of GDP per capita in developed versus developing countries. It is crucial to our story because it illustrates the economic divide that underlies the health disparity. The visualization reveals not just the difference in median GDP but also the much wider range and outliers in the developed nations category.

### Visualization 4: Life Expectancy vs. GDP (Scatter Plot)
![Scatter Plot of GDP vs Life Expectancy](/Users/Apexr/Documents/Poser 721/scatter_gdp_life_expenditure.pdf)

This higher-dimensional plot shows the relationship between GDP per capita and life expectancy, with point size representing health expenditure and color indicating country status. This visualization is central to our story because it demonstrates how economic factors correlate with health outcomes, while also showing that some developing countries achieve better life expectancy than their GDP would predict - suggesting other factors (like effective public health measures) play a role.

### Visualization 5: Life Expectancy vs. Immunization Coverage (Bubble Plot)
![Bubble Plot of Immunization, GDP, and Life Expectancy](/Users/Apexr/Documents/Poser 721/bubble_plot_immunization_gdp_life.pdf)

This multivariate bubble plot serves as our central visualization by showing the relationship between immunization coverage (using Polio as a proxy for overall immunization programs), life expectancy, GDP per capita (bubble size), and development status (color). This is the most important visualization for our story as it reveals how preventive public health measures like immunization correlate with higher life expectancy even in countries with lower GDP, offering a potential pathway for developing nations to improve health outcomes despite economic constraints.

## Section 4: Peer Feedback (to be completed after class)

### Comments about strengths:
[To be filled in after receiving peer feedback]

### Comments about areas for improvement:
[To be filled in after receiving peer feedback]

### Plans to address feedback:
[To be filled in after receiving peer feedback]