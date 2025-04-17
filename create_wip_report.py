"""
Create a PDF version of the WIP report.
"""

import os
from fpdf import FPDF
import re

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'WHO Life Expectancy Data Story - WIP Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_wip_pdf(md_file, pdf_file):
    """Convert markdown WIP report to PDF."""
    # Read markdown file
    with open(md_file, 'r') as f:
        md_content = f.read()
    
    # Create PDF
    pdf = PDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font('Arial', 'B', 16)
    
    # Title
    pdf.cell(0, 10, 'WHO Life Expectancy Data Story', 0, 1, 'C')
    pdf.cell(0, 10, 'Work-in-Progress Report', 0, 1, 'C')
    pdf.ln(10)
    
    # Add name, section, title section
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, 'NAME: [Your Name]', 0, 1)
    pdf.cell(0, 10, 'SECTION: [Your Section]', 0, 1)
    pdf.cell(0, 10, 'TITLE: Global Health Disparities: The Gap Between Developed and Developing Nations', 0, 1)
    pdf.ln(10)
    
    # Section 1
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Section 1: Dataset and Story', 0, 1)
    pdf.ln(5)
    
    # Dataset description
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Dataset Description', 0, 1)
    pdf.set_font('Arial', '', 12)
    dataset_desc = """This project uses the WHO Life Expectancy dataset, which contains health, economic, and social factors affecting life expectancy across 193 countries from 2000 to 2015. The dataset includes approximately 2,938 observations (193 countries × 15 years) with 22 columns of variables. These variables include life expectancy, country status (developed vs. developing), GDP, education, alcohol consumption, health expenditure, immunization coverage, and various disease prevalence rates. The dataset was sourced from the World Health Organization's Global Health Observatory data repository."""
    pdf.multi_cell(0, 10, dataset_desc)
    pdf.ln(5)
    
    # Story
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Story', 0, 1)
    pdf.set_font('Arial', '', 12)
    story_text = """This data story is a quest to understand what factors create the significant gap in life expectancy between developed and developing nations. It examines how economic prosperity (GDP), healthcare investment, and preventive measures (immunization) interact to shape life expectancy outcomes globally.

This is a quest story because it follows the journey of identifying the magnitude of global health disparities and seeks to uncover the underlying causes. The central visualization (bubble plot) shows how three critical factors (immunization, GDP, and life expectancy) interact, revealing patterns that aren't visible when examining each factor in isolation. Supporting visualizations further break down these relationships, showing how developed nations consistently outperform developing ones across key health indicators. The story concludes with evidence that investment in basic public health measures like immunization can significantly improve life expectancy even in countries with lower GDP."""
    pdf.multi_cell(0, 10, story_text)
    pdf.ln(5)
    
    # Section 2
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Section 2: Poster Layout Sketch', 0, 1)
    pdf.ln(5)
    
    # Simple ascii art representation of the poster
    pdf.set_font('Courier', '', 10)
    poster_sketch = """
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
|                                                                      |
|    |                                     |                           |
|    v                                     v                           |
|                                                                      |
|  [SUPPORT VIZ 1]                 [SUPPORT VIZ 2]     [SUPPORT VIZ 3] |
|  Histogram of Life              Bar Chart of Avg.    Scatter Plot of |
|  Expectancy                    Life Expectancy       GDP vs Life     |
|  Distribution                  by Country Status     Expectancy with |
|                                                      Health          |
|                                                      Expenditure     |
|    |                                     |                |          |
|    v                                     v                v          |
|                                                                      |
|  [KEY INSIGHTS]                                     [WOW ELEMENT]    |
|  Bulleted list of key                              Interactive       |
|  findings and implications                          time slider      |
|  for global health policy                           showing changes  |
|                                                     from 2000-2015   |
|                                                                      |
+----------------------------------------------------------------------+
"""
    pdf.multi_cell(0, 5, poster_sketch)
    pdf.ln(5)
    
    # Visual path description
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Visual Path', 0, 1)
    pdf.set_font('Arial', '', 12)
    path_desc = """The visual path (indicated by vertical arrows) guides the viewer from the introduction text to the central bubble plot visualization, which shows the three-way relationship between immunization coverage, life expectancy, and GDP. From there, the viewer explores the supporting visualizations that break down these relationships further. Finally, the path leads to key insights and the interactive element showing changes over time."""
    pdf.multi_cell(0, 10, path_desc)
    
    # WOW element
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'WOW Element', 0, 1)
    pdf.set_font('Arial', '', 12)
    wow_desc = """The poster will include an interactive time slider allowing viewers to see how life expectancy, GDP, and immunization coverage have changed over the 15-year period (2000-2015). This will highlight countries that have made significant improvements despite economic limitations."""
    pdf.multi_cell(0, 10, wow_desc)
    pdf.ln(5)
    
    # Section 3
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Section 3: Visualizations', 0, 1)
    pdf.ln(5)
    
    # Visualization descriptions
    visualizations = [
        {
            "title": "Visualization 1: Distribution of Life Expectancy (Histogram)",
            "desc": """This histogram shows the overall distribution of life expectancy across all countries and years. It establishes the bimodal nature of global life expectancy, with distinct peaks for developing and developed nations. This visualization is important to our story because it visually demonstrates the gap in life expectancy that we're investigating and provides context for the magnitude of global health disparities."""
        },
        {
            "title": "Visualization 2: Average Life Expectancy by Country Status (Bar Chart)",
            "desc": """This bar chart quantifies the average difference in life expectancy between developed and developing nations. It directly supports our story by showing the magnitude of the disparity (approximately 9-10 years) and serves as a clear, easy-to-understand visualization for viewers who may not be familiar with global health metrics."""
        },
        {
            "title": "Visualization 3: GDP per Capita by Country Status (Boxplot)",
            "desc": """This boxplot shows the distribution of GDP per capita in developed versus developing countries. It is crucial to our story because it illustrates the economic divide that underlies the health disparity. The visualization reveals not just the difference in median GDP but also the much wider range and outliers in the developed nations category."""
        },
        {
            "title": "Visualization 4: Life Expectancy vs. GDP (Scatter Plot)",
            "desc": """This higher-dimensional plot shows the relationship between GDP per capita and life expectancy, with point size representing health expenditure and color indicating country status. This visualization is central to our story because it demonstrates how economic factors correlate with health outcomes, while also showing that some developing countries achieve better life expectancy than their GDP would predict - suggesting other factors (like effective public health measures) play a role."""
        },
        {
            "title": "Visualization 5: Life Expectancy vs. Immunization Coverage (Bubble Plot)",
            "desc": """This multivariate bubble plot serves as our central visualization by showing the relationship between immunization coverage (using Polio as a proxy for overall immunization programs), life expectancy, GDP per capita (bubble size), and development status (color). This is the most important visualization for our story as it reveals how preventive public health measures like immunization correlate with higher life expectancy even in countries with lower GDP, offering a potential pathway for developing nations to improve health outcomes despite economic constraints."""
        }
    ]
    
    # Add each visualization with description
    for viz in visualizations:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, viz["title"], 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, viz["desc"])
        pdf.ln(5)
    
    # Section 4 (empty for now)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Section 4: Peer Feedback (to be completed after class)', 0, 1)
    pdf.ln(5)
    
    # Placeholders for feedback
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Comments about strengths:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, "[To be filled in after receiving peer feedback]")
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Comments about areas for improvement:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, "[To be filled in after receiving peer feedback]")
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Plans to address feedback:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, "[To be filled in after receiving peer feedback]")
    
    # Save the PDF
    pdf.output(pdf_file)
    print(f"PDF report saved to {pdf_file}")

if __name__ == "__main__":
    os.makedirs("presentations", exist_ok=True)
    
    md_file = "presentations/WHO_Life_Expectancy_WIP.md"
    pdf_file = "presentations/WHO_Life_Expectancy_WIP_Report.pdf"
    
    create_wip_pdf(md_file, pdf_file)