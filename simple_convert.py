#!/usr/bin/env python3
"""
Simple Markdown to HTML converter with instructions for PDF/DOCX conversion
"""

import os
import markdown
from pathlib import Path

# File paths
md_file = '/Users/Apexr/Documents/Disease_Prediction_Project/COVID19_Project_Report_4page.md'
html_file = '/Users/Apexr/Documents/Disease_Prediction_Project/COVID19_Project_Report_4page.html'
pdf_file = '/Users/Apexr/Documents/Disease_Prediction_Project/COVID19_Project_Report_4page.pdf'
docx_file = '/Users/Apexr/Documents/Disease_Prediction_Project/COVID19_Project_Report_4page.docx'

# Read markdown content
with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert markdown to HTML with simple extension support
try:
    html = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code']
    )
except ImportError:
    # If extensions aren't available, use basic conversion
    html = markdown.markdown(md_content)

# Add CSS styling
styled_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>COVID-19 Detection from Unstructured Medical Text</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            padding: 20px;
            max-width: 800px;
            margin: 0 auto;
            color: #333;
        }}
        h1, h2, h3, h4, h5, h6 {{
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
        }}
        h1 {{ font-size: 2em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
        h2 {{ font-size: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
        h3 {{ font-size: 1.25em; }}
        h4 {{ font-size: 1em; }}
        h5 {{ font-size: 0.875em; }}
        h6 {{ font-size: 0.85em; color: #6a737d; }}
        
        /* Enhanced code formatting */
        code, pre {{
            font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
        }}
        
        pre {{
            background-color: #f6f8fa;
            border-radius: 6px;
            padding: 16px;
            overflow: auto;
            line-height: 1.45;
            margin-bottom: 16px;
            border: 1px solid #e1e4e8;
        }}
        
        code {{
            background-color: #f6f8fa;
            padding: 0.2em 0.4em;
            margin: 0;
            font-size: 85%;
            border-radius: 3px;
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
            margin: 0;
            font-size: 100%;
            word-break: normal;
            white-space: pre;
            border: none;
        }}
        
        /* Method section special formatting */
        .method-step {{
            margin-bottom: 20px;
            padding-left: 10px;
            border-left: 3px solid #0366d6;
        }}
        
        .method-step h4 {{
            color: #0366d6;
            margin-top: 0;
        }}
        
        .code-example {{
            display: flex;
            margin-bottom: 20px;
        }}
        
        .code-example-before, .code-example-after {{
            flex: 1;
            padding: 10px;
            border-radius: 6px;
            background-color: #f6f8fa;
            border: 1px solid #e1e4e8;
            margin: 5px;
        }}
        
        .code-example-before h4, .code-example-after h4 {{
            margin-top: 0;
            color: #0366d6;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 16px;
        }}
        
        table th, table td {{
            padding: 8px 13px;
            border: 1px solid #dfe2e5;
        }}
        
        table th {{
            background-color: #f1f8ff;
            font-weight: 600;
        }}
        
        table tr {{
            background-color: #fff;
            border-top: 1px solid #c6cbd1;
        }}
        
        table tr:nth-child(2n) {{
            background-color: #f6f8fa;
        }}
        
        img {{
            max-width: 100%;
            box-sizing: border-box;
            display: block;
            margin: 20px auto;
            border-radius: 6px;
        }}
        
        hr {{
            height: 0.25em;
            padding: 0;
            margin: 24px 0;
            background-color: #e1e4e8;
            border: 0;
        }}
    </style>
</head>
<body>
    {html}
</body>
</html>"""

# Write the HTML file
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(styled_html)

print(f"HTML file created: {html_file}")

# Instructions for manual conversion
print("\n===== MANUAL CONVERSION INSTRUCTIONS =====")

print("\nTo convert to PDF:")
print(f"1. Open the HTML file in your browser: {html_file}")
print("2. Press Cmd+P (Mac) or Ctrl+P (Windows) to open the print dialog")
print("3. Choose 'Save as PDF' as the destination")
print(f"4. Save to: {pdf_file}")

print("\nTo convert to Word (DOCX):")
print(f"1. Open the HTML file in your browser: {html_file}")
print("2. Select all content (Cmd+A or Ctrl+A)")
print("3. Copy the content (Cmd+C or Ctrl+C)")
print("4. Open Microsoft Word or another word processor")
print("5. Paste the content (Cmd+V or Ctrl+V)")
print(f"6. Save as: {docx_file}")

print(f"\nHTML file location: {html_file}")