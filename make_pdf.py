import markdown
from pathlib import Path

# Read markdown file
md_path = Path("analysis_report.md")
md_content = md_path.read_text(encoding='utf-8')

# Convert markdown to HTML with math support
md_converter = markdown.Markdown(extensions=['tables', 'fenced_code'])
html_body = md_converter.convert(md_content)

# Fix image paths to be absolute for all images
base_path = Path(".").resolve()
# Replace all image src paths
import re
base_path_str = str(base_path).replace('\\', '/')
html_body = re.sub(r'src="([^"]+\.png)"', lambda m: f'src="file:///{base_path_str}/{m.group(1)}"', html_body)

# Create full HTML with styling
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Migration Matrix Analysis Report</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        @media print {{
            body {{ margin: 0.5in; }}
            h1, h2 {{ page-break-after: avoid; }}
            table, img {{ page-break-inside: avoid; }}
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
            font-size: 11pt;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            font-size: 22pt;
        }}
        h2 {{
            color: #2980b9;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
            margin-top: 30px;
            font-size: 16pt;
        }}
        h3 {{
            color: #34495e;
            font-size: 13pt;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 10pt;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 6px 10px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', monospace;
            font-size: 10pt;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
        strong {{
            color: #2c3e50;
        }}
        a {{
            color: #3498db;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""

# Write HTML file
html_path = Path("analysis_report.html")
html_path.write_text(html_content, encoding='utf-8')
print(f"HTML saved to: {html_path.resolve()}")
print(f"\nTo create PDF:")
print(f"  1. Open {html_path.resolve()} in your browser")
print(f"  2. Press Ctrl+P (Print)")
print(f"  3. Select 'Save as PDF' as the destination")
print(f"  4. Click Save")
