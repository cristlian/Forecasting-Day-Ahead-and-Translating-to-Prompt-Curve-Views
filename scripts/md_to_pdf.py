"""Convert submission.md to PDF using reportlab."""
import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted
from reportlab.lib import colors

# Paths
report_dir = Path(__file__).parent.parent / "report"
md_path = report_dir / "submission.md"
pdf_path = report_dir / "submission.pdf"

def clean_text(text):
    """Remove markdown formatting and handle special chars."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)  # bold
    text = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', text)  # inline code
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # links
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # images
    # Replace special characters
    text = text.replace('€', 'EUR')
    text = text.replace('✅', '[OK]')
    text = text.replace('✓', '[OK]')
    text = text.replace('₂', '2')
    text = text.replace('²', '2')
    text = text.replace('σ', 'sigma')
    text = text.replace('×', 'x')
    text = text.replace('→', '->')
    text = text.replace('&', '&amp;')
    text = text.replace('<b>', '###BOLD###').replace('</b>', '###/BOLD###')
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    text = text.replace('###BOLD###', '<b>').replace('###/BOLD###', '</b>')
    return text.strip()

def clean_plain(text):
    """Plain text without HTML."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = text.replace('€', 'EUR')
    text = text.replace('✅', '[OK]')
    text = text.replace('✓', '[OK]')
    text = text.replace('₂', '2')
    text = text.replace('²', '2')
    text = text.replace('σ', 'sigma')
    text = text.replace('×', 'x')
    text = text.replace('→', '->')
    return text.strip()

# Read markdown
content = md_path.read_text(encoding="utf-8")
lines = content.split('\n')

# Styles
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Title2', parent=styles['Heading1'], fontSize=16, spaceAfter=12))
styles.add(ParagraphStyle(name='H2', parent=styles['Heading2'], fontSize=13, spaceBefore=16, spaceAfter=8))
styles.add(ParagraphStyle(name='H3', parent=styles['Heading3'], fontSize=11, spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle(name='Body2', parent=styles['Normal'], fontSize=10, leading=14))
styles.add(ParagraphStyle(name='MyBullet', parent=styles['Normal'], fontSize=10, leftIndent=15, bulletIndent=5))
styles.add(ParagraphStyle(name='MyCode', fontName='Courier', fontSize=8, leading=10, backColor=colors.Color(0.95, 0.95, 0.95)))

# Build document
doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
story = []

in_code_block = False
code_lines = []
table_rows = []

for line in lines:
    # Code blocks
    if line.startswith('```'):
        if in_code_block:
            # End code block
            if code_lines:
                code_text = '\n'.join(code_lines)
                story.append(Preformatted(code_text, styles['MyCode']))
                story.append(Spacer(1, 6))
            code_lines = []
        in_code_block = not in_code_block
        continue
    
    if in_code_block:
        code_lines.append(clean_plain(line))
        continue
    
    # Tables
    if '|' in line and not line.startswith('!['):
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if cells and not all(c.replace('-', '').replace(':', '') == '' for c in cells):
            table_rows.append([clean_plain(c)[:40] for c in cells])
        continue
    elif table_rows:
        # End table
        t = Table(table_rows)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))
        table_rows = []
    
    # Skip empty lines
    if not line.strip():
        continue
    
    # Headers
    if line.startswith('# '):
        story.append(Paragraph(clean_text(line[2:]), styles['Title2']))
    elif line.startswith('## '):
        story.append(Paragraph(clean_text(line[3:]), styles['H2']))
    elif line.startswith('### '):
        story.append(Paragraph(clean_text(line[4:]), styles['H3']))
    elif line.startswith('---'):
        story.append(Spacer(1, 6))
    elif line.startswith('$$') or '\\text' in line:
        # LaTeX formula - render as code
        text = line.replace('$$', '').strip()
        if text:
            story.append(Preformatted(clean_plain(text), styles['MyCode']))
    elif line.startswith('- '):
        story.append(Paragraph('• ' + clean_text(line[2:]), styles['MyBullet']))
    elif line.startswith('Where:'):
        story.append(Paragraph(clean_text(line), styles['Body2']))
    else:
        text = clean_text(line)
        if text:
            story.append(Paragraph(text, styles['Body2']))

# Flush any remaining table
if table_rows:
    t = Table(table_rows)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

# Build PDF
doc.build(story)
print(f"PDF generated: {pdf_path}")
