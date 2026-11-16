
# Customer Churn - Full Project Documentation PDF

from pathlib import Path
import re
import html
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    Table,
    TableStyle,
    Preformatted
)

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"

MD_PATH = REPORTS_DIR / "Customer_Churn_Full_Documentation.md"
PDF_PATH = REPORTS_DIR / "Customer_Churn_Full_Documentation.pdf"

# 2. Check required files
if not MD_PATH.exists():
    raise FileNotFoundError(
        f"Documentation file not found: {MD_PATH}"
    )

# 3. Create PDF document
doc = SimpleDocTemplate(
    str(PDF_PATH),
    pagesize=A4,
    rightMargin=42,
    leftMargin=42,
    topMargin=48,
    bottomMargin=45,
    title="Customer Intelligence & Predictive Retention Platform"
)

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="ProjectTitle",
    parent=styles["Title"],
    fontSize=21,
    leading=27,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#17365D"),
    spaceAfter=18
))

styles.add(ParagraphStyle(
    name="SectionHeading",
    parent=styles["Heading1"],
    fontSize=16,
    leading=20,
    textColor=colors.HexColor("#17365D"),
    spaceBefore=14,
    spaceAfter=9,
    keepWithNext=True
))

styles.add(ParagraphStyle(
    name="SubHeading",
    parent=styles["Heading2"],
    fontSize=12,
    leading=16,
    textColor=colors.HexColor("#24527A"),
    spaceBefore=10,
    spaceAfter=6,
    keepWithNext=True
))

styles.add(ParagraphStyle(
    name="ProjectBody",
    parent=styles["BodyText"],
    fontSize=9,
    leading=13,
    spaceAfter=6
))

styles.add(ParagraphStyle(
    name="ProjectBullet",
    parent=styles["BodyText"],
    fontSize=9,
    leading=13,
    leftIndent=16,
    firstLineIndent=-9,
    spaceAfter=4
))

styles.add(ParagraphStyle(
    name="ProjectCode",
    parent=styles["Code"],
    fontSize=7.5,
    leading=10,
    backColor=colors.HexColor("#F2F4F7"),
    borderPadding=6,
    spaceAfter=8
))

styles.add(ParagraphStyle(
    name="ChartCaption",
    parent=styles["BodyText"],
    fontSize=8.5,
    leading=12,
    spaceAfter=12
))

story = []


# 4. Format simple Markdown inline text
def format_inline(text):
    text = html.escape(text.strip(), quote=False)

    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"<b>\1</b>",
        text
    )

    text = re.sub(
        r"\*(.+?)\*",
        r"<i>\1</i>",
        text
    )

    text = re.sub(
        r"`(.+?)`",
        r'<font name="Courier">\1</font>',
        text
    )

    return text


# 5. Convert Markdown text into PDF elements
def add_markdown_content(markdown_text):
    lines = markdown_text.splitlines()
    paragraph_lines = []
    code_lines = []
    in_code = False
    i = 0

    def flush_paragraph():
        if paragraph_lines:
            text = " ".join(
                line.strip() for line in paragraph_lines
            )
            story.append(
                Paragraph(format_inline(text), styles["ProjectBody"])
            )
            paragraph_lines.clear()

    while i < len(lines):
        line = lines[i].strip()

        # Fenced code blocks
        if line.startswith("```"):
            flush_paragraph()

            if not in_code:
                in_code = True
                code_lines.clear()
            else:
                story.append(
                    Preformatted(
                        "\n".join(code_lines),
                        styles["ProjectCode"]
                    )
                )
                story.append(Spacer(1, 5))
                in_code = False

            i += 1
            continue

        if in_code:
            code_lines.append(lines[i])
            i += 1
            continue

        # Blank lines
        if not line:
            flush_paragraph()
            i += 1
            continue

        # Horizontal separators
        if line in ("---", "***", "___"):
            flush_paragraph()
            story.append(Spacer(1, 8))
            i += 1
            continue

        # Markdown tables
        if line.startswith("|") and i + 1 < len(lines):
            next_line = lines[i + 1].strip()

            if re.match(r"^\|?[\s|:-]+\|?$", next_line):
                flush_paragraph()

                headers = [
                    cell.strip()
                    for cell in line.strip("|").split("|")
                ]

                rows = [[
                    Paragraph(
                        f"<b>{format_inline(cell)}</b>",
                        styles["ProjectBody"]
                    )
                    for cell in headers
                ]]

                i += 2

                while i < len(lines) and lines[i].strip().startswith("|"):
                    cells = [
                        cell.strip()
                        for cell in lines[i].strip().strip("|").split("|")
                    ]

                    while len(cells) < len(headers):
                        cells.append("")

                    rows.append([
                        Paragraph(
                            format_inline(cell),
                            styles["ProjectBody"]
                        )
                        for cell in cells[:len(headers)]
                    ])

                    i += 1

                col_width = (A4[0] - 84) / len(headers)

                table = Table(
                    rows,
                    colWidths=[col_width] * len(headers),
                    repeatRows=1,
                    hAlign="LEFT"
                )

                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0),
                     colors.HexColor("#D9EAF7")),
                    ("GRID", (0, 0), (-1, -1), 0.4,
                     colors.HexColor("#B7C9D6")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))

                story.append(table)
                story.append(Spacer(1, 8))
                continue

        # Headings
        if line.startswith("# "):
            flush_paragraph()
            story.append(Spacer(1, 30))
            story.append(Paragraph(
                format_inline(line[2:]),
                styles["ProjectTitle"]
            ))
            i += 1
            continue

        if line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(
                format_inline(line[3:]),
                styles["SectionHeading"]
            ))
            i += 1
            continue

        if line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(
                format_inline(line[4:]),
                styles["SubHeading"]
            ))
            i += 1
            continue

        # Bullet points
        if line.startswith("- ") or line.startswith("* "):
            flush_paragraph()
            story.append(Paragraph(
                "&#8226; " + format_inline(line[2:]),
                styles["ProjectBullet"]
            ))
            i += 1
            continue

        # Numbered lists
        match = re.match(r"^(\d+)\.\s+(.*)", line)

        if match:
            flush_paragraph()
            story.append(Paragraph(
                match.group(1) + ". " + format_inline(match.group(2)),
                styles["ProjectBullet"]
            ))
            i += 1
            continue

        paragraph_lines.append(line)
        i += 1

    flush_paragraph()


# 6. Read the complete Markdown documentation
markdown_text = MD_PATH.read_text(encoding="utf-8")
add_markdown_content(markdown_text)


# 7. Add all project visualization images
charts = [
    (
        "churn_count.png",
        "Churned vs Retained Customers",
        "Compares the number of customers who left with those who stayed."
    ),
    (
        "contract_churn.png",
        "Churn Rate by Contract",
        "Compares churn rates across contract types."
    ),
    (
        "plan_churn.png",
        "Churn Rate by Internet Plan",
        "Shows churn rates across internet service categories."
    ),
    (
        "monthly_charges_by_churn.png",
        "Monthly Charges by Churn Status",
        "Compares monthly charges for churned and retained customers."
    ),
    (
        "tenure_by_churn.png",
        "Customer Tenure by Churn Status",
        "Compares customer tenure across churn outcomes."
    ),
    (
        "correlation_heatmap.png",
        "Numerical Feature Correlation",
        "Shows linear relationships between numerical features. "
        "Correlation does not prove causation."
    ),
    (
        "model_comparison.png",
        "Machine Learning Model Comparison",
        "Compares the model evaluation metrics saved by the project."
    ),
    (
        "confusion_matrix.png",
        "Confusion Matrix",
        "Summarizes correct and incorrect model predictions."
    ),
    (
        "roc_curve.png",
        "ROC Curve",
        "Shows the trade-off between true positive and false positive rates."
    ),
    (
        "precision_recall_curve.png",
        "Precision-Recall Curve",
        "Shows the relationship between precision and recall."
    ),
    (
        "feature_importance.png",
        "Feature Importance",
        "Displays features ranked by importance for the supported model."
    ),
    (
        "customer_segments.png",
        "Customer Segmentation",
        "Plots customer tenure and monthly charges by assigned segment."
    ),
    (
        "risk_category_counts.png",
        "Customers by Churn Risk",
        "Shows the number of test customers in each predicted risk category."
    ),
    (
        "revenue_exposure_by_risk.png",
        "Monthly Charges by Risk Category",
        "Shows monthly charges associated with each risk group, "
        "not confirmed revenue loss."
    ),
    (
        "churn_probability_distribution.png",
        "Churn Probability Distribution",
        "Shows the distribution of predicted churn probabilities."
    ),
]

story.append(PageBreak())
story.append(Paragraph(
    "Visualization Gallery",
    styles["SectionHeading"]
))
story.append(Paragraph(
    "This section contains the charts generated from the project data. "
    "Each chart is accompanied by a short explanation. Interpretations "
    "should be checked against the underlying data and model outputs.",
    styles["ProjectBody"]
))

available_charts = 0

for filename, title, description in charts:
    image_path = REPORTS_DIR / filename

    if not image_path.exists():
        story.append(Paragraph(
            f"<b>{html.escape(title)}</b> — Image not found: "
            f"{html.escape(filename)}",
            styles["ProjectBody"]
        ))
        continue

    story.append(Paragraph(
        format_inline(title),
        styles["SubHeading"]
    ))

    story.append(Paragraph(
        format_inline(description),
        styles["ChartCaption"]
    ))

    chart_image = Image(str(image_path))
    chart_image._restrictSize(6.4 * inch, 4.2 * inch)

    story.append(chart_image)
    story.append(Spacer(1, 14))
    available_charts += 1


# 8. Page numbers
def add_page_number(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(
        A4[0] / 2,
        25,
        f"Customer Churn Project Documentation | Page {document.page}"
    )
    canvas.restoreState()


# 9. Generate final PDF
doc.build(
    story,
    onFirstPage=add_page_number,
    onLaterPages=add_page_number
)

print("\nFull project documentation PDF created successfully!")
print("PDF location:", PDF_PATH)
print("Visualization images included:", available_charts, "/", len(charts))
print("Source documentation:", MD_PATH)
