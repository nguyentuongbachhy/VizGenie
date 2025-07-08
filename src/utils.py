import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

DB_NAME = "db.sqlite"

def execute_plt_code(code: str, df: pd.DataFrame):
    try:
        # local_vars = {"plt": plt, "df": df}
        import seaborn as sns
        local_vars = {"plt": plt, "df": df, "sns": sns}

        compiled_code = compile(code, "<string>", "exec")
        exec(compiled_code, globals(), local_vars)
        return plt.gcf()
    except Exception as e:
        st.error(f"Error executing plt code: {e}")
        return None

def safe_read_csv(file_path):
    for enc in ['utf-8', 'ISO-8859-1', 'utf-16', 'cp1252']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to decode file with common encodings.")

def get_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            path TEXT,
            num_rows INTEGER,
            num_cols INTEGER,
            upload_time TEXT,
            status TEXT
        )''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            timestamp TEXT,
            question TEXT,
            answer TEXT,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id)
        )''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS chart_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            question TEXT,
            answer TEXT,
            code TEXT,
            created_at TEXT,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id)
        )''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            title TEXT,
            created_at TEXT
        )''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            role TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
        )''')
    conn.commit()
    conn.close()

def add_dataset(name, path, num_rows, num_cols, upload_time, status="Uploaded"):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO datasets (name, path, num_rows, num_cols, upload_time, status)
        VALUES (?, ?, ?, ?, ?, ?)''', (name, path, num_rows, num_cols, upload_time, status))
    conn.commit()
    conn.close()

def get_all_datasets():
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT id, name, num_rows, num_cols, upload_time, status FROM datasets')
    rows = c.fetchall()
    conn.close()
    return rows

def get_dataset(id):
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM datasets WHERE id = ?', (id,))
    row = c.fetchone()
    conn.close()
    return row

def add_chat(dataset_id, question, answer):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO chat_history (dataset_id, timestamp, question, answer)
        VALUES (?, ?, ?, ?)''', (dataset_id, timestamp, question, answer))
    conn.commit()
    conn.close()

def get_chats_by_dataset(dataset_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT timestamp, question, answer FROM chat_history WHERE dataset_id = ? ORDER BY id', (dataset_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def add_chart_card(dataset_id, question, answer, code):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO chart_cards (dataset_id, question, answer, code, created_at)
        VALUES (?, ?, ?, ?, ?)''', (dataset_id, question, answer, code, timestamp))
    conn.commit()
    conn.close()

def get_chart_cards_by_dataset(dataset_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT question, answer, code, created_at
        FROM chart_cards
        WHERE dataset_id = ?
        ORDER BY id DESC''', (dataset_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def create_chat_session(dataset_id, title):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO chat_sessions (dataset_id, title, created_at)
        VALUES (?, ?, ?)''', (dataset_id, title, timestamp))
    conn.commit()
    session_id = c.lastrowid
    conn.close()
    return session_id

def get_sessions_by_dataset(dataset_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT id, title, created_at
        FROM chat_sessions
        WHERE dataset_id = ?
        ORDER BY created_at DESC''', (dataset_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def add_chat_message(session_id, role, content):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO chat_messages (session_id, role, content, created_at)
        VALUES (?, ?, ?, ?)''', (session_id, role, content, timestamp))
    conn.commit()
    conn.close()

def get_chat_messages(session_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT id, role, content, created_at
        FROM chat_messages
        WHERE session_id = ?
        ORDER BY id
    ''', (session_id,))
    messages = c.fetchall()
    conn.close()
    return messages

def delete_chat_message(session_id, message_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        DELETE FROM chat_messages
        WHERE session_id = ? AND id = ?
    ''', (session_id, message_id))
    conn.commit()
    conn.close()

def delete_chat_session(session_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
    c.execute('DELETE FROM chat_sessions WHERE id = ?', (session_id,))
    conn.commit()
    conn.close()

def rename_chat_session(session_id, new_title):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        UPDATE chat_sessions
        SET title = ?
        WHERE id = ?
    ''', (new_title, session_id))
    conn.commit()
    conn.close()

def delete_dataset(dataset_id):
    conn = get_connection()
    c = conn.cursor()
    
    # Xoá liên quan (nếu cần)
    c.execute('DELETE FROM chat_messages WHERE session_id IN (SELECT id FROM chat_sessions WHERE dataset_id = ?)', (dataset_id,))
    c.execute('DELETE FROM chat_sessions WHERE dataset_id = ?', (dataset_id,))
    c.execute('DELETE FROM chart_cards WHERE dataset_id = ?', (dataset_id,))

    # Xoá chính dataset
    c.execute('DELETE FROM datasets WHERE id = ?', (dataset_id,))
    
    conn.commit()
    conn.close()

def rename_dataset(dataset_id, new_name):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        UPDATE datasets
        SET name = ?
        WHERE id = ?
    ''', (new_name, dataset_id))
    conn.commit()
    conn.close()

def delete_chart_card(dataset_id: int, question: str, created_at: str):
    conn = sqlite3.connect("db.sqlite")
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM chart_cards WHERE dataset_id = ? AND question = ? AND created_at = ?",
        (dataset_id, question, created_at)
    )
    conn.commit()
    conn.close()

# ---------------------------- Export Functions using ReportLab ---------------------------- 

def create_chart_image(code: str, df: pd.DataFrame) -> BytesIO:
    """Execute plotting code and return image as BytesIO object"""
    try:
        plt.figure(figsize=(8, 6))
        exec(code, {"df": df, "plt": plt, "sns": sns})
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        return img_buffer
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None

def create_styled_paragraph(text: str, style_name: str = 'Normal') -> Paragraph:
    """Create a styled paragraph from text"""
    styles = getSampleStyleSheet()
    if style_name in styles:
        style = styles[style_name]
    else:
        style = styles['Normal']
    
    # Clean up markdown-style formatting
    text = text.replace('**', '<b>').replace('**', '</b>')
    text = text.replace('*', '<i>').replace('*', '</i>')
    
    return Paragraph(text, style)

def export_eda_report_to_pdf(eda_sections, df, summary_response, dataset_name):
    """Create PDF report using ReportLab instead of pdfkit"""
    
    # Create a buffer to store PDF
    buffer = BytesIO()
    
    # Create document with ReportLab
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=HexColor('#2c3e50'),
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=HexColor('#2c3e50')
    )
    
    # Story (content) list
    story = []
    
    # Title
    story.append(Paragraph(f"EDA Report: {dataset_name}", title_style))
    story.append(Spacer(1, 20))
    
    # Introduction
    story.append(Paragraph("📘 Introduction", heading_style))
    story.append(create_styled_paragraph(eda_sections['introduction']))
    story.append(Spacer(1, 12))
    
    # Data Quality
    story.append(Paragraph("🧼 Data Quality", heading_style))
    story.append(create_styled_paragraph(eda_sections['data_quality']))
    story.append(Spacer(1, 12))
    
    # Dataset preview table
    story.append(Paragraph("Dataset Preview (First 5 rows):", styles['Heading3']))
    preview_data = df.head().to_string()
    story.append(Paragraph(f"<pre>{preview_data}</pre>", styles['Code']))
    story.append(Spacer(1, 12))
    
    # Univariate Analysis
    story.append(Paragraph("🔍 Univariate Analysis", heading_style))
    
    for idx, block in enumerate(eda_sections.get('univariate', [])):
        story.append(Paragraph(f"Analysis {idx + 1}: {block['insight']}", styles['Heading3']))
        
        # Add code block
        story.append(Paragraph("Code:", styles['Heading4']))
        story.append(Paragraph(f"<pre>{block['code']}</pre>", styles['Code']))
        story.append(Spacer(1, 6))
        
        # Add chart if possible
        chart_img = create_chart_image(block['code'], df)
        if chart_img:
            story.append(Image(chart_img, width=6*inch, height=4*inch))
            story.append(Spacer(1, 6))
        
        # Add insight after chart
        if 'insight_after_chart' in block:
            story.append(create_styled_paragraph(f"Insight: {block['insight_after_chart']}"))
        
        story.append(Spacer(1, 12))
    
    # Correlation Analysis
    story.append(PageBreak())
    story.append(Paragraph("📊 Correlation Analysis", heading_style))
    story.append(create_styled_paragraph(eda_sections['correlation']['insight']))
    story.append(Spacer(1, 6))
    
    # Correlation code
    story.append(Paragraph("Code:", styles['Heading4']))
    story.append(Paragraph(f"<pre>{eda_sections['correlation']['code']}</pre>", styles['Code']))
    story.append(Spacer(1, 6))
    
    # Correlation chart
    corr_img = create_chart_image(eda_sections['correlation']['code'], df)
    if corr_img:
        story.append(Image(corr_img, width=6*inch, height=4*inch))
        story.append(Spacer(1, 6))
    
    if 'insight_after_chart' in eda_sections['correlation']:
        story.append(create_styled_paragraph(f"Insight: {eda_sections['correlation']['insight_after_chart']}"))
    
    story.append(Spacer(1, 12))
    
    # Final Insights & Recommendations
    story.append(Paragraph("💡 Final Insights & Recommendations", heading_style))
    story.append(create_styled_paragraph(summary_response))
    
    # Build PDF
    doc.build(story)
    
    # Get PDF bytes
    buffer.seek(0)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes