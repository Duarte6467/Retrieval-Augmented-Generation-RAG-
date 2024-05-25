import pdfplumber


pdf = pdfplumber.open('CRPD.pdf')


# Extract text from the first page
first_page = pdf.pages[0]
text =''


for page in pdf.pages:
    text += page.extract_text()

"""Text is now processed into a string"""

