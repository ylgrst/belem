import pylatex as pl
import numpy as np

def import_extra_packages(doc: pl.document.Document):
    doc.packages.append(pl.Package('float'))
    doc.packages.append(pl.Package('titlepic'))
    doc.packages.append(pl.Package('babel', options="english"))

def generate_title_page(doc: pl.document.Document, start_page_number: int):
    doc.preamble.append(pl.Command("title", pl.LargeText("Results")))
    doc.preamble.append(pl.Command("date", ""))
    doc.append(pl.NoEscape(r"\maketitle"))
    doc.append(pl.NoEscape(r"\thispagestyle{empty}"))
    doc.append(pl.NewPage())
    doc.append(pl.NoEscape(r"\setcounter{page}{%d}" %start_page_number))

def generate_header(doc: pl.document.Document, shape: str, density: float):

    header = pl.PageStyle("header")
    with header.create(pl.Head("L")):
        header.append("Yasmin Legerstee")
    with header.create(pl.Head("C")):
        header.append("Non linear homogenization")
        with header.create(pl.Head("R")):
            header.append("I2M - SAMC")
    with header.create(pl.Foot("L")):
        header.append(shape + " " + str(int(100*density)))
    with header.create(pl.Foot("R")):
        header.append(pl.simple_page_number())

    doc.preamble.append(header)
    doc.change_document_style("header")

def generate_chapter_page(doc: pl.document.Document, shape: str):
    with doc.create(pl.Chapter(shape, numbering=False)):

        with doc.create(pl.Figure(position="H")) as intro_pic:
            intro_pic.add_image(pl.NoEscape(r"example-image-duck"), width=pl.NoEscape(r"0.8\textwidth"))

    doc.append(pl.NewPage())

def generate_section(doc: pl.document.Document, shape: str, density: float):
    ...

def generate_response_curves_subsection(doc: pl.document.Document):
    ...

def generate_hardening_curves_subsection(doc: pl.document.Document):
    ...

def generate_stress_concentration_map_subsection(doc: pl.document.Document):
    ...

def generate_dfa_subsection(doc: pl.document.Document):
    ...

def generate_identification_subsection(doc: pl.document.Document):
    ...