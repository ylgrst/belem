import pylatex as pl
import numpy as np

def import_extra_packages(doc: pl.document.Document):
    doc.packages.append(pl.Package('float'))
    doc.packages.append(pl.Package('titlepic'))
    doc.packages.append(pl.Package('babel', options="english"))

def center_all_floats(doc: pl.document.Document):
    doc.preamble.append(pl.NoEscape(r"\makeatletter"))
    doc.preamble.append(pl.NoEscape(r"\g@addto@macro\@floatboxreset\centering"))
    doc.preamble.append(pl.NoEscape(r"\makeatother"))

def generate_title_page(doc: pl.document.Document, start_page_number: int):
    doc.preamble.append(pl.Command("title", pl.LargeText("Results")))
    doc.preamble.append(pl.Command("date", ""))
    doc.append(pl.NoEscape(r"\maketitle"))
    doc.append(pl.NoEscape(r"\thispagestyle{empty}"))
    doc.append(pl.NewPage())
    doc.append(pl.NoEscape(r"\setcounter{page}{%d}" %start_page_number))

def generate_header(doc: pl.document.Document):

    header = pl.PageStyle("fancy")
    with header.create(pl.Head("L")):
        header.append("Yasmin Legerstee")
    with header.create(pl.Head("R")):
        header.append("Non linear homogenization")
        with header.create(pl.Head("R")):
            header.append("I2M - SAMC")
    with header.create(pl.Foot("L")):
        #header.append(shape + " " + str(int(100*density)) + "%") #, shape: str, density: float
        header.append(pl.Command("leftmark"))
    with header.create(pl.Foot("R")):
        header.append(pl.Command("thepage"))

    doc.preamble.append(header)
    doc.change_document_style("fancy")

def generate_chapter_page(doc: pl.document.Document, shape: str):
    with doc.create(pl.Chapter(shape, numbering=False)):

        with doc.create(pl.Figure(position="H")) as intro_pic:
            intro_pic.add_image(pl.NoEscape(r"example-image-duck"), width=pl.NoEscape(r"0.8\textwidth"))

    doc.append(pl.NewPage())

def generate_section(doc: pl.document.Document, shape: str, density: float):
    with doc.create(pl.Section(shape + " " + str(int(100*density)) + "%", numbering=False)):
        pl.Command("markboth", [shape + " " + str(int(100*density)) + "%", ""])
        generate_dfa_subsection(doc)
        generate_identification_subsection(doc)

def generate_response_curves_subsection(doc: pl.document.Document):
    ...

def generate_hardening_curves_subsection(doc: pl.document.Document):
    ...

def generate_stress_concentration_map_subsection(doc: pl.document.Document):
    ...

def generate_dfa_subsection(doc: pl.document.Document):
    with doc.create(pl.Subsection("Deshpande-Fleck-Ashby yield surface identification", numbering=False)):
        with doc.create(pl.Figure(position="H")) as dfa_1122_fig:
            dfa_1122_fig.add_image(pl.NoEscape(r"example-image-duck"), width=pl.NoEscape(r"0.8\textwidth"))
            dfa_1122_fig.add_caption(
                "Simulated yield surface and identified Deshpande-Fleck-Ashby yield surface in the (S11,S22) plane")

        with doc.create(pl.Figure(position="H")) as dfa_1112_fig:
            dfa_1112_fig.add_image(pl.NoEscape(r"example-image-duck"), width=pl.NoEscape(r"0.8\textwidth"))
            dfa_1112_fig.add_caption("Simulated yield surface and identified Deshpande-Fleck-Ashby yield surface in the (S11,S12) plane")

        with doc.create(pl.Table(position="H")) as dfa_tab:
            with doc.create(pl.Tabular("|c|c|c|c|c|c|c|")) as dfa_table:
                dfa_table.add_hline()
                dfa_table.add_row(("F", "G", "H", "L", "M", "N", "K"))
                dfa_table.add_hline()
                dfa_table.add_row((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                dfa_table.add_hline()
            dfa_tab.add_caption("Identified Deshpande-Fleck-Ashby yield surface parameters")


def generate_identification_subsection(doc: pl.document.Document):
    with doc.create(pl.Subsection("Homogenized behavior law identification", numbering=False)):
        with doc.create(pl.Figure(position="H")) as ident_graph_pic:
            ident_graph_pic.add_image(pl.NoEscape(r"example-image-duck"), width=pl.NoEscape(r"0.8\textwidth"))
            ident_graph_pic.add_caption("Comparison between simulated behavior and identified non linear behavior law")

        with doc.create(pl.Table(position="H")) as law_param_tab:
            with doc.create(pl.Tabular("|c|c|c|c|c|c|c|c|c|c|")) as law_param_table:
                law_param_table.add_hline()
                law_param_table.add_row(("E", pl.Math(data=pl.NoEscape(r"\nu"), inline=True), "G", pl.Math(data=pl.NoEscape(r"\alpha"), inline=True), "Q", "b", pl.Math(data=pl.NoEscape("C_{1}"), inline=True), pl.Math(data=pl.NoEscape("D_{1}"), inline=True), pl.Math(data=pl.NoEscape("C_{2}"), inline=True), pl.Math(data=pl.NoEscape("D_{2}"), inline=True)))
                law_param_table.add_hline()
                law_param_table.add_row((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                law_param_table.add_hline()
            law_param_tab.add_caption("Homogenized law parameters")
