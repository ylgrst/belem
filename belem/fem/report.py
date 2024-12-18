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

    doc.preamble.append(pl.NoEscape(r"\renewcommand{\sectionmark}[1]{\markboth{#1}{}}"))
    header = pl.PageStyle("fancy")
    with header.create(pl.Head("L")):
        header.append("Yasmin Legerstee")
    with header.create(pl.Head("R")):
        header.append("Non linear homogenization")
        with header.create(pl.Head("C")):
            header.append("I2M - SAMC")
    with header.create(pl.Foot("L")):
        header.append(pl.Command("sectionmark"))
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
        generate_dfa_subsection(doc)
        generate_identification_subsection(doc)

def generate_response_curves_subsection(doc: pl.document.Document):
    ...

def generate_hardening_curves_subsection(doc: pl.document.Document):
    ...

def generate_stress_concentration_map_subsection(doc: pl.document.Document):
    ...

def generate_dfa_subsection(doc: pl.document.Document, dfa_yield_image_file: str = "dfa_yield.png", dfa_shear_yield_image_file: str = "dfa_yield.png", dfa_params_file: str = "dfa_params.txt"):

    dfa_params = np.loadtxt(dfa_params_file)

    with doc.create(pl.Subsection("Deshpande-Fleck-Ashby yield surface identification", numbering=False)):
        with doc.create(pl.Figure(position="H")) as dfa_1122_fig:
            dfa_1122_fig.add_image(pl.NoEscape(dfa_yield_image_file), width=pl.NoEscape(r"0.8\textwidth"))
            dfa_1122_fig.add_caption(
                "Simulated yield surface and identified Deshpande-Fleck-Ashby yield surface in the (S11,S22) plane")

        with doc.create(pl.Figure(position="H")) as dfa_1112_fig:
            dfa_1112_fig.add_image(pl.NoEscape(dfa_shear_yield_image_file), width=pl.NoEscape(r"0.8\textwidth"))
            dfa_1112_fig.add_caption("Simulated yield surface and identified Deshpande-Fleck-Ashby yield surface in the (S11,S12) plane")

        with doc.create(pl.Table(position="H")) as dfa_tab:
            with doc.create(pl.Tabular("|c|c|c|c|c|c|c|")) as dfa_table:
                dfa_table.add_hline()
                dfa_table.add_row(("F", "G", "H", "L", "M", "N", "K"))
                dfa_table.add_hline()
                dfa_table.add_row((np.round(dfa_params[0], 2), np.round(dfa_params[1], 2), np.round(dfa_params[2], 2),
                                   np.round(dfa_params[3], 2), np.round(dfa_params[4], 2), np.round(dfa_params[5], 2),
                                   np.round(dfa_params[6], 2)))
                dfa_table.add_hline()
            dfa_tab.add_caption("Identified Deshpande-Fleck-Ashby yield surface parameters")


def generate_identification_subsection(doc: pl.document.Document, identification_graph_image_file: str,
                                       homogenized_law_parameters_file: str, young_modulus_file: str,
                                       shear_modulus_file: str):

    bulk_young_modulus = 147.7
    young_modulus = np.loadtxt(young_modulus_file)*bulk_young_modulus/1000.0
    shear_modulus = np.loadtxt(shear_modulus_file)*bulk_young_modulus/1000.0
    homogenized_law_params = np.loadtxt(homogenized_law_parameters_file)

    with doc.create(pl.Subsection("Homogenized behavior law identification", numbering=False)):
        with doc.create(pl.Figure(position="H")) as ident_graph_pic:
            ident_graph_pic.add_image(pl.NoEscape(identification_graph_image_file), width=pl.NoEscape(r"0.8\textwidth"))
            ident_graph_pic.add_caption("Comparison between simulated behavior and identified non linear behavior law")

        with doc.create(pl.Table(position="H")) as law_param_tab:
            with doc.create(pl.Tabular("|c|c|c|c|c|c|c|c|c|c|")) as law_param_table:
                law_param_table.add_hline()
                law_param_table.add_row(("E", pl.Math(data=pl.NoEscape(r"\nu"), inline=True), "G", pl.Math(data=pl.NoEscape(r"\alpha"), inline=True), "Q", "b", pl.Math(data=pl.NoEscape("C_{1}"), inline=True), pl.Math(data=pl.NoEscape("D_{1}"), inline=True), pl.Math(data=pl.NoEscape("C_{2}"), inline=True), pl.Math(data=pl.NoEscape("D_{2}"), inline=True)))
                law_param_table.add_hline()
                law_param_table.add_row((np.round(young_modulus), 0.3, np.round(shear_modulus), 1e-6,
                                         np.round(homogenized_law_params[0], 2), np.round(homogenized_law_params[1], 2),
                                         np.round(homogenized_law_params[2], 2), np.round(homogenized_law_params[3], 2),
                                         np.round(homogenized_law_params[4], 2), np.round(homogenized_law_params[5], 2)))
                law_param_table.add_hline()
            law_param_tab.add_caption("Homogenized law parameters")
