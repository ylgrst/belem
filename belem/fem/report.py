import pylatex as pl
import numpy as np
from simcoon import simmit as sim

def import_extra_packages(doc: pl.document.Document):
    doc.packages.append(pl.Package('float'))
    doc.packages.append(pl.Package('titlepic'))
    doc.packages.append(pl.Package('babel', options="english"))
    doc.packages.append(pl.Package('geometry', options=pl.base_classes.command.Options("a4paper", bindingoffset="0.2in", left="1in",
                                                                  right="1in", top="1in", bottom="1in",
                                                                  footskip="0.25in")))
    doc.preamble.append(pl.NoEscape(r"\renewcommand{\thefigure}{\hspace{-.333333em}}"))
    doc.preamble.append(pl.NoEscape(r"\renewcommand{\thetable}{\hspace{-.333333em}}"))

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

def generate_chapter_page(doc: pl.document.Document, shape: str, shape_image: str):
    with doc.create(pl.Chapter(shape, numbering=False)):

        with doc.create(pl.Figure(position="H")) as intro_pic:
            intro_pic.add_image(pl.NoEscape(shape_image), width=pl.NoEscape(r"0.9\textwidth"))

    doc.append(pl.NewPage())

def generate_section(doc: pl.document.Document, basedir: str, shape: str, density: float,response_curve_file_name: str,
                     hardening_curve_file_name: str, dfa_yield_image_file: str, dfa_shear_yield_image_file: str,
                     dfa_params_file: str, identification_graph_image_file: str, error_graph_image_file: str,
                     homogenized_law_parameters_file: str, young_modulus_file: str, shear_modulus_file: str):
    with doc.create(pl.Section(shape + " " + str(int(100*density)) + "%", numbering=False)):
        #generate_response_curves_subsection(doc, response_curve_file_name)
        #generate_hardening_curves_subsection(doc, hardening_curve_file_name)
        generate_fea_subsection(doc, response_curve_file_name, hardening_curve_file_name, basedir + shape + "/density" + str(int(100*density)) + "/")
        generate_dfa_subsection(doc, dfa_yield_image_file, dfa_shear_yield_image_file, dfa_params_file)
        #generate_identification_subsection(doc, identification_graph_image_file, error_graph_image_file, homogenized_law_parameters_file, young_modulus_file, shear_modulus_file)
        generate_chg_identification_subsection(doc, identification_graph_image_file, error_graph_image_file,
                                           homogenized_law_parameters_file, young_modulus_file, shear_modulus_file)
    doc.append(pl.NewPage())

def generate_response_curves_subsection(doc: pl.document.Document, response_curve_file_name: str = "all_vm_stress_vm_strain.png"):
    with doc.create(pl.Subsection("Simulated response curves", numbering=False)):
        with doc.create(pl.Figure(position="H")) as resp_fig:
            resp_fig.add_image(pl.NoEscape(response_curve_file_name), width=pl.NoEscape(r"0.8\textwidth"))
            resp_fig.add_caption("Mises stress vs mises strain for different load cases")

def generate_hardening_curves_subsection(doc: pl.document.Document, hardening_curve_file_name: str = "all_vm_hardening.png"):
    with doc.create(pl.Subsection("Simulated hardening curves", numbering=False)):
        with doc.create(pl.Figure(position="H")) as hard_fig:
            hard_fig.add_image(pl.NoEscape(hardening_curve_file_name), width=pl.NoEscape(r"0.8\textwidth"))
            hard_fig.add_caption("Mises stress vs plastic mises strain for different load cases")

def generate_fea_subsection(doc: pl.document.Document, response_curve_file_name: str, hardening_curve_file_name: str, shape_basedir: str):
    fea_data = {}
    sim_list = ["tension", "biaxial_tension", "compression", "biaxial_compression", "tencomp", "shear"]
    for sim in sim_list:
        sigma_vm_local_max = np.loadtxt(shape_basedir + sim + "/results/sigma_vm_max_local.txt")
        local_max_vm_plastic_strain = np.loadtxt(shape_basedir + sim + "/results/max_local_plastic_strain.txt")
        vm_local_global_pstrain_ratio = np.loadtxt(shape_basedir + sim + "/results/ratio_local_vs_global_max_ep.txt")
        fea_sim_data = {"sigma_vm_local_max":sigma_vm_local_max,
                             "local_max_vm_plastic_strain":local_max_vm_plastic_strain,
                             "vm_local_global_pstrain_ratio":vm_local_global_pstrain_ratio}
        fea_data[sim] = fea_sim_data

    with doc.create(pl.Subsection("Finite Element Analysis", numbering=False)):
        with doc.create(pl.Figure(position="H")) as resp_hard_fig:
            with doc.create(pl.SubFigure(position="t", width=pl.NoEscape(r"0.48\textwidth"))) as resp_fig:
                resp_fig.add_image(pl.NoEscape(response_curve_file_name), width=pl.NoEscape(r"\textwidth"))
                resp_fig.add_caption("Mises stress vs mises strain for different load cases")
            doc.append(pl.HFill())
            with doc.create(pl.SubFigure(position="t", width=pl.NoEscape(r"0.48\textwidth"))) as hard_fig:
                hard_fig.add_image(pl.NoEscape(hardening_curve_file_name), width=pl.NoEscape(r"\textwidth"))
                hard_fig.add_caption("Mises stress vs plastic mises strain for different load cases")
            resp_hard_fig.add_caption("Response and hardening curves for different load cases")

        with doc.create(pl.Table(position="H")) as fea_tab:
            with doc.create(pl.Tabular("|l|c|c|c|")) as fea_table:
                fea_table.add_hline()
                fea_table.add_row(("Load case", pl.Math(data=pl.NoEscape(r"\sigma^{max}_{{VM}_{local}} (MPa)"), inline=True),
                                   pl.Math(data=pl.NoEscape(r"\epsilon^{max}_{{VM}_{local}}"), inline=True),
                                   pl.Math(data=pl.NoEscape(r"\epsilon^{max}_{{VM}_{local}}/\epsilon^{max}_{{VM}_{global}}"), inline=True)))
                fea_table.add_hline()
                for sim in sim_list:
                    if sim == "tencomp":
                        simu = "tension-compression"
                    elif sim == "biaxial_tension":
                        simu = "biaxial tension"
                    elif sim == "biaxial_compression":
                        simu = "biaxial compression"
                    else:
                        simu = sim
                    fea_table.add_row((simu), np.round(fea_data[sim]["sigma_vm_local_max"], 2),
                                      np.round(fea_data[sim]["local_max_vm_plastic_strain"], 2),
                                      np.round(fea_data[sim]["vm_local_global_pstrain_ratio"], 2))
                fea_table.add_hline()
            fea_tab.add_caption("Local max stress and plastic strain data for different load cases")


def generate_stress_concentration_map_subsection(doc: pl.document.Document):
    ...

def generate_dfa_subsection(doc: pl.document.Document, dfa_yield_image_file: str = "dfa_yield.png", dfa_shear_yield_image_file: str = "dfa_yield_shear.png", dfa_params_file: str = "dfa_params.txt"):

    dfa_params = np.loadtxt(dfa_params_file)

    with doc.create(pl.Subsection("Deshpande-Fleck-Ashby yield surface identification", numbering=False)):
        with doc.create(pl.Figure(position="H")) as dfa_fig:
            with doc.create(pl.SubFigure(position="t", width=pl.NoEscape(r"0.48\textwidth"))) as dfa_1122_fig:
                dfa_1122_fig.add_image(pl.NoEscape(dfa_yield_image_file), width=pl.NoEscape(r"\textwidth"))
                dfa_1122_fig.add_caption(
                "Simulated yield surface and identified Deshpande-Fleck-Ashby yield surface in the (S11,S22) plane")
            doc.append(pl.HFill())
            with doc.create(pl.SubFigure(position="t", width=pl.NoEscape(r"0.48\textwidth"))) as dfa_1112_fig:
                dfa_1112_fig.add_image(pl.NoEscape(dfa_shear_yield_image_file), width=pl.NoEscape(r"\textwidth"))
                dfa_1112_fig.add_caption("Simulated yield surface and identified Deshpande-Fleck-Ashby yield surface in the (S11,S12) plane")
        dfa_fig.add_caption("Deshpande-Fleck-Ashby yield surface")

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
                                       error_graph_image_file: str,
                                       homogenized_law_parameters_file: str, stiffness_tensor_file: str):

    bulk_young_modulus = 197.7
    young_modulus, poisson_ratio, shear_modulus = sim.L_cubic_props(np.loadtxt(stiffness_tensor_file))
    young_modulus = young_modulus[0]*bulk_young_modulus/1000.0
    shear_modulus = shear_modulus[0]*bulk_young_modulus/1000.0
    homogenized_law_params = np.loadtxt(homogenized_law_parameters_file)

    with doc.create(pl.Subsection("Homogenized behavior law identification", numbering=False)):
        with doc.create(pl.Figure(position="H")) as ident_graph_pic:
            with doc.create(pl.SubFigure(position="t", width=pl.NoEscape(r"0.48\textwidth"))) as compare_ident_graph_pic:
                compare_ident_graph_pic.add_image(pl.NoEscape(identification_graph_image_file), width=pl.NoEscape(r"\textwidth"))
                compare_ident_graph_pic.add_caption("Comparison between simulated behavior and identified non linear behavior law")
            doc.append(pl.HFill())
            with doc.create(pl.SubFigure(position='t', width=pl.NoEscape(r"0.48\textwidth"))) as error_ident_graph_pic:
                error_ident_graph_pic.add_image(pl.NoEscape(error_graph_image_file), width=pl.NoEscape(r"\textwidth"))
                error_ident_graph_pic.add_caption("Normalized identification error (RMSE) with respect to time")
            ident_graph_pic.add_caption("Identified homogenized law compared to FE analysis")

        with doc.create(pl.Table(position="H")) as law_param_tab:
            with doc.create(pl.Tabular("|l|r|")) as law_param_table:
                law_param_table.add_hline()
                law_param_table.add_row(("Parameter", "Value"))
                law_param_table.add_hline()
                law_param_table.add_row(("E (GPa)", np.round(young_modulus, 2)))
                law_param_table.add_row((pl.Math(data=pl.NoEscape(r"\nu"), inline=True), np.round(poisson_ratio, 3)))
                law_param_table.add_row(("G (GPa)", np.round(shear_modulus,2)))
                law_param_table.add_row((pl.Math(data=pl.NoEscape(r"\alpha"), inline=True), 1e-6))
                law_param_table.add_row((pl.Math(data=pl.NoEscape(r"\sigma_Y (MPa)"), inline=True), np.round(homogenized_law_params[0], 2)))
                law_param_table.add_row(("Q (MPa)", np.round(homogenized_law_params[1], 2)))
                law_param_table.add_row(
                    ("b", np.round(homogenized_law_params[2], 2)))
                law_param_table.add_row(
                    (pl.Math(data=pl.NoEscape("C_{1} (MPa)"), inline=True), np.round(homogenized_law_params[3], 2)))
                law_param_table.add_row(
                    (pl.Math(data=pl.NoEscape("D_{1}"), inline=True), np.round(homogenized_law_params[4], 2)))
                law_param_table.add_row(
                    (pl.Math(data=pl.NoEscape("C_{2} (MPa)"), inline=True), np.round(homogenized_law_params[5], 2)))
                law_param_table.add_row(
                    (pl.Math(data=pl.NoEscape("D_{2}"), inline=True), np.round(homogenized_law_params[6], 2)))
                law_param_table.add_hline()
            law_param_tab.add_caption("Homogenized law parameters")

def generate_chg_identification_subsection(doc: pl.document.Document, identification_graph_image_file: str,
                                           error_graph_image_file: str,
                                       homogenized_law_parameters_file: str, stiffness_tensor_file: str):

    bulk_young_modulus = 197.7
    young_modulus, poisson_ratio, shear_modulus = sim.L_cubic_props(np.loadtxt(stiffness_tensor_file))
    young_modulus = young_modulus[0]*bulk_young_modulus/1000.0
    shear_modulus = shear_modulus[0]*bulk_young_modulus/1000.0
    homogenized_law_params = np.loadtxt(homogenized_law_parameters_file)

    with doc.create(pl.Subsection("Homogenized behavior law identification", numbering=False)):
        with doc.create(pl.Figure(position="H")) as ident_graph_pic:
            with doc.create(pl.SubFigure(position="t", width=pl.NoEscape(r"0.48\textwidth"))) as compare_ident_graph_pic:
                compare_ident_graph_pic.add_image(pl.NoEscape(identification_graph_image_file),
                                                  width=pl.NoEscape(r"\textwidth"))
                compare_ident_graph_pic.add_caption(
                    "Comparison between simulated behavior and identified non linear behavior law")
            doc.append(pl.HFill())
            with doc.create(pl.SubFigure(position='t', width=pl.NoEscape(r"0.48\textwidth"))) as error_ident_graph_pic:
                error_ident_graph_pic.add_image(pl.NoEscape(error_graph_image_file), width=pl.NoEscape(r"\textwidth"))
                error_ident_graph_pic.add_caption("Normalized identification error (RMSE) with respect to time")
            ident_graph_pic.add_caption("Identified homogenized law compared to FE analysis")

        with doc.create(pl.Table(position="H")) as law_param_tab:
            with doc.create(pl.Tabular("|l|r|")) as law_param_table:
                law_param_table.add_hline()
                law_param_table.add_row(("Parameter", "Value"))
                law_param_table.add_hline()
                law_param_table.add_row(("E (GPa)", np.round(young_modulus, 2)))
                law_param_table.add_row((pl.Math(data=pl.NoEscape(r"\nu"), inline=True), np.round(poisson_ratio, 3)))
                law_param_table.add_row(("G (GPa)", np.round(shear_modulus,2)))
                law_param_table.add_row((pl.Math(data=pl.NoEscape(r"\alpha"), inline=True), 1e-6))
                law_param_table.add_row(
                    (pl.Math(data=pl.NoEscape("\sigma_Y (MPa)"), inline=True), np.round(homogenized_law_params[0], 2)))
                law_param_table.add_row((pl.Math(data=pl.NoEscape("Q_{1} (MPa)"), inline=True), np.round(homogenized_law_params[1], 2)))
                law_param_table.add_row(
                    (pl.Math(data=pl.NoEscape("b_{1}"), inline=True), np.round(homogenized_law_params[2], 2)))
                law_param_table.add_row(
                    (pl.Math(data=pl.NoEscape("Q_{2} (MPa)"), inline=True), np.round(homogenized_law_params[3], 2)))
                law_param_table.add_row(
                    (pl.Math(data=pl.NoEscape("b_{2}"), inline=True), np.round(homogenized_law_params[4], 2)))
                law_param_table.add_row(
                    (pl.Math(data=pl.NoEscape("Q_{3} (MPa)"), inline=True), np.round(homogenized_law_params[5], 2)))
                law_param_table.add_row(
                    (pl.Math(data=pl.NoEscape("b_{3}"), inline=True), np.round(homogenized_law_params[6], 2)))
                law_param_table.add_row(
                    (pl.Math(data=pl.NoEscape("C_{1} (MPa)"), inline=True), np.round(homogenized_law_params[7], 2)))
                law_param_table.add_row(
                    (pl.Math(data=pl.NoEscape("D_{1}"), inline=True), np.round(homogenized_law_params[8], 2)))
                law_param_table.add_row(
                    (pl.Math(data=pl.NoEscape("C_{2} (MPa)"), inline=True), np.round(homogenized_law_params[9], 2)))
                law_param_table.add_row(
                    (pl.Math(data=pl.NoEscape("D_{2}"), inline=True), np.round(homogenized_law_params[10], 2)))
                law_param_table.add_hline()
            law_param_tab.add_caption("Homogenized law parameters")