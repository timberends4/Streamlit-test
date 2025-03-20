import pandas as pd
import streamlit as st
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpInteger, LpStatus, value
from solver import calculate_priority_and_constraints, solver
import json
import locale

locale.setlocale(locale.LC_ALL, 'nl_NL.UTF-8')

########################################
# 1) Export & Import functions
########################################

def export_model_tojson(df_wonen, df_wonen_parkeren, df_nwonen_comm, df_nwonen_maat, df_parkeren_type, df_constraint, df_constraints_perc, df_kosten):
    default_filename = "optiPlan_input_export.json"
    filename = st.text_input(
        "Geef een bestandsnaam (JSON) om te exporteren:",
        value=default_filename
    )
    
    data_to_export = {
        "df_wonen": df_wonen.to_dict(orient="records"),
        "df_wonen_parkeren": df_wonen_parkeren.to_dict(orient="records"),
        "df_nwonen_comm": df_nwonen_comm.to_dict(orient="records"),
        "df_nwonen_maat": df_nwonen_maat.to_dict(orient="records"),
        "df_parkeren_type": df_parkeren_type.to_dict(orient="records"),
        "df_constraint": df_constraint.to_dict(orient="records"),
        "df_constraints_perc": df_constraints_perc.to_dict(orient="records"),
        "df_kosten": df_kosten.to_dict(orient="records"),
    }
    
    json_data = json.dumps(data_to_export, indent=4, )
    download_clicked = st.download_button(
        label="Download JSON",
        data=json_data,
        file_name=filename,
        mime="application/json"
    )
    if download_clicked:
        st.success(f"Bestand '{filename}' is gedownload.")


def import_model_fromjson(json_file):
    if not json_file:
        st.error("Please upload a JSON file.")
        return

    data = json.load(json_file)

    def overwrite_in_place(session_df: pd.DataFrame, new_records: list):
        """Overwrite `session_df` rows/columns in place with `new_records` (list of dicts)."""
        new_df = pd.DataFrame(new_records, columns=session_df.columns)
        session_df.drop(session_df.index, inplace=True)  # drop rows
        for col in new_df.columns:
            session_df[col] = new_df[col].values

    overwrite_in_place(st.session_state.df_wonen,            data["df_wonen"])
    overwrite_in_place(st.session_state.df_wonen_parkeren,   data["df_wonen_parkeren"])
    overwrite_in_place(st.session_state.df_nwonen_comm,      data["df_nwonen_comm"])
    overwrite_in_place(st.session_state.df_nwonen_maat,      data["df_nwonen_maat"])
    overwrite_in_place(st.session_state.df_parkeren_type,    data["df_parkeren_type"])
    overwrite_in_place(st.session_state.df_constraint,       data["df_constraint"])
    overwrite_in_place(st.session_state.df_constraints_perc, data["df_constraints_perc"])
    overwrite_in_place(st.session_state.df_kosten,           data["df_kosten"])

    st.success("Data geïmporteerd")
    st.rerun()  # Immediately re-run so imported data appears on the next run


########################################
# 2) Session State, Column Definitions
########################################

if "run_model_clicked" not in st.session_state:
    st.session_state.run_model_clicked = False

columns_wonen = [
    "Type", "Beschrijving", "GO (m²)", "Vormfactor", "BVO (m²)", "Minimaal aantal bouwlagen", "Variabel aantal bouwlagen (voor gestapelde bouw)",
    "Maximaal aantal bouwlagen", "Percentage bebouwd perceel (%)", "Totaal perceel (uitgeefbaar gebied) (m²)", "Gemiddelde residuele grondwaarde per woning (€)",
    "Residuele grondwaarde per m² uitgeefbaar (€ / m²)" , "Infrastructuur benodigd in openbaar gebied per woning (excl. parkeren) (m²)" ,"Minimaal aantal van type woning",
    "Minimaal aantal woningen", "Maximaal aantal woningen", "Sociale woning", "Betaalbaar/ Middenduur",
    "Overig type woning"
]
columns_wonen_parkeren = [
    "Type", "Parkeernorm bewoners", "Parkeernorm bezoekers", "Totale norm",
    "Korting op norm bewoners", "Korting op norm bezoekers", 
    "Parkeernorm voor model", "Parkeren op eigen terrein"
]
columns_kosten = ["Kosten", "Eenheid", "Waarde"]
columns_nwonen_comm = [
    "Type", "Beschrijving", "Uitgeefbaar", "Residuele grondwaarde m2 uitgeefbaar",
    "Vormfactor", "Percentage bebouwd perceel (%)", "Lagen", "Min opp m2", "Max opp m2",
    "Benodigde infra", "PP per 100m2 BVO", "Benodigde PP per m2",
]
columns_nwonen_maat = [
    "Type", "Beschrijving", "Uitgeefbaar", "Residuele grondwaarde m2 uitgeefbaar",
    "Vormfactor", "Percentage bebouwd perceel (%)", "Lagen", "Min opp m2", "Max opp m2",
    "Benodigde infra", "PP per 100m2 BVO", "Benodigde PP per m2",
]
columns_parkeren = [
    "Type", "Opp m2 BVO per P.P.", "Max aantal lagen", "Res. Grondwaarde per P.P.",
    "Verminderende Factor Grondwaarde per laag"
]
columns_parkeren_output = [
    "Type", "Aantal P.P", "Aantal lagen", "Totaal BVO",
    "Totaal res. grondwaarde", "Footprint"
]
columns_wonen_output = [
    "Type", "Aantal", "Lagen", "Uitgeefbaar m2", "Grondwaarde",
]
columns_niet_wonen_output = [
    "Type", "Aantal", "Uitgeefbaar m2", "Grondwaarde"
]

data_constraint = {
    "Constraint": [
        "Oppervlakte gebied", "Sociaal", "Betaalbaar/ Middenhuur", "Woningen", "Niet-wonen",
        "Niet-wonen (commercieel)", "Grondwaarde", "Groennorm", "Waternorm", "Groen", "Water"
    ],
    "Eenheid": [
        "m2", "Aantallen", "Aantallen", "Aantallen", "m2", 
        "m2", "Euro", "m2/ woning", "m2/ woning", "m2", "m2"
    ],
    "Min": [None for _ in range(11)],
    "Max": [None for _ in range(11)],
}
data_constraints_perc = {
    "Constraint": ["Sociaal perc.", "Betaalbaar/ Middenhuur perc."],
    "Eenheid": ["Percentage", "Percentage"],
    "Min": [None for _ in range(2)],
    "Max": [None for _ in range(2)],
}
data_output_grond = {
    "Output": [
        "Totaal opp. Gebied","Woningen totaal","Groen","Water","Infra",
        "Uitgeefbaar","Groennorm","Waternorm"
    ],
    "Eenheid": [
        "m2","aantal","m2","m2","m2","m2","m2/ woning","m2/ woning"
    ],
    "Hoeveelheid": ["","","","","","","",""]
}
data_kosten = {
    "Kosten": [
        "Bouwrijp maken gebied","Kosten Groen","Kosten Infrastructuur","Kosten Water"
    ],
    "Eenheid": ["Euro/ m2","Euro/ m2","Euro/ m2","Euro/ m2"],
    "Waarde": [None,None,None,None]
}


def enforce_numeric_columns(df, exclude_columns):
    df = df.copy()
    for col in df.columns:
        if col not in exclude_columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace(".", "")
            df[col] = df[col].str.replace(",", ".")
            df[col] = df[col].str.replace("€", "")
            df[col] = df[col].str.replace("%", "")
            df[col] = df[col].str.replace("m2", "")
            df[col] = df[col].str.replace("m", "")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

exclude_columns = [
    "Type","Beschrijving","Eenheid","Kosten", "Constraint",
    "Korting op norm bewoners","Korting op norm bezoekers"
]


########################################
# 3) Initialize session-state DataFrames
########################################

if "df_wonen" not in st.session_state:
    st.session_state.df_wonen = pd.DataFrame(columns=columns_wonen)
if "df_wonen_parkeren" not in st.session_state:
    st.session_state.df_wonen_parkeren = pd.DataFrame(columns=columns_wonen_parkeren)
if "df_nwonen_comm" not in st.session_state:
    st.session_state.df_nwonen_comm = pd.DataFrame(columns=columns_nwonen_comm)
if "df_nwonen_maat" not in st.session_state:
    st.session_state.df_nwonen_maat = pd.DataFrame(columns=columns_nwonen_maat)
if "df_parkeren_type" not in st.session_state:
    st.session_state.df_parkeren_type = pd.DataFrame(columns=columns_parkeren)
if "df_constraint" not in st.session_state:
    st.session_state.df_constraint = pd.DataFrame(data_constraint)
if "df_constraints_perc" not in st.session_state:
    st.session_state.df_constraints_perc = pd.DataFrame(data_constraints_perc)
if "df_kosten" not in st.session_state:
    st.session_state.df_kosten = pd.DataFrame(data_kosten)

# Model results
if "df_wonen_output_data" not in st.session_state:
    st.session_state.df_wonen_output_data = pd.DataFrame(columns=columns_wonen_output)
if "df_nwonen_output_comm_data" not in st.session_state:
    st.session_state.df_nwonen_output_comm_data = pd.DataFrame(columns=columns_niet_wonen_output)
if "df_nwonen_output_maat_data" not in st.session_state:
    st.session_state.df_nwonen_output_maat_data = pd.DataFrame(columns=columns_niet_wonen_output)
if "df_grond_output_data" not in st.session_state:
    st.session_state.df_grond_output_data = pd.DataFrame(data_output_grond)
if "df_parkeren_output_data" not in st.session_state:
    st.session_state.df_parkeren_output_data = pd.DataFrame(columns=columns_parkeren_output)

########################################
# 4) on_change Callbacks (force DataFrame)
########################################
def apply_data_editor_delta(old_df: pd.DataFrame, delta: dict) -> pd.DataFrame:
    """
    Merge partial changes into old_df. The delta dict might look like:
      {
        "edited_rows": { 0: {"BVO": 200}, 2: {"Sociaal": True} },
        "added_rows":  [ {...}, {...} ],
        "deleted_rows": [1,3]
      }
    """
    new_df = old_df.copy()

    edited = delta.get("edited_rows", {})
    added = delta.get("added_rows", [])
    deleted = delta.get("deleted_rows", [])

    # 1) Edited rows
    for row_idx, changes in edited.items():
        for col, val in changes.items():
            new_df.at[row_idx, col] = val

    # 2) Added rows
    for row_data in added:
        # If columns missing, fill them with e.g. None
        row_to_add = {}
        for col in new_df.columns:
            row_to_add[col] = row_data.get(col, None)
        new_df.loc[len(new_df)] = row_to_add

    # 3) Deleted rows
    for d_idx in sorted(deleted, reverse=True):
        if d_idx in new_df.index:
            new_df.drop(d_idx, inplace=True)

    new_df.reset_index(drop=True, inplace=True)
    return new_df

def format_nl_numbers(number, col_name, factor_columns, int_columns):
    # try:
    if pd.isna(number):
        return number
    number = parse_nl_number_str(number, col_name, factor_columns, int_columns)
    number_string = "{:,}".format(number).replace(",", "TEMP").replace(".", ",").replace("TEMP", ".")
    if number_string.endswith(",0"):
        number_string = number_string[:-2]
    # except:
    #     st.error(f"Error formatting number: {number}")
    return number_string

def parse_nl_number_str(number_str, col_name, factor_columns, int_columns):
    if isinstance(number_str, str):
        number_str = number_str.replace('.', '').replace(',', '.')
        if (col_name in factor_columns) and (float(number_str) < 0 or float(number_str) > 1):
            st.error(f"Invalid value for {col_name}: {number_str}")
            return 0
        if col_name in int_columns:
            return int(float(number_str))
        return float(number_str)
    return number_str

def on_change_generic(editor_key, session_key, columns_to_format = None, factor_columns=None, int_columns=None):
    delta = st.session_state[editor_key]
    old_df = st.session_state[session_key]
    merged_df = apply_data_editor_delta(old_df, delta)
    st.session_state[session_key] = merged_df

    if columns_to_format:
        for col in columns_to_format:
            merged_df[col] = merged_df[col].apply(format_nl_numbers, col_name=col, factor_columns=factor_columns, int_columns=int_columns)
    

    # st.rerun()


########################################
# 5) Streamlit UI
########################################
st.set_page_config(layout="wide", page_title="OptiPlan")
with st.container():
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        st.image("RYSE-logo.jpg", width=100)

whitelines = 2

columns_format_wonen = ["GO (m²)", "Vormfactor", "BVO (m²)", "Minimaal aantal bouwlagen", "Totaal perceel (uitgeefbaar gebied) (m²)", "Gemiddelde residuele grondwaarde per woning (€)", "Residuele grondwaarde per m² uitgeefbaar (€ / m²)", 
                        "Infrastructuur benodigd in openbaar gebied per woning (excl. parkeren) (m²)", "Minimaal aantal van type woning", "Minimaal aantal woningen", "Maximaal aantal woningen"]
disabled_columns_wonen = ["BVO (m²)", "Totaal perceel (uitgeefbaar gebied) (m²)", "Residuele grondwaarde per m² uitgeefbaar (€ / m²)"]
factor_columns_wonen = ["Vormfactor"]
int_columns_wonen = ["Minimaal aantal woningen", "Maximaal aantal woningen"]
# ——— Wonen Typologieën Editor ———
st.write("🏠 **Wonen Typologieën**")

st.data_editor(
    st.session_state.df_wonen,
    num_rows='dynamic',
    use_container_width=True,
    hide_index=True,
    key="df_wonen_editor",
    # on_change=lambda: on_change_generic("df_wonen_editor", "df_wonen", columns_format_wonen, factor_columns_wonen, int_columns_wonen),
    column_config={
        "Type": st.column_config.Column("Type", help="Enter the type of housing.", pinned=True),
        "Sociale woning": st.column_config.CheckboxColumn("Sociale woning", default=False),
        "Betaalbaar/ Middenduur": st.column_config.CheckboxColumn("Betaalbaar/ Middenduur", default=False),
        "Overig type woning": st.column_config.CheckboxColumn("Overig type woning", default=False),
        "Variabel aantal bouwlagen (voor gestapelde bouw)": st.column_config.CheckboxColumn("Variabel aant. lagen", default=False),   
        "Maximaal aantal bouwlagen": st.column_config.NumberColumn("Maximaal aantal bouwlagen", format="localized"),
        "Percentage bebouwd perceel (%)": st.column_config.NumberColumn("Percentage bebouwd perceel (%)", format="%.2f %%", step=1, min_value=0, max_value=100),
        "BVO (m²)": st.column_config.Column("BVO (m²)", disabled=True),
        "Totaal perceel (uitgeefbaar gebied) (m²)": st.column_config.Column("Totaal perceel (uitgeefbaar gebied) (m²)", disabled=True),
        "Residuele grondwaarde per m² uitgeefbaar (€ / m²)": st.column_config.Column("Residuele grondwaarde per m² uitgeefbaar (€ / m²)", disabled=True, width="small"),
    },
)
df_wonen = st.session_state.df_wonen

st.write("🏠🅿️ **Parkeersituatie Wonen**")
# Reindex first
st.session_state.df_wonen_parkeren = st.session_state.df_wonen_parkeren.reindex(df_wonen.index)
st.session_state.df_wonen_parkeren["Type"] = df_wonen["Type"]
st.session_state.df_wonen_parkeren.reset_index(drop=True, inplace=True)

st.data_editor(
    st.session_state.df_wonen_parkeren,
    num_rows='dynamic',
    use_container_width=True,
    hide_index=True,
    key="df_wonen_parkeren_editor",
    on_change=lambda: on_change_generic("df_wonen_parkeren_editor", "df_wonen_parkeren"),
    disabled=["Type", "Beschrijving"],
    column_config={
        "Type": st.column_config.Column("Type", help="Enter the type of housing.", pinned=True),
    }
)
df_wonen_parkeren = st.session_state.df_wonen_parkeren

# ——— Niet Wonen Commercieel ———
st.write("🏪 **Commercieel Typologieën**")
st.data_editor(
    st.session_state.df_nwonen_comm,
    num_rows='dynamic',
    use_container_width=True,
    hide_index=True,
    key="df_nwonen_comm_editor",
    on_change=lambda: on_change_generic("df_nwonen_comm_editor", "df_nwonen_comm"),
    column_config={
        "Type": st.column_config.Column("Type", help="Enter the type of commercial building.", pinned=True),
        "Residuele grondwaarde m2 uitgeefbaar": st.column_config.NumberColumn(
            "Residuele grondwaarde m2 uitgeefbaar", format="€ %d", step=1
        ),
    }
)
df_nwonen_comm = st.session_state.df_nwonen_comm

# ——— Niet Wonen Maatschappelijk ———
st.write("🏪 **Maatschappelijk Typologieën**")
st.data_editor(
    st.session_state.df_nwonen_maat,
    num_rows='dynamic',
    use_container_width=True,
    hide_index=True,
    key="df_nwonen_maat_editor",
    on_change=lambda: on_change_generic("df_nwonen_maat_editor", "df_nwonen_maat"),
    column_config={
        "Type": st.column_config.Column("Type", help="Enter the type of social building.", pinned=True),
        "Residuele grondwaarde m2 uitgeefbaar": st.column_config.NumberColumn(
            "Residuele grondwaarde m2 uitgeefbaar", format="€ %d", step=1
        ),
    }
)
df_nwonen_maat = st.session_state.df_nwonen_maat

# ——— Parkeren Typologieën ———
st.write("🅿️ **Parkeren Typologieën**")
st.data_editor(
    st.session_state.df_parkeren_type,
    num_rows='dynamic',
    use_container_width=True,
    hide_index=True,
    key="df_parkeren_type_editor",
    on_change=lambda: on_change_generic("df_parkeren_type_editor", "df_parkeren_type"),
    column_config={
        "Type": st.column_config.Column("Type", help="Enter the type of parking.", pinned=True),
    }
)
df_parkeren_type = st.session_state.df_parkeren_type

for _ in range(whitelines):
    st.write("")

if 'last_file_imported' not in st.session_state:
    st.session_state['last_file_imported'] = None

########################
# Constraints & Kosten
########################
with st.container():
    col1, col2 = st.columns([0.5, 0.5])

    
    with col1:
        st.subheader("Randvoorwaarden")
        st.data_editor(
            st.session_state.df_constraint,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            disabled=["Constraint", "Eenheid"],
            key="df_constraints_editor",
            on_change=lambda: on_change_generic("df_constraints_editor", "df_constraint", ["Min", "Max"]),            
        )
        df_constraint = st.session_state.df_constraint

        st.data_editor(
            st.session_state.df_constraints_perc,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            disabled=["Constraint", "Eenheid"],
            key="df_constraints_perc_editor",
            on_change=lambda: on_change_generic("df_constraints_perc_editor", "df_constraints_perc"),
            column_config={
                "Min": st.column_config.NumberColumn("Minimum Value", help="Enter the minimum value as a percentage.",
                                                     min_value=0, max_value=100, step=1, format="%.2f %%"),
                "Max": st.column_config.NumberColumn("Maximum Value", help="Enter the maximum value as a percentage.",
                                                     min_value=0, max_value=100, step=1, format="%.2f %%")
            },
        )
        df_constraints_perc = st.session_state.df_constraints_perc

    with col2:
        button_col = st.columns([1, 4, 1])
        with button_col[1]:
            st.subheader("Model")
            st.data_editor(
                st.session_state.df_kosten,
                num_rows='fixed',
                use_container_width=True,
                hide_index=True,
                key="df_kosten_editor",
                on_change=lambda: on_change_generic("df_kosten_editor", "df_kosten"),
                column_config={
                    "Waarde": st.column_config.NumberColumn("Kosten", help="Enter the cost type.", format="€ %d", step=1),
                }
            )
            df_kosten = st.session_state.df_kosten

            def helper_toggle_model():
                st.session_state.run_model_clicked = False

            st.selectbox(
                "Optimaliseer eerst voor:",
                ["Grondwaarde", "Uitgeefbaar", "Woonprogramma", "Betaalbaar en sociaal woonprogramma"],
                key="optimize_for",
                on_change=helper_toggle_model
            )

            st.selectbox(
                "Optimaliseer dan voor:",
                ["Geen", "Grondwaarde", "Uitgeefbaar", "Woonprogramma",
                 "Betaalbaar en sociaal woonprogramma"],
                key="optimize_for_2",
                on_change=helper_toggle_model
            )
            
            if st.button('Model Uitvoeren', use_container_width=True):
                st.write("Model wordt uitgevoerd...")

                # Convert checkbox booleans to int for solver
                df_wonen["Sociaal"] = df_wonen["Sociaal"].astype(int)
                df_wonen["Variabel aant. lagen"] = df_wonen["Variabel aant. lagen"].astype(int)
                df_wonen["Betaalbaar/ Middenduur"] = df_wonen["Betaalbaar/ Middenduur"].astype(int)
                
                df_wonen_copy = enforce_numeric_columns(df_wonen.copy(), exclude_columns)
                df_nwonen_comm_copy = enforce_numeric_columns(df_nwonen_comm.copy(), exclude_columns)
                df_nwonen_maat_copy = enforce_numeric_columns(df_nwonen_maat.copy(), exclude_columns)
                df_wonen_parkeren_copy = enforce_numeric_columns(df_wonen_parkeren.copy(), exclude_columns)
                df_parkeren_type_copy = enforce_numeric_columns(df_parkeren_type.copy(), exclude_columns)
                df_kosten_copy = enforce_numeric_columns(df_kosten.copy(), exclude_columns)
                df_constraint = enforce_numeric_columns(df_constraint.copy(), exclude_columns)
                df_constraints_perc = enforce_numeric_columns(df_constraints_perc.copy(), exclude_columns)

                # Run solver
                (df_wonen_out,
                 df_nwonen_comm_out,
                 df_nwonen_maat_out,
                 df_ruimte_out,
                 df_parkeren_out) = solver(
                    df_wonen=df_wonen_copy,
                    df_nwonen_comm=df_nwonen_comm_copy,
                    df_nwonen_maat=df_nwonen_maat_copy,
                    df_wonen_parkeren=df_wonen_parkeren_copy,
                    df_parkeren_type=df_parkeren_type_copy,
                    df_constraints=df_constraint,
                    df_constraints_perc=df_constraints_perc,
                    df_kosten=df_kosten_copy,
                    optimize_for=st.session_state.optimize_for,
                    optimize_for_2=st.session_state.optimize_for_2
                )

                # Store solver results
                st.session_state.df_wonen_output_data = df_wonen_out
                st.session_state.df_nwonen_output_comm_data = df_nwonen_comm_out
                st.session_state.df_nwonen_output_maat_data = df_nwonen_maat_out
                st.session_state.df_grond_output_data = df_ruimte_out
                st.session_state.df_parkeren_output_data = df_parkeren_out

                st.session_state.run_model_clicked = True
            
            if st.session_state.get("run_model_clicked", False):
                st.success("Model succesvol afgerond!")

            uploaded_json_file = st.file_uploader('uploader-input', type="json", label_visibility="hidden")
            if uploaded_json_file is not None and st.session_state.last_file_imported != uploaded_json_file:
                st.session_state.last_file_imported = uploaded_json_file
                import_model_fromjson(uploaded_json_file)
                

            if st.button('Model Exporteren', use_container_width=True):
                export_model_tojson(
                    df_wonen,
                    df_wonen_parkeren,
                    df_nwonen_comm,
                    df_nwonen_maat,
                    df_parkeren_type,
                    df_constraint,
                    df_constraints_perc,
                    df_kosten
                )

#############################################
# 6) Dynamically update output DataFrames
#############################################
def update_output_df(input_df, output_columns, type_column='Type'):
    if type_column in input_df.columns:
        unique_types = input_df[type_column].dropna().unique()
        new_output_df = pd.DataFrame({type_column: unique_types})
        for col in output_columns:
            if col != type_column:
                new_output_df[col] = ""
        return new_output_df
    else:
        return pd.DataFrame(columns=output_columns)

df_wonen_output = update_output_df(st.session_state.df_wonen, columns_wonen_output, type_column='Type')
df_nwonen_output_comm = update_output_df(st.session_state.df_nwonen_comm, columns_niet_wonen_output, type_column='Type')
df_nwonen_output_maat = update_output_df(st.session_state.df_nwonen_maat, columns_niet_wonen_output, type_column='Type')
df_parkeren_output = update_output_df(st.session_state.df_parkeren_type, columns_parkeren_output, type_column='Type')

#############################################
# 7) Display Output DataFrames
#############################################
with st.container():
    st.subheader("Resultaten van het Model")
    col1, col2 = st.columns([0.5, 0.5])
    
    with col1:
        st.write("🏠 **Wonen Typologieën Output**")
        st.data_editor(
            st.session_state.df_wonen_output_data,
            use_container_width=True,
            hide_index=True,
            disabled=False,
            key="df_wonen_output_editor"
        )
    
    with col2:
        st.write("🏠🌳🟦⬜ **Totaal**")
        st.data_editor(
            st.session_state.df_grond_output_data,
            use_container_width=True,
            hide_index=True,
            disabled=False,
            key="df_grond_output_editor"
        )

with st.container():
    col1, col2 = st.columns([0.5, 0.5])
    
    with col1:
        st.write("🏪 **Niet Wonen Commercieel Output**")
        st.data_editor(
            st.session_state.df_nwonen_output_comm_data,
            use_container_width=True,
            hide_index=True,
            disabled=False,
            key="df_nwonen_output_comm_editor2"
        )

        st.write("🏪 **Niet Wonen Maatschappelijk Output**")
        st.data_editor(
            st.session_state.df_nwonen_output_maat_data,
            use_container_width=True,
            hide_index=True,
            disabled=False,
            key="df_nwonen_output_maat_editor2"
        )

    with col2:
        st.write("🅿️ **Parkeren Typologieën Output**")
        st.data_editor(
            st.session_state.df_parkeren_output_data,
            use_container_width=True,
            hide_index=True,
            disabled=False,
            key="df_parkeren_output_editor2",
        )
