import pandas as pd
import streamlit as st
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpInteger, LpStatus, value
from solver import solve
import json
import locale

# Set locale to Dutch
locale.setlocale(locale.LC_ALL, 'nl_NL.UTF-8')
########################################
# 1) Export & Import functions
########################################

def export_model_tojson(df_wonen, df_wonen_parkeren, df_nwonen_comm, df_nwonen_maat, df_parkeren_type, df_constraints_context, df_constraints_program_perc, df_constraints_program, df_constraints_ruimtelijk, df_constraints_financieel): 
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
        "df_constraints_context": df_constraints_context.to_dict(orient="records"),
        "df_constraints_program_perc": df_constraints_program_perc.to_dict(orient="records"),
        "df_constraints_program": df_constraints_program.to_dict(orient="records"),
        "df_constraints_ruimtelijk": df_constraints_ruimtelijk.to_dict(orient="records"),
        "df_constraints_financieel": df_constraints_financieel.to_dict(orient="records"),
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
    overwrite_in_place(st.session_state.df_constraints_context, data["df_constraints_context"])
    overwrite_in_place(st.session_state.df_constraints_program_perc, data["df_constraints_program_perc"])
    overwrite_in_place(st.session_state.df_constraints_program, data["df_constraints_program"])
    overwrite_in_place(st.session_state.df_constraints_ruimtelijk, data["df_constraints_ruimtelijk"])
    overwrite_in_place(st.session_state.df_constraints_financieel, data["df_constraints_financieel"])

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
    "Minimaal aantal woningen", "Maximaal aantal woningen", "Sociale woning", "Betaalbaar/ Middenduur woning",
    "Overig type woning"
]
columns_wonen_parkeren = [
    "Type", "Parkeernorm bewoners", "Parkeernorm bezoekers", "Totale norm",
    "Korting op norm bewoners", "Korting op norm bezoekers", 
    "Parkeernorm inclusief korting", "Waarvan parkeren op eigen terrein"
]
columns_nwonen = [
    "Type", "Beschrijving", "Minimaal VVO (m²)", "Maximaal VVO (m²)", "Vormfactor", "Aantal bouwlagen", "Percentage bebouwd perceel (%)", "Gemiddelde residuele grondwaarde per m² uitgeefbaar (€)",
    "Infrastructuur benodigd in openbaar gebied per m² VVO (excl parkeren) (m²)", "Parkeerplaatsen per 100 m² BVO"
]
columns_parkeren = [
    "Parkeer typologiën", "BVO per parkeerplaats (m²)", "Maximaal aantal lagen", "Gemiddelde residuele grondwaarde per parkeerplaats (€)",
    "Verminderende Factor Grondwaarde per laag"
]

data_constraints_context = {
    "Randvoorwaarden": ["Oppervlakte gebied", "Minimale totale grondwaarde", "Maximale totale grondwaarde"],
    "Gestelde eis": [None, None, None],
    "Eenheid": ["m²", "Euro", "Euro"],
}

data_constraints_program = {
    "Randvoorwaarden": [
        "Totaal aantal woningen","Aantal sociale woningen", "Aantal betaalbare/ middenhuur woningen", "Aantal m² commercieel vastgoed", "Aantal m² maatschappelijk vastgoed",
    ],
    "Min": [None for _ in range( 5)],
    "Max": [None for _ in range( 5)],
    "Eenheid": ["Aantal", "Aantal", "Aantal", "m²", "m²"],
}
data_constraints_program_perc = {
    "Randvoorwaarden": ["Percentage sociale woningen", "Percentage betaalbare/ middenhuur woningen"],
    "Min": [None for _ in range(2)],
    "Max": [None for _ in range(2)],
    "Eenheid": ["Percentage", "Percentage"],
}

data_constraints_ruimtelijk = {
    "Randvoorwaarden": ["Groennorm per woning", "Aantal m² groen", "Waternorm per woning", "Aantal m² water"],
    "Min": [None for _ in range(4)],
    "Max": [None for _ in range(4)],
    "Eenheid": ["m²/ woning", "m²", "m²/ woning", "m²"],
}

data_constraints_financieel = {
    "Randvoorwaarden": ["Bouwrijp maken gebied", "Kosten Groen", "Kosten Infrastructuur", "Kosten Water"],
    "Kosten": [None for _ in range(4)],
    "Eenheid": ["Euro/ m²", "Euro/ m²", "Euro/ m²", "Euro/ m²"],
}

data_results_context = {
    "Onderdeel": ["Oppervlakte gebied", "Totale residuele grondwaarde"],
    "Waarde": ["", ""],
    "Eenheid": ["m²", "Euro"],
}

data_results_ruimtelijk = {
    "Onderdeel": ["Totaal uitgeefbaar gebied", "Totaal oppervlak infrastructuur (exclusief parkeren)", "Totaal footprint parkeren", "Totaal oppervlak groen", "Totaal oppervlak water",
                  "Groennorm per woning", "Waternorm per woning"],
    "Output": ["", "", "", "", "", "", ""],
    "Eenheid": ["m²", "m²", "m²", "m²", "m²", "m²/ woning", "m²/ woning"],
}

data_results_program = {
    "Onderdeel": ["Totaal aantal woningen", "Aantal sociale woningen", "Aantal betaalbare/ middenhuur woningen", "Aantal m² commercieel vastgoed", "Aantal m² maatschappelijk vastgoed"],
    "Output": ["", "", "", "", ""],
    "Eenheid": ["Aantal", "Aantal", "Aantal", "m²", "m²"],
}
data_results_program_perc ={
    "Onderdeel": ["Percentage sociale woningen", "Percentage betaalbare/ middenhuur woningen"],
    "Output": ["", ""],
    "Eenheid": ["Percentage", "Percentage"],
}
data_results_parkeren = {
    "Onderdeel": ["Parkeerplaatsen in openbaar gebied t.b.v wonen",
                  "Parkeerplaatsen in openbaar gebied t.b.v commercieel vastgoed",
                  "Parkeerplaatsen in openbaar gebied t.b.v maatschappelijk vastgoed"],
    "Output": ["", "", ""],
    "Eenheid": ["Aantal", "Aantal", "Aantal"]
}

columns_invulling_wonen = ["Type", "Aantal", "Aantal lagen", "Totaal uitgeefbaar gebied (m²)", "Totale residuele grondwaarde (€)"]
columns_invulling_nwonen = ["Type", "Aantal", "VVO (m²)", "Totaal uitgeefbaar gebied (m²)", "Totale residuele grondwaarde (€)"]
columns_invulling_parkeren = ["Type", "Aantal parkeerplaatsen", "Totaal BVO (m²)", "Totale residuele grondwaarde (€)", "Aantal lagen", "Footprint (m²)"]


def enforce_numeric_columns(df, exclude_columns):
    df = df.copy()
    for col in df.columns:
        if col not in exclude_columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace("€", "")
            df[col] = df[col].str.replace("%", "")
            df[col] = df[col].str.replace("m²", "")
            df[col] = df[col].str.replace("m", "")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

exclude_columns = [
    "Type","Beschrijving","Eenheid","Kosten", "Constraint", "Gestelde eis",
    "Korting op norm bewoners","Korting op norm bezoekers",
    "Percentage bebouwd perceel (%)", "Sociale woning", "Betaalbaar/ Middenduur woning", "Overig type woning",
    "Variabel aantal bouwlagen (voor gestapelde bouw)", "Onderdeel"
]


########################################
# 3) Initialize session-state input DataFrames
########################################
if "df_wonen" not in st.session_state:
    st.session_state.df_wonen = pd.DataFrame(columns=columns_wonen)
if "df_wonen_parkeren" not in st.session_state:
    st.session_state.df_wonen_parkeren = pd.DataFrame(columns=columns_wonen_parkeren)
if "df_nwonen_comm" not in st.session_state:
    st.session_state.df_nwonen_comm = pd.DataFrame(columns=columns_nwonen)
if "df_nwonen_maat" not in st.session_state:
    st.session_state.df_nwonen_maat = pd.DataFrame(columns=columns_nwonen)
if "df_parkeren_type" not in st.session_state:
    st.session_state.df_parkeren_type = pd.DataFrame(columns=columns_parkeren)
if "df_constraints_context" not in st.session_state:
    st.session_state.df_constraints_context = pd.DataFrame(data_constraints_context)
if "df_constraints_program" not in st.session_state:
    st.session_state.df_constraints_program = pd.DataFrame(data_constraints_program)
if "df_constraints_program_perc" not in st.session_state:
    st.session_state.df_constraints_program_perc = pd.DataFrame(data_constraints_program_perc)
if "df_constraints_ruimtelijk" not in st.session_state:
    st.session_state.df_constraints_ruimtelijk = pd.DataFrame(data_constraints_ruimtelijk)
if "df_constraints_financieel" not in st.session_state:
    st.session_state.df_constraints_financieel = pd.DataFrame(data_constraints_financieel)

# Model results
if "df_results_context" not in st.session_state:
    st.session_state.df_results_context = pd.DataFrame(data_results_context)
if "df_results_program" not in st.session_state:
    st.session_state.df_results_program = pd.DataFrame(data_results_program)
if "df_results_program_perc" not in st.session_state:
    st.session_state.df_results_program_perc = pd.DataFrame(data_results_program_perc)
if "df_results_ruimtelijk" not in st.session_state:
    st.session_state.df_results_ruimtelijk = pd.DataFrame(data_results_ruimtelijk)
if "df_results_parkeren" not in st.session_state:
    st.session_state.df_results_parkeren = pd.DataFrame(data_results_parkeren)
if "df_wonen_invulling" not in st.session_state:
    st.session_state.df_wonen_invulling = pd.DataFrame(columns=columns_invulling_wonen)
if "df_nwonen_invulling_comm" not in st.session_state:
    st.session_state.df_nwonen_invulling_comm = pd.DataFrame(columns=columns_invulling_nwonen)
if "df_nwonen_invulling_maat" not in st.session_state:
    st.session_state.df_nwonen_invulling_maat = pd.DataFrame(columns=columns_invulling_nwonen)
if "df_results_invulling_parkeren" not in st.session_state:
    st.session_state.df_results_invulling_parkeren = pd.DataFrame(columns=columns_invulling_parkeren)

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

def on_change_generic(editor_key, session_key):
    delta = st.session_state[editor_key]
    old_df = st.session_state[session_key]
    merged_df = apply_data_editor_delta(old_df, delta)
    st.session_state[session_key] = merged_df
    return merged_df

def calculate_columns():
    try:
        on_change_generic("df_wonen_editor", "df_wonen")
        on_change_generic("df_wonen_parkeren_editor", "df_wonen_parkeren")

        if st.session_state.df_wonen["Vormfactor"].isnull().any():
            st.error("Vormfactor mag niet leeg zijn.")
        else:
            st.session_state.df_wonen["BVO (m²)"] = (st.session_state.df_wonen["GO (m²)"] / st.session_state.df_wonen["Vormfactor"]).round(0)

        if st.session_state.df_wonen["Minimaal aantal bouwlagen"].isnull().any() or st.session_state.df_wonen["Percentage bebouwd perceel (%)"].isnull().any():
            st.error("Minimaal aantal bouwlagen en/of Percentage bebouwd perceel mag niet leeg zijn.")
        else:
            st.session_state.df_wonen["Totaal perceel (uitgeefbaar gebied) (m²)"] = (st.session_state.df_wonen["BVO (m²)"] / st.session_state.df_wonen["Minimaal aantal bouwlagen"] / (st.session_state.df_wonen["Percentage bebouwd perceel (%)"] /100)).round(0)
        if st.session_state.df_wonen["Totaal perceel (uitgeefbaar gebied) (m²)"].isnull().any() or st.session_state.df_wonen["Gemiddelde residuele grondwaarde per woning (€)"].isnull().any():
            st.error("Totaal perceel en/of Gemiddelde residuele grondwaarde mag niet leeg zijn.")
        else:
            st.session_state.df_wonen["Residuele grondwaarde per m² uitgeefbaar (€ / m²)"] = (st.session_state.df_wonen["Gemiddelde residuele grondwaarde per woning (€)"] / st.session_state.df_wonen["Totaal perceel (uitgeefbaar gebied) (m²)"]).round(0)

        if st.session_state.df_wonen_parkeren["Korting op norm bewoners"].isnull().any() or st.session_state.df_wonen_parkeren["Korting op norm bezoekers"].isnull().any():
            st.error("Korting op norm bewoners en/of Korting op norm bezoekers mag niet leeg zijn.")
        else:
            st.session_state.df_wonen_parkeren["Totale norm"] = (
                pd.to_numeric(st.session_state.df_wonen_parkeren["Parkeernorm bewoners"], errors='raise') +
                pd.to_numeric(st.session_state.df_wonen_parkeren["Parkeernorm bezoekers"], errors='raise')
            ).round(2)

            st.session_state.df_wonen_parkeren["Parkeernorm inclusief korting"] = (
                pd.to_numeric(st.session_state.df_wonen_parkeren["Totale norm"], errors='raise') -
                pd.to_numeric(st.session_state.df_wonen_parkeren["Parkeernorm bewoners"] * st.session_state.df_wonen_parkeren["Korting op norm bewoners"] / 100, errors='raise') -
                pd.to_numeric(st.session_state.df_wonen_parkeren["Parkeernorm bezoekers"] * st.session_state.df_wonen_parkeren["Korting op norm bezoekers"] / 100, errors='raise')
            )
        
    except Exception as e:
        st.error(f"Error calculating columns: {e}")
    return
######################################
# 5) Streamlit UI
########################################
st.set_page_config(layout="wide", page_title="OptiPlan")
with st.container():
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        st.image("RYSE-logo.jpg", width=100, )

whitelines = 2

########################
# Constraints & Kosten
########################

st.subheader("De Opgave")

with st.container():
    col1, col2 = st.columns([0.5, 0.5])

    
    with col1:
        st.write("**Context**")
        st.data_editor(
            st.session_state.df_constraints_context,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            disabled=["Randvoorwaarden", "Eenheid"],
            key="df_constraint_context_editor",
            column_config={
                "Gestelde eis": st.column_config.NumberColumn("Gestelde eis", help="Enter the value.", format="localized", step=1),

            },
        )
        df_constraints_context = st.session_state.df_constraints_context

        st.write("**Programmitische randvoorwaarden**")
        st.data_editor(
            st.session_state.df_constraints_program,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            disabled=["Randvoorwaarden", "Eenheid"],
            key="df_constraints_program_editor",
            column_config={
                "Min": st.column_config.NumberColumn("Min", help="Enter the minimum value.", format="localized", step=1),
                "Max": st.column_config.NumberColumn("Max", help="Enter the maximum value.", format="localized", step=1)

            },     
        )
        df_constraints_program = st.session_state.df_constraints_program

    
        st.data_editor(
            st.session_state.df_constraints_program_perc,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            disabled=["Randvoorwaarden", "Eenheid"],
            key="df_constraints_program_perc_editor",
            column_config={
                "Min": st.column_config.NumberColumn("Min", help="Enter the minimum value as a percentage.",
                                                     min_value=0, max_value=100, step=1, format="%.2f %%"),
                "Max": st.column_config.NumberColumn("Max", help="Enter the maximum value as a percentage.",
                                                     min_value=0, max_value=100, step=1, format="%.2f %%")
            },
        )
        df_constraints_program_perc = st.session_state.df_constraints_program_perc

    with col2:
        st.write("**Ruimtelijke randvoorwaarden**")
        st.data_editor(
            st.session_state.df_constraints_ruimtelijk,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            disabled=["Randvoorwaarden", "Eenheid"],
            key="df_constraints_ruimtelijk_editor",
            column_config={
                "Min": st.column_config.NumberColumn("Min", help="Enter the minimum value.", format="localized", step=1),
                "Max": st.column_config.NumberColumn("Max", help="Enter the maximum value.", format="localized", step=1)
            },
        )
        df_constraints_ruimtelijk = st.session_state.df_constraints_ruimtelijk

        st.write("**Financiële randvoorwaarden**")
        st.data_editor(
            st.session_state.df_constraints_financieel,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            disabled=["Randvoorwaarden", "Eenheid"],
            key="df_constraints_financieel_editor", 
            column_config={
                "Kosten": st.column_config.NumberColumn("Kosten", format="localized", step=1),
            },
        )            
        df_constraints_financieel = st.session_state.df_constraints_financieel

# ——— Wonen Typologieën Editor ———
with st.container():
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        st.subheader("Mogelijkheden invulling programma")
    with col2:
        st.button("Bereken kolommen", on_click=calculate_columns)

st.write("🏠 **Wonen typologieën**")

st.data_editor(
    st.session_state.df_wonen,
    num_rows='dynamic',
    use_container_width=True,
    hide_index=True,
    key="df_wonen_editor",
    column_config={
        "Type": st.column_config.Column("Type", help="Enter the type of housing.", pinned=True),
        "BVO (m²)": st.column_config.NumberColumn("BVO (m²)", disabled=True, step=1, min_value=0, format="localized"),
        "GO (m²)": st.column_config.NumberColumn("GO (m²)", min_value=0, step=1, format="localized"),
        "Totaal perceel (uitgeefbaar gebied) (m²)": st.column_config.NumberColumn("Totaal perceel (uitgeefbaar gebied) (m²)", disabled=True, step=1, min_value=0, format="localized"),
        "Residuele grondwaarde per m² uitgeefbaar (€ / m²)": st.column_config.NumberColumn("Residuele grondwaarde per m² uitgeefbaar (€ / m²)", disabled=True, step=1, min_value=0, format="localized"),
        "Sociale woning": st.column_config.CheckboxColumn("Sociale woning", help="Check if the housing is social.", default=False),
        "Betaalbaar/ Middenduur woning": st.column_config.CheckboxColumn("Betaalbaar/ Middenduur woning", help="Check if the housing is affordable/middle-priced.", default=False),
        "Overig type woning": st.column_config.CheckboxColumn("Overig type woning", help="Check if the housing is of another type.", default=False),
        "Percentage bebouwd perceel (%)": st.column_config.NumberColumn("Percentage bebouwd perceel (%)", min_value=0, max_value=100, step=1, format="%.2f %%"),
        "Variabel aantal bouwlagen (voor gestapelde bouw)": st.column_config.CheckboxColumn("Variabel aantal bouwlagen (voor gestapelde bouw)", default=False),
        "Vormfactor": st.column_config.NumberColumn("Vormfactor", min_value=0, step=0.01, max_value=1, format="localized"),
        "Minimaal aantal bouwlagen": st.column_config.NumberColumn("Minimaal aantal bouwlagen", min_value=1, step=0.1, format="localized"),
        "Maximaal aantal bouwlagen": st.column_config.NumberColumn("Maximaal aantal bouwlagen", min_value=1, step=1, format="localized"),
        "Gemiddelde residuele grondwaarde per woning (€)": st.column_config.NumberColumn("Gemiddelde residuele grondwaarde per woning (€)", format="localized", step=1, min_value=0),
        "Infrastructuur benodigd in openbaar gebied per woning (excl. parkeren) (m²)": st.column_config.NumberColumn("Infrastructuur benodigd in openbaar gebied per woning (excl. parkeren) (m²)", format="localized", step=1, min_value=0),
        "Minimaal aantal woningen": st.column_config.NumberColumn("Minimaal aantal woningen", format="localized", step=1, min_value=0),
        "Minimaal aantal van type woning": st.column_config.NumberColumn("Minimaal aantal van type woning", format="localized", step=1, min_value=0),
        "Maximaal aantal woningen": st.column_config.NumberColumn("Maximaal aantal woningen", format="localized", step=1, min_value=0),
    },
)

df_wonen = st.session_state.df_wonen

st.write("🏠🅿️ **Parkeersituatie per woningtypologie**")

st.data_editor(
    st.session_state.df_wonen_parkeren,
    num_rows='dynamic',
    use_container_width=True,
    hide_index=True,
    key="df_wonen_parkeren_editor",
    column_config={
        "Type": st.column_config.Column("Type", help="Enter the type of housing.", pinned=True),
        "Parkeernorm bewoners": st.column_config.NumberColumn("Parkeernorm bewoners", format="localized", min_value=0, step=0.1),
        "Parkeernorm bezoekers": st.column_config.NumberColumn("Parkeernorm bezoekers", format="localized", min_value=0, step=0.1),
        "Totale norm": st.column_config.NumberColumn("Totale norm", disabled=True, format="localized"),
        "Korting op norm bewoners": st.column_config.NumberColumn("Korting op norm bewoners", format="%.2f %%", min_value=0, max_value=100, step=1),
        "Korting op norm bezoekers": st.column_config.NumberColumn("Korting op norm bezoekers", format="%.2f %%", min_value=0, max_value=100, step=1),
        "Parkeernorm inclusief korting": st.column_config.NumberColumn("Parkeernorm inclusief korting", disabled=True, format="localized", min_value=0, step=0.01),
        "Waarvan parkeren op eigen terrein": st.column_config.NumberColumn("Waarvan parkeren op eigen terrein", format="localized", min_value=0, step=0.1),
    },
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
    column_config={
        "Type": st.column_config.Column("Type", help="Enter the type of commercial building.", pinned=True),
        "Vormfactor": st.column_config.NumberColumn("Vormfactor", min_value=0, step=0.01, max_value=1, format="localized"),
        "Aantal bouwlagen": st.column_config.NumberColumn("Aantal bouwlagen", min_value=1, step=1, format="localized"),
        "Percentage bebouwd perceel (%)": st.column_config.NumberColumn("Percentage bebouwd perceel (%)", min_value=0, max_value=100, step=1, format="%.2f %%"),
        "Gemiddelde residuele grondwaarde per m² uitgeefbaar (€)": st.column_config.NumberColumn("Gemiddelde residuele grondwaarde per m² uitgeefbaar (€)", format="localized", step=1, min_value=0),
        "Infrastructuur benodigd in openbaar gebied per m² VVO (excl parkeren) (m²)": st.column_config.NumberColumn("Infrastructuur benodigd in openbaar gebied per m² VVO (excl parkeren) (m²)", format="localized", step=0.01, min_value = 0),
        "Parkeerplaatsen per 100 m² BVO": st.column_config.NumberColumn("Parkeerplaatsen per 100 m² BVO", format="localized", step=1, min_value=0),
        "Minimaal VVO (m²)": st.column_config.NumberColumn("Minimaal VVO (m²)", format="localized", step=1, min_value=0),
        "Maximaal VVO (m²)": st.column_config.NumberColumn("Maximaal VVO (m²)", format="localized", step=1, min_value=0),
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
    column_config={
        "Type": st.column_config.Column("Type", help="Enter the type of commercial building.", pinned=True),
        "Vormfactor": st.column_config.NumberColumn("Vormfactor", min_value=0, step=0.01, max_value=1, format="localized"),
        "Aantal bouwlagen": st.column_config.NumberColumn("Aantal bouwlagen", min_value=1, step=1, format="localized"),
        "Percentage bebouwd perceel (%)": st.column_config.NumberColumn("Percentage bebouwd perceel (%)", min_value=0, max_value=100, step=1, format="%.2f %%"),
        "Gemiddelde residuele grondwaarde per m² uitgeefbaar (€)": st.column_config.NumberColumn("Gemiddelde residuele grondwaarde per m² uitgeefbaar (€)", format="localized", step=1, min_value=0),
        "Infrastructuur benodigd in openbaar gebied per m² VVO (excl parkeren) (m²)": st.column_config.NumberColumn("Infrastructuur benodigd in openbaar gebied per m² VVO (excl parkeren) (m²)", format="localized", step=0.01, min_value = 0),
        "Parkeerplaatsen per 100 m² BVO": st.column_config.NumberColumn("Parkeerplaatsen per 100 m² BVO", format="localized", step=1, min_value=0),
        "Minimaal VVO (m²)": st.column_config.NumberColumn("Minimaal VVO (m²)", format="localized", step=1, min_value=0),
        "Maximaal VVO (m²)": st.column_config.NumberColumn("Maximaal VVO (m²)", format="localized", step=1, min_value=0),
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
    column_config={
        "Type": st.column_config.Column("Type", help="Enter the type of parking.", pinned=True),
        "BVO per parkeerplaats (m²)": st.column_config.NumberColumn("BVO per parkeerplaats (m²)", format="localized", step=1, min_value=0),
        "Maximaal aantal lagen": st.column_config.NumberColumn("Maximaal aantal lagen", format="localized", step=1, min_value=0),
        "Gemiddelde residuele grondwaarde per parkeerplaats (€)": st.column_config.NumberColumn("Gemiddelde residuele grondwaarde per parkeerplaats (€)", format="localized", step=1, min_value=0),
        "Verminderende Factor Grondwaarde per laag": st.column_config.NumberColumn("Verminderende Factor Grondwaarde per laag", format="localized", step=0.01, min_value=0, max_value=1),
    }
)   
df_parkeren_type = st.session_state.df_parkeren_type

for _ in range(whitelines):
    st.write("")

with st.container():
    col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
    with col2:
        st.markdown("<h2 style='text-align: center;'>Model</h2>", unsafe_allow_html=True)

        if 'last_file_imported' not in st.session_state:
            st.session_state['last_file_imported'] = None


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

            # Run all on changes to ensure all data is up-to-date
            df_wonen  = on_change_generic("df_wonen_editor", "df_wonen")
            df_wonen_parkeren = on_change_generic("df_wonen_parkeren_editor", "df_wonen_parkeren")
            df_nwonen_comm = on_change_generic("df_nwonen_comm_editor", "df_nwonen_comm")
            df_nwonen_maat = on_change_generic("df_nwonen_maat_editor", "df_nwonen_maat")
            df_parkeren_type = on_change_generic("df_parkeren_type_editor", "df_parkeren_type")
            df_constraints_context = on_change_generic("df_constraint_context_editor", "df_constraints_context")
            df_constraints_program = on_change_generic("df_constraints_program_editor", "df_constraints_program")
            df_constraints_program_perc = on_change_generic("df_constraints_program_perc_editor", "df_constraints_program_perc")
            df_constraints_ruimtelijk = on_change_generic("df_constraints_ruimtelijk_editor", "df_constraints_ruimtelijk")
            df_constraints_financieel = on_change_generic("df_constraints_financieel_editor", "df_constraints_financieel")

            checkbox_cols = ["Sociale woning", "Variabel aantal bouwlagen (voor gestapelde bouw)", "Betaalbaar/ Middenduur woning", "Overig type woning"]
            # Convert checkbox booleans to int for solver
            for col in checkbox_cols:
                df_wonen[col] = df_wonen[col].astype(int)

            # Run solver
            (
                df_results_context,
                df_results_ruimtelijk,
                df_results_program,
                df_results_program_perc,
                df_results_parkeren,
                df_wonen_invulling,
                df_nwonen_invulling_comm,
                df_nwonen_invulling_maat,
                df_results_invulling_parkeren
            ) = solve(
                df_wonen=df_wonen.copy(),
                df_nwonen_comm=df_nwonen_comm.copy(),
                df_nwonen_maat=df_nwonen_maat.copy(),
                df_wonen_parkeren=df_wonen_parkeren.copy(),
                df_parkeren_type=df_parkeren_type.copy(),
                df_constraints_context=df_constraints_context.copy(),
                df_constraints_program=df_constraints_program.copy(),
                df_constraints_program_perc=df_constraints_program_perc.copy(),
                df_constraints_ruimtelijk=df_constraints_ruimtelijk.copy(),
                df_constraints_financieel=df_constraints_financieel.copy(),       
                optimize_for=st.session_state.optimize_for,
                optimize_for_2=st.session_state.optimize_for_2
            )

            print("==== Results dataframes =====")
            print(df_results_context)
            print(df_results_ruimtelijk)
            print(df_results_program)
            print(df_results_program_perc)
            print(df_results_parkeren)
            print(df_wonen_invulling)
            print(df_nwonen_invulling_comm)
            print(df_nwonen_invulling_maat)
            print(df_results_invulling_parkeren)
            
            # Store solver results
            st.session_state.df_results_context = enforce_numeric_columns(df_results_context, exclude_columns)
            st.session_state.df_results_program = enforce_numeric_columns(df_results_program, exclude_columns)
            st.session_state.df_results_program_perc = enforce_numeric_columns(df_results_program_perc, exclude_columns)
            st.session_state.df_results_ruimtelijk = enforce_numeric_columns(df_results_ruimtelijk, exclude_columns)
            st.session_state.df_results_parkeren = enforce_numeric_columns(df_results_parkeren, exclude_columns)
            st.session_state.df_wonen_invulling = enforce_numeric_columns(df_wonen_invulling, exclude_columns)
            st.session_state.df_nwonen_invulling_comm = enforce_numeric_columns(df_nwonen_invulling_comm, exclude_columns)
            st.session_state.df_nwonen_invulling_maat = enforce_numeric_columns(df_nwonen_invulling_maat, exclude_columns)
            st.session_state.df_results_invulling_parkeren = enforce_numeric_columns(df_results_invulling_parkeren, exclude_columns)

            st.session_state.run_model_clicked = True

        if st.session_state.get("run_model_clicked", False):
            st.success("Model succesvol afgerond!")

        uploaded_json_file = st.file_uploader('uploader-input', type="json", label_visibility="hidden")
        if uploaded_json_file is not None and st.session_state.last_file_imported != uploaded_json_file:
            st.session_state.last_file_imported = uploaded_json_file
            import_model_fromjson(uploaded_json_file)
            
        if st.button('Model Exporteren', use_container_width=True):
            df_wonen  = on_change_generic("df_wonen_editor", "df_wonen")
            df_wonen_parkeren = on_change_generic("df_wonen_parkeren_editor", "df_wonen_parkeren")
            df_nwonen_comm = on_change_generic("df_nwonen_comm_editor", "df_nwonen_comm")
            df_nwonen_maat = on_change_generic("df_nwonen_maat_editor", "df_nwonen_maat")
            df_parkeren_type = on_change_generic("df_parkeren_type_editor", "df_parkeren_type")
            df_constraints_context = on_change_generic("df_constraint_context_editor", "df_constraints_context")
            df_constraints_program = on_change_generic("df_constraints_program_editor", "df_constraints_program")
            df_constraints_program_perc = on_change_generic("df_constraints_program_perc_editor", "df_constraints_program_perc")
            df_constraints_ruimtelijk = on_change_generic("df_constraints_ruimtelijk_editor", "df_constraints_ruimtelijk")
            df_constraints_financieel = on_change_generic("df_constraints_financieel_editor", "df_constraints_financieel")

            checkbox_cols = ["Sociale woning", "Variabel aantal bouwlagen (voor gestapelde bouw)", "Betaalbaar/ Middenduur woning", "Overig type woning"]
            for col in checkbox_cols:
                df_wonen[col] = df_wonen[col].astype(int)

            export_model_tojson(
                df_wonen,
                df_wonen_parkeren,
                df_nwonen_comm,
                df_nwonen_maat,
                df_parkeren_type,
                df_constraints_context,
                df_constraints_program_perc,
                df_constraints_program,
                df_constraints_ruimtelijk,
                df_constraints_financieel
            )

#############################################
# 7) Display Output DataFrames
#############################################
with st.container():
    st.subheader("Resultaten van het Model")
    col1, col2 = st.columns([0.5, 0.5])
    
    with col1:
        st.write("**Context**")
        st.dataframe(
            st.session_state.df_results_context,
            use_container_width=True,
            hide_index=True,
            key="df_results_context_editor",
            column_config={
                "Output": st.column_config.NumberColumn("Waarde", format="localized", step=1)
            }

        )
    
        st.write("**Ruimtelijke output**")
        st.dataframe(
            st.session_state.df_results_ruimtelijk,
            use_container_width=True,
            hide_index=True,
            key="df_results_ruimtelijk_editor",
            column_config={
                "Output": st.column_config.NumberColumn("Waarde", format="localized", step=1)
            }
        )

    with col2:
        st.write("**Programmatische output**")
        st.dataframe(
            st.session_state.df_results_program,
            use_container_width=True,
            hide_index=True,
            key="df_results_program_editor",
            column_config={
                "Output": st.column_config.NumberColumn("Waarde", format="localized", step=1)
            }
        )

        st.dataframe(
            st.session_state.df_results_program_perc,
            use_container_width=True,
            hide_index=True,
            key="df_results_program_perc_editor",
            column_config={
                "Output": st.column_config.NumberColumn("Output", format="%.2f %%")
            }
        )

        st.write("🅿️ **Output parkeren**")
        st.dataframe(
            st.session_state.df_results_parkeren,
            use_container_width=True,
            hide_index=True,
            key="df_parkeren_output_editor",
            column_config={
                "Output": st.column_config.NumberColumn("Waarde", format="localized", step=1)
            }

        )

st.subheader("Invulling van het Programma")
with st.container():
    col1, col2 = st.columns([0.5, 0.5])
    
    with col1:
        st.write("🏠 **Wonen**")
        st.dataframe(
            st.session_state.df_wonen_invulling,
            use_container_width=True,
            hide_index=True,
            key="df_wonen_output_editor",
            column_config={
                "Aantal": st.column_config.NumberColumn("Aantal", format="localized", step=1),
                "Aantal lagen": st.column_config.NumberColumn("Aantal lagen", format="localized", step=1),
                "Totaal uitgeefbaar gebied (m²)": st.column_config.NumberColumn("Totaal uitgeefbaar gebied (m²)", format="localized", step=1),
                "Totale residuele grondwaarde (€)": st.column_config.NumberColumn("Totale residuele grondwaarde (€)", format="localized", step=1),
            }
        )

       
    with col2:
        st.write("🏪 **Commercieel vastgoed**")
        st.dataframe(
            st.session_state.df_nwonen_invulling_comm,
            use_container_width=True,
            hide_index=True,
            key="df_nwonen_output_editor",
            column_config={
                "Aantal": st.column_config.NumberColumn("Aantal", format="localized", step=1),
                "VVO (m²)": st.column_config.NumberColumn("VVO (m²)", format="localized", step=1),
                "Totaal uitgeefbaar gebied (m²)": st.column_config.NumberColumn("Totaal uitgeefbaar gebied (m²)", format="localized", step=1),
                "Totale residuele grondwaarde (€)": st.column_config.NumberColumn("Totale residuele grondwaarde (€)", format="localized", step=1),
            }
        )

        st.write("**Maatschappelijk vastgoed**")
        st.dataframe(
            st.session_state.df_nwonen_invulling_maat,
            use_container_width=True,
            hide_index=True,
            key="df_nwonen_output_editor",
            column_config={
                "Aantal": st.column_config.NumberColumn("Aantal", format="localized", step=1),
                "VVO (m²)": st.column_config.NumberColumn("VVO (m²)", format="localized", step=1),
                "Totaal uitgeefbaar gebied (m²)": st.column_config.NumberColumn("Totaal uitgeefbaar gebied (m²)", format="localized", step=1),
                "Totale residuele grondwaarde (€)": st.column_config.NumberColumn("Totale residuele grondwaarde (€)", format="localized", step=1),
            }
        )

        st.write("🅿️ **Parkeren**")
        st.dataframe(
            st.session_state.df_results_invulling_parkeren,
            use_container_width=True,
            hide_index=True,
            key="df_parkeren_output_editor",
            column_config={
                "Aantal parkeerplaatsen": st.column_config.NumberColumn("Aantal parkeerplaatsen", format="localized", step=1),
                "Totaal BVO (m²)": st.column_config.NumberColumn("Totaal BVO (m²)", format="localized", step=1),
                "Totale residuele grondwaarde (€)": st.column_config.NumberColumn("Totale residuele grondwaarde (€)", format="localized", step=1),
                "Aantal lagen": st.column_config.NumberColumn("Aantal lagen", format="localized", step=1),
                "Footprint (m²)": st.column_config.NumberColumn("Footprint (m²)", format="localized", step=1),
            }
        )