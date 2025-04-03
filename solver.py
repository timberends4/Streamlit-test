import pulp

import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpInteger, LpStatus, value, PULP_CBC_CMD, LpContinuous
import numpy as np
from itertools import product


def extract_constraints(df_constraints_context, df_constraints_program, df_constraints_program_perc, df_constraints_ruimtelijk, df_constraints_financieel):
    
    def mm(df, row):
        vals = df.loc[row, ["Min", "Max"]]
        vals = pd.to_numeric(vals, errors='coerce')
        return vals
    
    def k(df, row):
        val = df.loc[row, "Kosten"]
        return val

    def e(df, row):
        val = df.loc[row, "Gestelde eis"]
        return val
    
    constraints ={
        'opp': e(df_constraints_context, "Oppervlakte gebied"),
        'grondwaarde': (e(df_constraints_context, "Minimale totale grondwaarde"), 
                        e(df_constraints_context, "Maximale totale grondwaarde")),

        'woningen': mm(df_constraints_program, "Totaal aantal woningen"),
        'sociaal': mm(df_constraints_program, "Aantal sociale woningen"),
        'betaalbaar': mm(df_constraints_program, "Aantal betaalbare/ middenhuur woningen"),
        'niet_wonen_comm': mm(df_constraints_program, "Aantal m² commercieel vastgoed"),
        'niet_wonen_maat': mm(df_constraints_program, "Aantal m² maatschappelijk vastgoed"),

        'perc_sociaal': mm(df_constraints_program_perc, "Percentage sociale woningen") / 100,
        'perc_betaalbaar': mm(df_constraints_program_perc, "Percentage betaalbare/ middenhuur woningen") / 100,

        'norm_groen_pw': mm(df_constraints_ruimtelijk, "Groennorm per woning"),
        'norm_water_pw': mm(df_constraints_ruimtelijk, "Waternorm per woning"),
        'opp_groen': mm(df_constraints_ruimtelijk, "Aantal m² groen"),
        'opp_water': mm(df_constraints_ruimtelijk, "Aantal m² water"),

        'kosten_groen': k(df_constraints_financieel, "Kosten Groen"),
        'kosten_water': k(df_constraints_financieel, "Kosten Water"),
        'kosten_infra': k(df_constraints_financieel, "Kosten Infrastructuur"),
        'kosten_bouwrijp': k(df_constraints_financieel, "Bouwrijp maken gebied"),
    }
    return constraints

def solve(
                df_wonen,
                df_nwonen_comm,
                df_nwonen_maat,
                df_wonen_parkeren,
                df_parkeren_type,
                df_constraints_context,
                df_constraints_program,
                df_constraints_program_perc,
                df_constraints_ruimtelijk,
                df_constraints_financieel,       
                optimize_for,
                optimize_for_2
            ):
    df_wonen = df_wonen.copy()
    df_nwonen_comm = df_nwonen_comm.copy()
    df_nwonen_maat = df_nwonen_maat.copy()
    df_wonen_parkeren = df_wonen_parkeren.copy()
    df_parkeren_type = df_parkeren_type.copy()
    df_constraints_context = df_constraints_context.copy().set_index("Randvoorwaarden")
    df_constraints_program = df_constraints_program.copy().set_index("Randvoorwaarden")
    df_constraints_program_perc = df_constraints_program_perc.copy().set_index("Randvoorwaarden")
    df_constraints_ruimtelijk = df_constraints_ruimtelijk.copy().set_index("Randvoorwaarden")
    df_constraints_financieel = df_constraints_financieel.copy().set_index("Randvoorwaarden")    




    # Extract constraint values
    constraints = extract_constraints(df_constraints_context, 
                                    df_constraints_program, 
                                    df_constraints_program_perc, 
                                    df_constraints_ruimtelijk, 
                                    df_constraints_financieel)
    
    assert not pd.isna(constraints["opp"]), "Stel een eis aan het oppervlakte van het gebied."
    assert not pd.isna(constraints["kosten_groen"]), "Vul de kosten voor groen in."
    assert not pd.isna(constraints["kosten_water"]), "Vul de kosten voor water in."
    assert not pd.isna(constraints["kosten_infra"]), "Vul de kosten voor infrastructuur in."
    assert not pd.isna(constraints["kosten_bouwrijp"]), "Vul de kosten voor bouwrijp maken in."

    # Merge niet wonen types
    df_nwonen_comm['Commercieel'] = True
    df_nwonen_comm['Maatschappelijk'] = False
    df_nwonen_maat['Commercieel'] = False
    df_nwonen_maat['Maatschappelijk'] = True
    df_niet_wonen = pd.concat([df_nwonen_comm, df_nwonen_maat])

    df_wonen = df_wonen.dropna(axis=0, how='all')
    df_niet_wonen = df_niet_wonen.dropna(axis=0, how='all')
    df_wonen_parkeren = df_wonen_parkeren.dropna(axis=0, how='all')
    df_parkeren_type = df_parkeren_type.dropna(axis=0, how='all')

    df_parkeren_type.rename(columns={"Parkeer typologiën": "Type"}, inplace=True)
    index_by_type(df_wonen)
    index_by_type(df_niet_wonen)
    index_by_type(df_wonen_parkeren)
    index_by_type(df_parkeren_type)


    # We add a relevant column from df_wonen_parkeren to df_wonen
    # We want to do this in the DataFrame to make our lives easier once we make the dictionaries
    # We use join to make sure it does not break down if one of the two data frames has a different order of the types
    # DO THIS BEFORE SPLITTING LAYERS

    df_wonen_parkeren["Waarvan parkeren buiten eigen terrein"] = (df_wonen_parkeren["Parkeernorm inclusief korting"].fillna(0) - 
                                                                  df_wonen_parkeren["Waarvan parkeren op eigen terrein"].fillna(0))
    df_wonen = df_wonen.join(df_wonen_parkeren[['Waarvan parkeren buiten eigen terrein']])
  
 
    df_wonen = split_layers(
                            df_wonen, df_wonen["Minimaal aantal bouwlagen"], 
                            df_wonen["Maximaal aantal bouwlagen"], 
                            df_wonen["Variabel aantal bouwlagen (voor gestapelde bouw)"])
    df_parkeren_type = split_layers(df_parkeren_type, 1, 
                                    df_parkeren_type["Maximaal aantal lagen"], 
                                    True)

    assert_column_filled(df_wonen["BVO (m²)"])
    assert_column_filled(df_wonen["Lagen"])
    assert_column_filled(df_wonen["Percentage bebouwd perceel (%)"])
    df_wonen["Uitgeefbaar"] = df_wonen["BVO (m²)"] / df_wonen["Lagen"] / (df_wonen["Percentage bebouwd perceel (%)"] / 100)
    assert_column_filled(df_wonen["Gemiddelde residuele grondwaarde per woning (€)"])
    df_wonen["Grondwaarde per woning"] = df_wonen["Gemiddelde residuele grondwaarde per woning (€)"]

    assert_column_filled(df_niet_wonen["Aantal bouwlagen"])
    assert_column_filled(df_niet_wonen["Percentage bebouwd perceel (%)"])
    df_niet_wonen["Uitgeefbaar per m² VVO"] = df_niet_wonen["Aantal bouwlagen"] / (df_niet_wonen["Percentage bebouwd perceel (%)"] / 100)

    prob, (aw, yh, ap, vnw, g, w), (context, grondwaarde, ruimtelijk, programma, parkeren) = build_lp(df_wonen, 
                                                                                                      df_niet_wonen, 
                                                                                                      df_parkeren_type, 
                                                                                                      constraints)

    objective, obj_name = get_objective(context, grondwaarde,ruimtelijk, programma, parkeren, optimize_for)
    prob.name = obj_name
    prob.setObjective(objective)

    prob.solve(PULP_CBC_CMD(msg=False))
    status = prob.status
    

    print(prob)
    
    if status == 1:
        print_ouput(prob, aw, yh, ap, vnw, g, w, context, grondwaarde, ruimtelijk, programma, parkeren)
    else:
        print("No solution found for objective 1:", optimize_for)
        raise ValueError("No solution found for given input and constraints")

    if optimize_for_2 != "Geen" and optimize_for_2 != optimize_for:
        set_pre_optimized_constraints(prob, context, grondwaarde, ruimtelijk, programma, parkeren, 
                                      optimize_for, 
                                      objective.value())
        
        objective_2, obj_name_2 = get_objective(context, grondwaarde, ruimtelijk, programma, parkeren, optimize_for_2)
        prob.setObjective(objective_2)
        prob.name = f"First_{obj_name}_Second_{obj_name_2}"

        prob.solve(PULP_CBC_CMD(msg=False))
        status = prob.status

        if status == 1:
            print_ouput(prob, aw, yh, ap, vnw, g, w, context, grondwaarde, ruimtelijk, programma, parkeren)
        else:
            print("No solution found for objective 2:", optimize_for_2)
            raise ValueError("No solution found for given input and constraints")

    else:
        print("No second optimization run because optimize_for_2 is Geen or the same as optimize_for")

    return create_result_dfs(df_wonen, df_niet_wonen, df_parkeren_type,
                      aw, yh, ap, vnw, g, w, 
                      context, grondwaarde, ruimtelijk, programma, parkeren)


def print_ouput(prob, aw, yh, ap, vnw, g, w, context, grondwaarde, ruimtelijk, programma, parkeren):
    print("Status:", LpStatus[prob.status])
    print("Output:")
    for d, name in zip([aw, yh, ap, vnw], ["Aantal woningen", "Use woning type", "Aantal parkeren", "VVO niet-wonen"]):
        print("===VARIABLEA:",name)
        for key, value in d.items():
            print(key, value.value())
    print("===Groen", g.value())
    print("===Water", w.value())
    for d, name in zip([context, grondwaarde, ruimtelijk, programma, parkeren], ["context", "grondwaarde", "ruimtelijk", "programma", "parkeren"]):
        print("===VARIABLE DICTIONARY:",name)
        for key, value in d.items():
            print(key, value.value())


def create_result_dfs(df_wonen, df_niet_wonen, df_parkeren_type,
                      aw, yh, ap, vnw, g, w, 
                      context, grondwaarde, ruimtelijk, programma, parkeren):
    out_context = pd.DataFrame([
        ["Oppervlakte gebied", context["opp"].value(), "m²"],
        ["Totale residuele grondwaarde", context["grondwaarde"].value(), "€"]],
        columns=["Onderdeel", "Waarde", "Eenheid"])

    out_ruimtelijk = pd.DataFrame([
        ["Totaal uitgeefbaar gebied", ruimtelijk["opp_uitgeefbaar"].value(), "m²"],
        ["Totaal oppervlak infrastructuur (exclusief parkeren)", ruimtelijk["opp_infra_ex_parkeren"].value(), "m²"],
        ["Totaal oppervlak parkeren", ruimtelijk["opp_parkeren"].value(), "m²"],
        ["Totaal groen", g.value(), "m²"],
        ["Totaal water", w.value(), "m²"],
        ["Groennorm per woning", g.value()/programma["woningen"].value() if programma["woningen"].value() != 0 else "-", "m²/woning"],
        ["Waternorm per woning", w.value()/programma["woningen"].value() if programma["woningen"].value() != 0 else "-", "m²/woning"]],
        columns=["Onderdeel", "Waarde", "Eenheid"])
    
    out_programma = pd.DataFrame([
        ["Totaal aantal woningen", programma["woningen"].value(), "Aantal"],
        ["Aantal sociale woningen", programma["sociaal"].value(), "Aantal"],
        ["Aantal betaalbare/ middenhuur woningen", programma["betaalbaar"].value(), "Aantal"],
        ["Aantal m² commercieel vastgoed", programma["niet_wonen_comm"].value(), "m²"],
        ["Aantal m² maatschappelijk vastgoed", programma["niet_wonen_maat"].value(), "m²"]],
        columns=["Onderdeel", "Waarde", "Eenheid"])
    
    out_programma_perc = pd.DataFrame([
        ["Percentage sociale woningen", programma["sociaal"].value()/programma["woningen"].value() *100 if programma["woningen"].value() != 0 else "-", "%"],
        ["Percentage betaalbare/ middenhuur woningen", programma["betaalbaar"].value()/programma["woningen"].value() *100 if programma["woningen"].value() != 0 else "-", "%"]],
        columns=["Onderdeel", "Waarde", "Eenheid"])
    
    out_parkeren = pd.DataFrame([
        ["Parkeerplaatsen in openbaar gebied t.b.v. woningen", parkeren["parkeren_tbv_wonen"].value(), "Aantal"],
        ["Parkeerplaatsen in openbaar gebied t.b.v. commercieel vastgoed", parkeren["parkeren_tbv_nwonen_comm"].value(), "Aantal"],
        ["Parkeerplaatsen in openbaar gebied t.b.v. maatschappelijk vastgoed", parkeren["parkeren_tbv_nwonen_maat"].value(), "Aantal"]],
        columns=["Onderdeel", "Waarde", "Eenheid"])


    housing_type_parents = df_wonen["Type parent"].unique()
    housing_type_children = inv_map(df_wonen["Type parent"].to_dict())
    wonen_rows = []
    for tp in housing_type_parents:
        num_used = 0
        for t in housing_type_children[tp]:
            aantal = aw[t].value()
            # Only show a row with this number of layers if it is actually used
            if aantal > 0:
                num_used += 1
                wonen_rows.append({
                    'Type': tp.replace('_', ' '),
                    'Aantal': aantal,
                    'Aantal lagen': df_wonen.loc[t, "Lagen"],
                    'Totaal uitgeefbaar gebied (m²)': aantal * df_wonen.loc[t, "Uitgeefbaar"],
                    'Totale residuele grondwaarde (€)': aantal * df_wonen.loc[t, "Grondwaarde per woning"]
                })
        # If none of this housing type are used at all, we still add a row indicating that none were used
        if num_used == 0:
            wonen_rows.append({
                'Type': tp.replace('_', ' '),
                'Aantal': 0,
                'Aantal lagen': 0,
                'Totaal uitgeefbaar gebied (m²)': 0,
                'Totale residuele grondwaarde (€)': 0
            })
    out_wonen = pd.DataFrame(wonen_rows)


    non_housing_types = df_niet_wonen.index.unique()
    comm_rows = []
    maat_rows = []
    for t in non_housing_types:
        vvo = vnw[t].value()
            # Only show a row with this number of layers if it is actually used
        num_used += 1
        row = {
            'Type': t.replace('_', ' '),
            'VVO (m²)': vvo,
            'Aantal lagen': df_niet_wonen.loc[t, "Aantal bouwlagen"],
            'Totaal uitgeefbaar gebied (m²)': vvo * df_niet_wonen.loc[t, "Uitgeefbaar per m² VVO"],   
            'Totale residuele grondwaarde (€)': vvo * df_niet_wonen.loc[t, "Gemiddelde residuele grondwaarde per m² uitgeefbaar (€)"]
        }

        if df_niet_wonen.loc[t, "Commercieel"] == 1:
            comm_rows.append(row)
        else:
            maat_rows.append(row)
    out_comm = pd.DataFrame(comm_rows)
    out_maat = pd.DataFrame(maat_rows)


    parking_type_parents = df_parkeren_type["Type parent"].unique()
    parking_type_children = inv_map(df_parkeren_type["Type parent"].to_dict())
    parkeren_rows = []
    for tp in parking_type_parents:
        num_used = 0
        for t in parking_type_children[tp]:
            aantal = ap[t].value()
            # Only show a row with this number of layers if it is actually used
            if aantal > 0:
                num_used += 1
                row = {
                    'Type': tp.replace('_', ' '),
                    'Aantal parkeerplaatsen': aantal,
                    'Totaal BVO (m²)': aantal * df_parkeren_type.loc[t, "BVO per parkeerplaats (m²)"],
                    'Totale residuele grondwaarde (€)': aantal * df_parkeren_type.loc[t, "Gemiddelde residuele grondwaarde per parkeerplaats (€)"],
                    'Aantal lagen': df_parkeren_type.loc[t, "Lagen"]
                }
                row['Footprint (m²)'] = row['Totaal BVO (m²)']/row['Aantal lagen']
                parkeren_rows.append(row)
        # If none of this housing type are used at all, we still add a row indicating that none were used
        if num_used == 0:
            parkeren_rows.append({
                'Type': tp.replace('_', ' '),
                'Aantal parkeerplaatsen': 0,
                'Totaal BVO (m²)': 0,
                'Totale residuele grondwaarde (€)': 0,
                'Aantal lagen': 0
            })

    out_parkeren_types = pd.DataFrame(parkeren_rows)
    return out_context, out_ruimtelijk, out_programma, out_programma_perc, out_parkeren, out_wonen, out_comm, out_maat, out_parkeren_types
    

def build_lp(df_wonen, df_niet_wonen, df_parkeren_type, constraints):
    M = 1e10
    eps = 1e-6

    # If set to True, only one choice of layer per (parking/housing) type is possible.
    # If set to False, several choices of layers per type can be used
    # Mostly useful to benchmark against previous version
    homogenize_layers_wonen = False
    homogenize_layers_parking = False


    housing_types = df_wonen.index.to_list()
    non_housing_types = df_niet_wonen.index.to_list()
    parking_types = df_parkeren_type.index.to_list()

    # Create dictionaries for niet wonen types
    is_maatschappelijk_nwonen = df_niet_wonen["Maatschappelijk"].to_dict()
    is_commercieel_nwonen = df_niet_wonen["Commercieel"].to_dict()
    
    min_opp_nwonen = df_niet_wonen["Minimaal VVO (m²)"].to_dict()
    max_opp_nwonen = df_niet_wonen["Maximaal VVO (m²)"].to_dict()
    vormfactor_nwonen = to_dict_not_na(df_niet_wonen["Vormfactor"])

    uitgeefbaar_per_m2_vvo_nwoning = to_dict_not_na(df_niet_wonen["Uitgeefbaar per m² VVO"])
    grondwaarde_per_m2_uitgeefbaar_nwoning = to_dict_not_na(df_niet_wonen["Gemiddelde residuele grondwaarde per m² uitgeefbaar (€)"])
    
    pp_per_100_m2_bvo_nwonen = df_niet_wonen["Parkeerplaatsen per 100 m² BVO"].fillna(0).to_dict()
    infra_per_m2_vvo_nwonen = df_niet_wonen["Infrastructuur benodigd in openbaar gebied per m² VVO (excl parkeren) (m²)"].fillna(0).to_dict()

     # grondwaarde
    # uitgeefbaar
    # parkeren
    # TODO

    # Create dictionaries for wonen types
    housing_type_parents = df_wonen["Type parent"].unique()
    housing_type_children = inv_map(df_wonen["Type parent"].to_dict())

    is_sociaal_wonen = df_wonen["Sociale woning"].to_dict()
    is_betaalbaar_wonen = df_wonen["Betaalbaar/ Middenduur woning"].to_dict()

    min_wonen_type = df_wonen["Minimaal aantal van type woning"].fillna(0).to_dict()
    min_wonen = df_wonen["Minimaal aantal woningen"].fillna(0).to_dict()
    max_wonen = df_wonen["Maximaal aantal woningen"].to_dict()

    uitgeefbaar_per_woning = df_wonen["Uitgeefbaar"].to_dict()
    grondwaarde_per_woning = df_wonen["Grondwaarde per woning"].to_dict()
    
    infra_per_woning_ex_pp = to_dict_not_na(df_wonen["Infrastructuur benodigd in openbaar gebied per woning (excl. parkeren) (m²)"])
    pp_per_woning = to_dict_not_na(df_wonen["Waarvan parkeren buiten eigen terrein"])
    


    # Create dictionaries for parkeren types
    parking_type_parents = df_parkeren_type["Type parent"].unique()
    parking_type_children = inv_map(df_parkeren_type["Type parent"].to_dict())
    lagen_pp = to_dict_not_na(df_parkeren_type["Lagen"])
    bvo_pp = to_dict_not_na(df_parkeren_type["BVO per parkeerplaats (m²)"])
    res_grondwaarde_pp = to_dict_not_na(df_parkeren_type["Gemiddelde residuele grondwaarde per parkeerplaats (€)"])
    discount_factor_pp = df_parkeren_type["Verminderende Factor Grondwaarde per laag"].fillna(0).to_dict()

    prob = LpProblem("Model", LpMaximize)
    aw = LpVariable.dicts("Aantal_woning", housing_types, lowBound=0, cat=LpInteger)
    yh = LpVariable.dicts("Binary", housing_types, cat="Binary")  # Binary variables, whether housing type placed
    ap = LpVariable.dicts("Aantal_parkeren", parking_types, lowBound=0, cat=LpInteger)
    vnw = LpVariable.dicts("Oppervlakte_niet_wonen", non_housing_types, lowBound=0, cat=LpInteger)
    
    g = LpVariable("g", lowBound=0, cat=LpInteger)
    w = LpVariable("w", lowBound=0, cat=LpInteger)

    # # TODO check deze berekningen
    # grondwaarde_woningen = lpSum([grondwaarde_per_woning[i] * aw[i] for i in housing_types])
    # grondwaarde_niet_wonen = lpSum([grondwaarde_per_nwoning[i] * anw[i] for i in non_housing_types])
    # grondwaarde_parkeren = lpSum([res_grondwaarde_pp[i] * ap[i] *
    #                                 (1 - (discount_factor_pp[i] * lagen_pp[i])) for i
    #                                 in parking_types])


    # groennorm per woning cannot be implemented here,
    ruimtelijk = {
        "opp_uitgeefbaar": (lpSum([uitgeefbaar_per_woning[i] * aw[i] for i in housing_types]) +
                            lpSum([uitgeefbaar_per_m2_vvo_nwoning[i] * vnw[i] for i in non_housing_types])),
        "opp_infra_ex_parkeren": lpSum([infra_per_m2_vvo_nwonen[i] * vnw[i] for i in non_housing_types]) + lpSum([infra_per_woning_ex_pp[i] * aw[i] for i in housing_types]),
        "opp_parkeren": lpSum([ap[i] * bvo_pp[i] / lagen_pp[i] for i in parking_types]),
    }

    programma = {
        "woningen": lpSum([aw[i] for i in housing_types]),
        "sociaal": lpSum([aw[i] for i in housing_types if is_sociaal_wonen[i]]),
        "betaalbaar": lpSum([aw[i] for i in housing_types if is_betaalbaar_wonen[i]]),
        "niet_wonen_comm": lpSum([vnw[i] for i in non_housing_types if is_commercieel_nwonen[i]]),
        "niet_wonen_maat": lpSum([vnw[i] for i in non_housing_types if is_maatschappelijk_nwonen[i]]),
    }

    # Percentages can be implemented as constraints, but we cannot compute them here. Has to be done after solving

    parkeren_fractional = {
        "parkeren_tbv_wonen": lpSum([pp_per_woning[i] * aw[i] for i in housing_types]),
        "parkeren_tbv_nwonen_comm": lpSum([pp_per_100_m2_bvo_nwonen[i] * vnw[i] / vormfactor_nwonen[i] / 100 for i in non_housing_types if is_commercieel_nwonen[i]]),
        "parkeren_tbv_nwonen_maat": lpSum([pp_per_100_m2_bvo_nwonen[i] * vnw[i] / vormfactor_nwonen[i] / 100 for i in non_housing_types if is_maatschappelijk_nwonen[i]]),
    }

        # Parking tbv integer maken
    p_wonen = LpVariable("pw", cat=LpInteger, lowBound=0)
    p_nwonen_comm = LpVariable("pnw_comm", cat=LpInteger, lowBound=0)
    p_nwonen_maat = LpVariable("pnw_maat", cat=LpInteger, lowBound=0)

    prob += p_wonen >= parkeren_fractional["parkeren_tbv_wonen"] 
    prob += p_nwonen_comm >= parkeren_fractional["parkeren_tbv_nwonen_comm"]
    prob += p_nwonen_maat >= parkeren_fractional["parkeren_tbv_nwonen_maat"]

    prob += p_wonen <= parkeren_fractional["parkeren_tbv_wonen"] + 1 - eps
    prob += p_nwonen_comm <= parkeren_fractional["parkeren_tbv_nwonen_comm"] + 1 - eps
    prob += p_nwonen_maat <= parkeren_fractional["parkeren_tbv_nwonen_maat"] + 1 - eps
    
    parkeren = {
        "parkeren_tbv_wonen": p_wonen,
        "parkeren_tbv_nwonen_comm": p_nwonen_comm,
        "parkeren_tbv_nwonen_maat": p_nwonen_maat,
    }

    prob += lpSum([ap[i] for i in parking_types]) == parkeren["parkeren_tbv_wonen"] + parkeren["parkeren_tbv_nwonen_comm"] + parkeren["parkeren_tbv_nwonen_maat"]


    context = {
        "opp": (ruimtelijk["opp_uitgeefbaar"] + 
                ruimtelijk["opp_infra_ex_parkeren"] + 
                ruimtelijk["opp_parkeren"] + 
                g + 
                w)
    }
    

    grondwaarde = {
        "woningen": lpSum([grondwaarde_per_woning[i] * aw[i] for i in housing_types]),
        "niet_wonen": lpSum([grondwaarde_per_m2_uitgeefbaar_nwoning[i] * 
                             uitgeefbaar_per_m2_vvo_nwoning[i] * 
                             vnw[i] 
                             for i in non_housing_types]),
        "parkeren": lpSum([res_grondwaarde_pp[i] * ap[i] *
                                      (1 - (discount_factor_pp[i] * lagen_pp[i])) for i
                                      in parking_types]),
    }

    costs = {
        "cost_groen": g * constraints['kosten_groen'],
        "cost_water": w * constraints['kosten_water'],
        "cost_infra": ruimtelijk["opp_infra_ex_parkeren"] * constraints['kosten_infra'],
        "cost_bouwrijp": context["opp"] * constraints['kosten_bouwrijp'],
    }

    context["grondwaarde"] = lpSum(grondwaarde["woningen"]
                             + grondwaarde["niet_wonen"] 
                             + grondwaarde["parkeren"]
                             - costs["cost_groen"]
                             - costs["cost_water"]
                             - costs["cost_infra"]
                             - costs["cost_bouwrijp"])

    
    # --- Min type woningen constraint ---
    # This is layer-specific
    for i in housing_types:
        prob += aw[i] >= min_wonen_type[i] * yh[i], f"Conditional_Min_{i}"  # Ensure minimum if placed
        prob += aw[i] <= M * yh[i], f"Placed_{i}"

        # --- Homogenize layers within housing type ---
    if homogenize_layers_wonen:
        for htp in housing_type_parents:
            prob += (lpSum([yh[ht] for ht in housing_type_children[htp]]) <= 1,
                        f"Homogenize layers housing type {htp}")
            
    if homogenize_layers_parking:
        yp = LpVariable.dicts("Binary", parking_types,
                                cat="Binary")  # Binary variables, whether housing type placed
        for pt in parking_types:
            prob += ap[pt] <= M * yp[pt], f"Used {pt}"

        for ptp in parking_type_parents:
            prob += (lpSum([yp[pt] for pt in parking_type_children[ptp]]) <= 1,
                        f"Homogenize layers parking type {ptp}")

    # The "Min" and "Max" are bounds on the total number of a type parent
    # Since the variables are for type children (orig. type + nr of layers),
    # this is a constraint on a sum of variables
    for tp in housing_type_parents:
        sum_amount_children = lpSum([aw[t] for t in housing_type_children[tp] ])
        # IMPORTANT: we assume here the Min and Max woningen are EQUAL among housing type children.
        # In practice this is true because we don't touch it when splitting the types
        add_min_max_constraints(prob, sum_amount_children,
                                    (min_wonen[housing_type_children[tp][0]],
                                    max_wonen[housing_type_children[tp][0]]),
                                    f"Min_{tp}")

    # Minimum and maximum number of houses
    add_min_max_constraints(prob, programma["woningen"], 
                                constraints['woningen'], "Woningen")

    # Sociaal housing constraints
    add_min_max_constraints(prob, programma["sociaal"], 
                                constraints['sociaal'], "Sociaal")

    
    # Since percentage constraints involve division, we need to linearize them
    add_min_max_constraints(prob, programma["sociaal"], 
                                constraints['perc_sociaal'],
                                "Perc_Sociaal",
                                multiplier=programma["woningen"])

    # Betaalbaar housing constraints
    add_min_max_constraints(prob, programma["betaalbaar"], 
                                constraints['betaalbaar'], 
                                "Betaalbaar")

    add_min_max_constraints(prob, programma["betaalbaar"], 
                                constraints['perc_betaalbaar'],
                                "Perc_Betaalbaar",
                                multiplier=programma["woningen"])
    # Niet wonen

    for nw_type in non_housing_types:
        add_min_max_constraints(prob, vnw[nw_type],
                                    (min_opp_nwonen[nw_type],
                                    max_opp_nwonen[nw_type]),
                                    f"VVO_{nw_type}")

    add_min_max_constraints(prob, programma["niet_wonen_comm"],
                                constraints['niet_wonen_comm'],
                                "Niet_Wonen_Comm")

    add_min_max_constraints(prob, programma["niet_wonen_maat"],
                                constraints['niet_wonen_maat'],
                                "Niet_Wonen_Maat")

    # Groen
    add_min_max_constraints(prob, g,
                                constraints['norm_groen_pw'],
                                "Groennorm",
                                multiplier=programma["woningen"])
    add_min_max_constraints(prob, g,
                                constraints['opp_groen'],
                                "Groenoppervlakte")

    # Water
    add_min_max_constraints(prob, w,
                                constraints['norm_water_pw'],
                                "Waternorm",
                                multiplier=programma["woningen"])
    
    add_min_max_constraints(prob, w,
                                constraints['opp_water'],
                                "Wateroppervlakte")
    
    # Parking

    # Make sure there is enough parking
    total_parking = lpSum([ap[i] for i in parking_types])

    total_parking_required = parkeren["parkeren_tbv_wonen"] + parkeren["parkeren_tbv_nwonen_comm"] + parkeren["parkeren_tbv_nwonen_maat"]
    prob += total_parking >= total_parking_required
    prob += total_parking <= total_parking_required + 1

    prob += context["opp"] >= constraints['opp'] - 1 + eps
    prob += context["opp"] <= constraints['opp']

    # Grondwaarde
    add_min_max_constraints(prob, context["grondwaarde"],
                                constraints['grondwaarde'],
                                "Grondwaarde")



    return prob, (aw, yh, ap, vnw, g, w), (context, grondwaarde, ruimtelijk, programma, parkeren)



def set_pre_optimized_constraints(lp, context, grondwaarde, ruimtelijk, programma, parkeren, pre_optimized_for, pre_optimized_value):
    objective_variable, name = get_objective(context, grondwaarde,ruimtelijk, programma, parkeren, pre_optimized_for)
    lp += objective_variable >= pre_optimized_value - 0.5

def get_objective(context, grondwaarde, ruimtelijk, programma, parkeren, optimize_for):
    if optimize_for == 'Grondwaarde':
        return context["grondwaarde"], "Optimize_Grondwaarde"
    elif optimize_for == 'Woonprogramma':
        return programma["woningen"], "Optimize_Woningen"
    elif optimize_for == 'Uitgeefbaar':
        return ruimtelijk["opp_uitgeefbaar"], "Optimize_Uitgeefbaar"  
    elif optimize_for == 'Betaalbaar en sociaal woonprogramma':
        return programma["betaalbaar"] + programma["sociaal"], "Optimize_Betaalbaar_Sociaal_Woningen"
    else:
        raise Exception("Undefined objective for optimization problem")

# Split housing types (parents) into one type (child) per layer
# The df_wonen dataframe is replaced by a new one with new rows
# We end up with housing type parents (original types) and children (type + layer)
def split_layers(df, min_layers_series, max_layers_series, split_mask_series):
    df = df.copy()
    df["Min lagen"] = min_layers_series
    df["Max lagen"] = max_layers_series
    df["Split lagen"] = split_mask_series
    df["Lagen"] = min_layers_series

    df = df.reset_index()
    df["Type parent"] = df["Type"]

    new_rows = []
    type_parents = []
    for index, row in df.iterrows():
        type_parents.append(row["Type"])
        if row["Split lagen"]:
            opties_voor_lagen = range(int(row['Min lagen']), int(row['Max lagen']) + 1)
            for i in opties_voor_lagen:
                new_row = row.copy()
                new_row["Lagen"] = i
                new_row["Type"] = f"{row["Type"]} ({i} {'lagen' if i > 1 else 'laag'})"
                new_rows.append(new_row)
        else:
            new_rows.append(row)
    df_split = pd.DataFrame(columns=df.columns, data=new_rows).set_index("Type")
    return df_split

# IN PLACE!
def index_by_type(df):
    df["Type"] = clean_string_series(df["Type"])
    df.set_index("Type", inplace=True)

def clean_string_series(series):
    return series.str.strip().str.replace('[-\s]+', '_', regex=True)

# Inverts a map. So values become keys,
# and keys become (lists of) values (if values are not unique, which here they are typically not)
def inv_map(d):
    inv = {}
    for k, v in d.items():
        inv[v] = inv.get(v, []) + [k]
    return inv

def add_min_max_constraints(prob, var, mm, name, multiplier=1):
    min, max = mm
    if min is not None and not pd.isna(min):
        prob += var >= min * multiplier, f"Min_{name}"
    if max is not None and not pd.isna(max):
        prob += var <= max * multiplier, f"Max_{name}"

def to_dict_not_na(series):
    assert_column_filled(series)
    return series.to_dict()

def assert_column_filled(column):
    assert not column.isna().any(), f"Kolom '{column.name}' mist waarden. Vul de kolom helemaal in."