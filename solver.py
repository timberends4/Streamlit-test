import pulp

import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpInteger, LpStatus, value, PULP_CBC_CMD, LpContinuous
import numpy as np
from itertools import product


def calculate_priority_and_constraints(df_wonen, df_niet_wonen, df_constraints, df_constraints_perc):
    M = 1e8
    df_wonen['m2 nodig'] = df_wonen['BVO'] / df_wonen['Min lagen'] / df_wonen['Bebouwd']
    df_wonen['Grondwaarde/woning'] = df_wonen['Residuele grondwaarde per woning']
    df_wonen['Min lagen'] = df_wonen['Min lagen'].fillna(1)
    df_wonen['Max lagen'] = df_wonen['Max lagen'].fillna(100)

    df_niet_wonen['m2'] = 1
    df_niet_wonen['m2 nodig'] = df_niet_wonen['m2']
    df_niet_wonen['Residuele grondwaarde/m2'] = df_niet_wonen['Residuele grondwaarde m2 uitgeefbaar'] * df_niet_wonen['Uitgeefbaar']

    df_constraints['Min'] = pd.to_numeric(df_constraints['Min'], errors='coerce')
    df_constraints['Min'] = df_constraints['Min'].fillna(0)
    df_constraints['Max'] = pd.to_numeric(df_constraints['Max'], errors='coerce')
    df_constraints['Max'] = df_constraints['Max'].fillna(M)

    df_constraints_perc['Min'] = pd.to_numeric(df_constraints_perc['Min'], errors='coerce')
    df_constraints_perc['Min'] = df_constraints_perc['Min'].fillna(0)
    df_constraints_perc['Max'] = pd.to_numeric(df_constraints_perc['Max'], errors='coerce')
    df_constraints_perc['Max'] = df_constraints_perc['Max'].fillna(100)

    constraints = {
        #min constraints
        'min_opp': df_constraints.loc[df_constraints['Constraint'] == 'Oppervlakte gebied', 'Min'].values[0],
        'min_woningen': df_constraints.loc[df_constraints['Constraint'] == 'Woningen', 'Min'].values[0],
        'min_sociaal': df_constraints.loc[df_constraints['Constraint'] == 'Sociaal', 'Min'].values[0],
        'min_betaalbaar': df_constraints.loc[df_constraints['Constraint'] == 'Betaalbaar/ Middenhuur', 'Min'].values[0],
        'min_niet_wonen': df_constraints.loc[df_constraints['Constraint'] == 'Niet-wonen', 'Min'].values[0],
        'min_niet_wonen_comm': df_constraints.loc[df_constraints['Constraint'] == 'Niet-wonen (commercieel)', 'Min'].values[0],
        'min_grondwaarde': df_constraints.loc[df_constraints['Constraint'] == 'Grondwaarde', 'Min'].values[0],
        
        'min_perc_sociaal': df_constraints_perc.loc[df_constraints_perc['Constraint'] == 'Sociaal perc.', 'Min'].values[0]/100,
        'min_perc_betaalbaar': df_constraints_perc.loc[df_constraints_perc['Constraint'] == 'Betaalbaar/ Middenhuur perc.', 'Min'].values[0]/100,

        'min_norm_groen': df_constraints.loc[df_constraints['Constraint'] == 'Groennorm', 'Min'].values[0],
        'min_norm_water': df_constraints.loc[df_constraints['Constraint'] == 'Waternorm', 'Min'].values[0],
        'min_opp_groen': df_constraints.loc[df_constraints['Constraint'] == 'Groen', 'Min'].values[0],
        'min_opp_water': df_constraints.loc[df_constraints['Constraint'] == 'Water', 'Min'].values[0],

        #max constraints
        'max_opp': df_constraints.loc[df_constraints['Constraint'] == 'Oppervlakte gebied', 'Max'].values[0],
        'max_woningen': df_constraints.loc[df_constraints['Constraint'] == 'Woningen', 'Max'].values[0],
        'max_sociaal': df_constraints.loc[df_constraints['Constraint'] == 'Sociaal', 'Max'].values[0],
        'max_betaalbaar': df_constraints.loc[df_constraints['Constraint'] == 'Betaalbaar/ Middenhuur', 'Max'].values[0],
        'max_niet_wonen': df_constraints.loc[df_constraints['Constraint'] == 'Niet-wonen', 'Max'].values[0],
        'max_niet_wonen_comm': df_constraints.loc[df_constraints['Constraint'] == 'Niet-wonen (commercieel)', 'Max'].values[0],
        'max_grondwaarde': df_constraints.loc[df_constraints['Constraint'] == 'Grondwaarde', 'Max'].values[0],

        'max_perc_sociaal': df_constraints_perc.loc[df_constraints_perc['Constraint'] == 'Sociaal perc.', 'Max'].values[0]/100,
        'max_perc_betaalbaar': df_constraints_perc.loc[df_constraints_perc['Constraint'] == 'Betaalbaar/ Middenhuur perc.', 'Max'].values[0]/100,

        'max_norm_groen': df_constraints.loc[df_constraints['Constraint'] == 'Groennorm', 'Max'].values[0],
        'max_norm_water': df_constraints.loc[df_constraints['Constraint'] == 'Waternorm', 'Max'].values[0],
        'max_opp_groen': df_constraints.loc[df_constraints['Constraint'] == 'Groen', 'Max'].values[0],
        'max_opp_water': df_constraints.loc[df_constraints['Constraint'] == 'Water', 'Max'].values[0],
    }

    return df_wonen, df_niet_wonen, constraints

def solver(df_wonen, df_nwonen_comm, df_nwonen_maat, df_wonen_parkeren, df_parkeren_type, df_constraints, df_constraints_perc, df_kosten, optimize_for='Grondwaarde', optimize_for_2 = "Geen"):
    M = 1e10

    # If set to True, only one choice of layer per (parking/housing) type is possible.
    # If set to False, several choices of layers per type can be used
    # Mostly useful to benchmark against previous version
    homogenize_layers_wonen = False
    homogenize_layers_parking = False

    # ---- Creating DataFrames -----
    # Before turning the data into dictionaries, we modify dataframes.
    # This includes a join between df_wonen and df_wonen_parkeren,
    # as well as splitting of housing and parking types for possible layers

    df_wonen = df_wonen.copy()
    df_nwonen_comm = df_nwonen_comm.copy()
    df_nwonen_maat = df_nwonen_maat.copy()
    df_wonen_parkeren = df_wonen_parkeren.copy()
    df_parkeren_type = df_parkeren_type.copy()
    df_constraints = df_constraints.copy()
    df_constraints_perc = df_constraints_perc.copy()

    kosten_groen = df_kosten.loc[df_kosten['Kosten'] == 'Kosten Groen', 'Waarde'].values[0]
    kosten_water = df_kosten.loc[df_kosten['Kosten'] == 'Kosten Water', 'Waarde'].values[0]
    kosten_infra = df_kosten.loc[df_kosten['Kosten'] == 'Kosten Infrastructuur', 'Waarde'].values[0]
    bouwrijp_maken = df_kosten.loc[df_kosten['Kosten'] == 'Bouwrijp maken gebied', 'Waarde'].values[0]

    df_nwonen_comm['Commercieel'] = 1
    df_nwonen_comm['Maatschappelijk'] = 0

    df_nwonen_maat['Commercieel'] = 0
    df_nwonen_maat['Maatschappelijk'] = 1

    df_niet_wonen = pd.concat([df_nwonen_comm, df_nwonen_maat])

    df_wonen, df_niet_wonen, constraints = calculate_priority_and_constraints(df_wonen, df_niet_wonen, df_constraints, df_constraints_perc)

    df_wonen['Type'] = df_wonen['Type'].str.strip().str.replace('[-\s]+', '_', regex=True)
    df_niet_wonen['Type'] = df_niet_wonen['Type'].str.strip().str.replace('[-\s]+', '_', regex=True)
    df_wonen_parkeren['Type'] = df_wonen_parkeren['Type'].str.strip().str.replace('[-\s]+', '_', regex=True)
    df_parkeren_type['Type'] = df_parkeren_type['Type'].str.strip().str.replace('[-\s]+', '_', regex=True)

    df_niet_wonen['m2 nodig'] = df_niet_wonen['m2']
    df_niet_wonen['Residuele grondwaarde/m2'] = df_niet_wonen['Residuele grondwaarde m2 uitgeefbaar'] * df_niet_wonen['Uitgeefbaar'] # berekening?

    #  Fill unconstrained values of wonen en nwonen
    df_wonen['Min van type woning'] = df_wonen['Min van type woning'].fillna(0)
    df_wonen['Min woningen'] = df_wonen['Min woningen'].fillna(0)
    df_wonen['Max woningen'] = df_wonen['Max woningen'].fillna(M)

    df_niet_wonen['Min opp m2'] = df_niet_wonen['Min opp m2'].fillna(0)
    df_niet_wonen['Max opp m2'] = df_niet_wonen['Max opp m2'].fillna(M)

    df_wonen_parkeren['Parkeren buiten terrein model'] = df_wonen_parkeren['Parkeernorm voor model'] - df_wonen_parkeren['Parkeren op eigen terrein']

    # We add a relevant column from df_wonen_parkeren to df_wonen
    # We want to do this in the DataFrame to make our lives easier once we make the dictionaries
    # We use join to make sure it does not break down if one of the two data frames has a different order of the types
    df_wonen = df_wonen.set_index("Type").join(df_wonen_parkeren[["Type", 'Parkeren buiten terrein model']].set_index("Type")).reset_index()


    # Split housing types (parents) into one type (child) per layer
    # The df_wonen dataframe is replaced by a new one with new rows
    # We end up with housing type parents (original types) and children (type + layer)
    df_wonen["Lagen"] = df_wonen["Min lagen"]
    df_wonen["Type parent"] = df_wonen["Type"]
    new_rows = []
    housing_type_parents = []
    for index, row in df_wonen.iterrows():
        housing_type_parents.append(row["Type"])
        if row['Variabel aant. lagen']:
            opties_voor_lagen = range(int(row['Min lagen']), int(row['Max lagen']) + 1)
            for i in opties_voor_lagen:
                new_row = row.copy()
                new_row["Lagen"] = i
                new_row["Type"] = f"{row['Type']} ({i} {'lagen' if i > 1 else 'laag'})"
                new_rows.append(new_row)
        else:
            new_rows.append(row)
    df_wonen = pd.DataFrame(columns=df_wonen.columns, data=new_rows)

    # With each housing type having a set number of layers, we can compute Uitgeefbaar woning
    df_wonen["Uitgeefbaar woning"] = df_wonen["BVO"] / df_wonen["Lagen"] / df_wonen["Bebouwd"]


    # Split parking typologies into one per layer
    # Similar to the splitting per housing type

    df_parkeren_type["Lagen"] = 1
    df_parkeren_type["Type parent"] = df_parkeren_type["Type"]
    new_rows = []
    parking_type_parents = []
    for index, row in df_parkeren_type.iterrows():
        parking_type_parents.append(row["Type"])
        if int(row['Max aantal lagen']) > 1:
            opties_voor_lagen = range(1, int(row['Max aantal lagen']) + 1)
            for i in opties_voor_lagen:
                new_row = row.copy()
                new_row["Lagen"] = i
                new_row["Type"] = f"{row['Type']} ({i} {'lagen' if i > 1 else 'laag'})"
                new_rows.append(new_row)
        else:
            new_rows.append(row)
    df_parkeren_type = pd.DataFrame(columns=df_parkeren_type.columns, data=new_rows)

    # Only now do we compute all the types (for housing an parking these are actually the children)
    housing_types = df_wonen['Type'].to_list()
    niet_wonen_types = df_niet_wonen['Type'].to_list()
    parkeren_types = df_parkeren_type['Type'].to_list()

    # ---- Creating Dictionaries -----

    # Inverts a map. So values become keys,
    # and keys become (lists of) values (if values are not unique, which here they are typically not)
    def inv_map(d):
        inv = {}
        for k, v in d.items():
            inv[v] = inv.get(v, []) + [k]
        return inv

    # Create dictionaries for niet wonen types
    grondwaarde_per_nwoning = dict(zip(df_niet_wonen['Type'], df_niet_wonen['Residuele grondwaarde/m2']))
    maatschapppelijk = dict(zip(df_niet_wonen['Type'], df_niet_wonen['Maatschappelijk']))
    commercieel = dict(zip(df_niet_wonen['Type'], df_niet_wonen['Commercieel']))
    area_per_nwoning = dict(zip(df_niet_wonen['Type'], df_niet_wonen['m2 nodig']))
    pp_per_nwoning = dict(zip(df_niet_wonen['Type'], df_niet_wonen['Benodigde PP per m2']))
    min_nwonen = dict(zip(df_niet_wonen['Type'], df_niet_wonen['Min opp m2']))
    max_nwonen = dict(zip(df_niet_wonen['Type'], df_niet_wonen['Max opp m2']))
    uitgeefbaar_nwoning = dict(zip(df_niet_wonen['Type'], df_niet_wonen['Uitgeefbaar']))


    # Create dictionaries for wonen types
    housing_type_parent = dict(zip(df_wonen["Type"], df_wonen["Type parent"]))
    housing_type_children = inv_map(housing_type_parent)
    grondwaarde_per_woning = dict(zip(df_wonen['Type'], df_wonen['Residuele grondwaarde per woning']))
    is_sociaal = dict(zip(df_wonen['Type'], df_wonen['Sociaal']))
    is_betaalbaar = dict(zip(df_wonen['Type'], df_wonen['Betaalbaar/ Middenduur']))
    min_wonen_type = dict(zip(df_wonen['Type'], df_wonen['Min van type woning']))
    min_wonen = dict(zip(df_wonen['Type'], df_wonen['Min woningen']))
    max_wonen = dict(zip(df_wonen['Type'], df_wonen['Max woningen']))
    uitgeefbaar_woning = dict(zip(df_wonen['Type'], df_wonen['Uitgeefbaar woning']))
    parkeren_buiten_terrein = dict(zip(df_wonen['Type'], df_wonen["Parkeren buiten terrein model"]))

    # Create dictionaries for parkeren types
    parking_type_parent = dict(zip(df_parkeren_type['Type'], df_parkeren_type['Type parent']))
    parking_type_children = inv_map(parking_type_parent)
    lagen_pp = dict(zip(df_parkeren_type['Type'], df_parkeren_type['Lagen']))
    bvo_pp = dict(zip(df_parkeren_type['Type'], df_parkeren_type['Opp m2 BVO per P.P.']))
    res_grondwaarde_pp = dict(zip(df_parkeren_type['Type'], df_parkeren_type['Res. Grondwaarde per P.P.']))
    discount_factor_pp = dict(zip(df_parkeren_type['Type'], df_parkeren_type['Verminderende Factor Grondwaarde per laag']))


    # Builds the LP.  uses the function variables
    # Optimize for label is the name of the LP objective (as it is written in the UI)
    # If pre-optimized for is also such a label, then the corresponding objective will be fixed at
    # pre-optimized-value in the LP. Hence, the objective becomes a "second objective"
    def build_lp(optimize_for_label, pre_optimized_for = None, pre_optimized_value = None):

        if optimize_for == 'Grondwaarde':
            prob = LpProblem("Optimize_Grondwaarde", LpMaximize)
        elif optimize_for == 'Woonprogramma':
            prob = LpProblem("Optimize_Woningen", LpMaximize)
        elif optimize_for == 'Uitgeefbaar':
            prob = LpProblem("Optimize_Uitgeefbaar", LpMaximize)
        elif optimize_for == 'Betaalbaar en sociaal woonprogramma':
            prob = LpProblem("Optimize_Betaalbaar_Sociaal_Woningen", LpMaximize)
        else:
            raise Exception("Undefined objective for optimization problem")

        # Define variables
        x = LpVariable.dicts("Aantal_woning", housing_types, lowBound=0, cat=LpInteger)
        y = LpVariable.dicts("Binary", housing_types, cat="Binary")  # Binary variables, whether housing type placed
        a = LpVariable.dicts("Aantal_parkeren", parkeren_types, lowBound=0, cat=LpInteger)
        z = LpVariable.dicts("Aantal_niet_wonen", niet_wonen_types, lowBound=0, cat=LpInteger)



        # Objective function
        grondwaarde_woningen = lpSum([grondwaarde_per_woning[i] * x[i] for i in housing_types])
        grondwaarde_niet_wonen = lpSum([grondwaarde_per_nwoning[i] * z[i] for i in niet_wonen_types])
        grondwaarde_parkeren = lpSum([res_grondwaarde_pp[i] * a[i] *
                                      (1 - (discount_factor_pp[i] * lagen_pp[i])) for i
                                      in parkeren_types])

        # Compute areas
        total_woningen = lpSum([x[i] for i in housing_types])

        infra_area = (lpSum(
            [x[i] * df_wonen.loc[df_wonen['Type'] == i, 'Infra niet PP'].values[0] for i in housing_types])
                      + lpSum(
                    [z[j] * df_niet_wonen.loc[df_niet_wonen['Type'] == j, 'Benodigde infra'].values[0] for j in
                     niet_wonen_types]))


        # def df_val(df, idx_col, idx, column):
        #     return df.loc[df[idx_col] == idx, column].values[0]

        total_area_wonen = lpSum([x[i] * uitgeefbaar_woning[i]
                                  for i in housing_types])  # Variabel lagen

        # === Constraints ===

        # --- Min type woningen constraint ---
        # This is layer-specific
        for i in housing_types:
            prob += x[i] >= min_wonen_type[i] * y[i], f"Conditional_Min_{i}"  # Ensure minimum if placed
            prob += x[i] <= M * y[i], f"Placed_{i}"

        # --- Homogenize layers within housing type ---
        if homogenize_layers_wonen:
            for htp in housing_type_parents:
                print(housing_type_children)
                prob += (lpSum([y[ht] for ht in housing_type_children[htp]]) <= 1,
                         f"Homogenize layers housing type {htp}")

        if homogenize_layers_parking:
            yp = LpVariable.dicts("Binary", parkeren_types,
                                  cat="Binary")  # Binary variables, whether housing type placed
            for pt in parkeren_types:
                prob += a[pt] <= M * yp[pt], f"Used {pt}"

            for ptp in parking_type_parents:
                prob += (lpSum([yp[pt] for pt in parking_type_children[ptp]]) <= 1,
                         f"Homogenize layers parking type {ptp}")

        # --- Min and max per type parent ----

        # The "Min" and "Max" are bounds on the total number of a type parent
        # Since the variables are for type children (orig. type + nr of layers),
        # this is a constraint on a sum of variables
        for tp in housing_type_parents:
            sum_amount_children = lpSum([x[t] for t in housing_type_children[tp] ])
            # IMPORTANT: we assume here the Min and Max woningen are EQUAL among housing type children.
            # In practice this is true because we don't touch it when splitting the types
            prob += sum_amount_children >= min_wonen[housing_type_children[tp][0]], f"Min_{tp}"
            prob += sum_amount_children <= max_wonen[housing_type_children[tp][0]], f"Max_{tp}"

        # Niet wonen is not split, so there are no parents/children, just types.
        for i in niet_wonen_types:
            prob += z[i] >= min_nwonen[i], f"Min_{i}"
            prob += z[i] <= max_nwonen[i], f"Max_{i}"


        # Parkeren
        total_parking_required_wonen = lpSum(
            [parkeren_buiten_terrein[i] * x[i]
             for i in housing_types])  # 12 is the m2 per parking spot
        total_parking_required_niet_wonen = lpSum([pp_per_nwoning[i] * z[i] for i in niet_wonen_types])

        total_parking_required = total_parking_required_wonen + total_parking_required_niet_wonen

        prob += lpSum([a[i] for i in parkeren_types]) >= total_parking_required, "Total_Parking"

        # Niet wonen area constraint
        total_area_niet_wonen = lpSum([area_per_nwoning[i] * z[i] for i in niet_wonen_types])

        total_woningen = lpSum([x[i] for i in housing_types])

        # Minimum and maximum number of houses
        prob += total_woningen >= constraints['min_woningen'], "Min_Woningen"
        prob += total_woningen <= constraints['max_woningen'], "Max_Woningen"

        # Sociaal housing constraints
        total_sociaal = lpSum([is_sociaal[i] * x[i] for i in housing_types])
        prob += total_sociaal >= constraints['min_sociaal'], "Min_Sociaal"
        prob += total_sociaal <= constraints['max_sociaal'], "Max_Sociaal"

        # Since percentage constraints involve division, we need to linearize them
        prob += total_sociaal >= constraints['min_perc_sociaal'] * total_woningen, "Min_Perc_Sociaal"
        prob += total_sociaal <= constraints['max_perc_sociaal'] * total_woningen, "Max_Perc_Sociaal"

        total_betaalbaar = lpSum([is_betaalbaar[i] * x[i] for i in housing_types])

        # Betaalbaar housing constraints
        prob += total_betaalbaar >= constraints['min_betaalbaar'], "Min_Betaalbaar"
        prob += total_betaalbaar <= constraints['max_betaalbaar'], "Max_Betaalbaar"

        # Since percentage constraints involve division, we need to linearize them
        prob += total_betaalbaar >= constraints['min_perc_betaalbaar'] * total_woningen, "Min_Perc_Betaalbaar"
        prob += total_betaalbaar <= constraints['max_perc_betaalbaar'] * total_woningen, "Max_Perc_Betaalbaar"

        # Niet wonen constraints
        total_nwonen = lpSum([z[i] for i in niet_wonen_types])
        prob += total_nwonen >= constraints['min_niet_wonen'], "Min_Niet_Wonen"
        prob += total_nwonen <= constraints['max_niet_wonen'], "Max_Niet_Wonen"


        # Commercieel niet wonen constraints
        total_commercieel = lpSum([commercieel[i] * z[i] for i in niet_wonen_types])
        prob += total_commercieel >= constraints[
            'min_niet_wonen_comm'], "Min_Niet_Wonen_Comm"
        prob += total_commercieel <= constraints[
            'max_niet_wonen_comm'], "Max_Niet_Wonen_Comm"

        # == Groen en Water norm == #

        # Total parking footprint for Parkeerplaatsen
        total_parking_footprint = lpSum([
            (a[i] * bvo_pp[i]) / lagen_pp[i]
            for i in parkeren_types
        ])


        remaining_available_area = constraints[
                                       'max_opp'] - total_area_wonen - total_parking_footprint - total_area_niet_wonen - infra_area

        g = LpVariable("g", lowBound=0, cat="LpInteger")
        prob += g >= total_woningen * constraints['min_norm_groen']
        prob += g >= constraints['min_opp_groen']
        prob += g <= constraints['max_opp_groen']

        w = LpVariable("w", lowBound=0, cat="LpInteger")
        prob += w >= total_woningen * constraints['min_norm_water']
        prob += w >= constraints['min_opp_water']
        prob += w <= constraints['max_opp_water']

        prob += g + w <= remaining_available_area


        # ====  Cost components =====

        total_opp_gebied = total_area_wonen + total_area_niet_wonen + total_parking_footprint + infra_area + g + w

        cost_groen = g * kosten_groen
        cost_water = w * kosten_water

        cost_infra = infra_area * kosten_infra
        cost_bouwrijp = total_opp_gebied * bouwrijp_maken



        # full area constraint
        prob += total_opp_gebied <= constraints['max_opp'], "Max_Oppervlakte_Gebied"
        prob += total_opp_gebied >= constraints['min_opp'], "Min_Oppervlakte_Gebied"

        # === OBJECTIVES === #

        # Put possible objectives in a variable so they can be easily extracted if used for pre-optimization
        gw = LpVariable(name="Net_Grondwaarde", lowBound=None, upBound=None, cat=LpContinuous)
        prob += gw == (grondwaarde_parkeren
                    + grondwaarde_woningen
                    + grondwaarde_niet_wonen
                    - cost_groen
                    - cost_water
                    - cost_infra
                    - cost_bouwrijp), "Net_Grondwaarde_constr"

        # === grondwaarde constraint ===
        prob += gw >= constraints['min_grondwaarde'], "Min_Grondwaarde"
        prob += gw <= constraints['max_grondwaarde'], "Max_Grondwaarde"


        tw = LpVariable(name="Total_Woningen", lowBound=None, upBound=None, cat=LpContinuous)
        prob += tw == total_woningen, "Total_Woningen_constr"

        totaal_uitgeefbaar = LpVariable(name="Totaal_Uitgeefbaar", lowBound=None, upBound=None, cat=LpContinuous)
        prob += totaal_uitgeefbaar == (lpSum([x[t] * uitgeefbaar_woning[t] for t in housing_types])
                                       + lpSum([z[t] * uitgeefbaar_nwoning[t] for t in niet_wonen_types]))

        totaal_sociaal_en_betaalbaar = LpVariable(name="Totaal_Sociaal_Betaalbaar", lowBound=None, upBound=None, cat=LpContinuous)
        prob += totaal_sociaal_en_betaalbaar == (total_sociaal + total_betaalbaar)

        # Fix pre-optimized values in the LP
        if pre_optimized_for == 'Grondwaarde':
            print(f"Fix grondwaarde at {pre_optimized_value}")
            # We need this -1 here or we will get infeasibility complaints. Probably has to do with rounding.
            prob += gw >= pre_optimized_value - 1, f"Fix grondwaarde at {pre_optimized_value}"

        elif pre_optimized_for == 'Woonprogramma':
            prob += tw == pre_optimized_value, f"Fix total woningen at {pre_optimized_value}"

        elif pre_optimized_for == "Uitgeefbaar":
            # TODO check if this gives problems
            prob += totaal_uitgeefbaar >= pre_optimized_value

        elif pre_optimized_for == "Betaalbaar en sociaal woonprogramma":
            prob += totaal_sociaal_en_betaalbaar == pre_optimized_value


        # Add the actual objective
        if optimize_for_label == 'Grondwaarde':
            prob += gw, "Max_Grondwaarde"

        elif optimize_for_label == 'Woonprogramma':
            prob += tw, "Max_Woningen"

        elif optimize_for_label == "Uitgeefbaar":
            prob += totaal_uitgeefbaar

        elif optimize_for_label == "Betaalbaar en sociaal woonprogramma":
            prob += totaal_sociaal_en_betaalbaar

        #TODO objective uitgeefbaar = sum uitgeefbaar * aantal van wonen, niet-wonen
        # TODO sociaal /betaalbaar = sum aantal van die parents

        return prob

    print("Starting optimize.....")
    print("First objective", optimize_for)

    prob = build_lp(optimize_for)

    # Solve the problem with first objective
    prob.solve(PULP_CBC_CMD(msg=0))

    # Output the solver status
    print("Status:", LpStatus[prob.status])
    if prob.status == 1:
        # Output the results
        print(f"Optimal solution found for objective 1: {optimize_for}. Solution:")
        for v in prob.variables():
            print(v.name, "=", v.varValue)

    elif prob.status != 1:
        for  name , constraint in prob.constraints.items():
            print(name, ":", constraint.value())
        print("No solution found for objective 1")
        raise ValueError("No solution found for given input and constraints")

    # Solve with second objective if necessary
    if prob.status == 1 and optimize_for_2 != "Geen" and optimize_for_2 != optimize_for:
        # check the original objective so we pass its optimal value for the second round of optimization.
        if optimize_for == "Grondwaarde":
            max_grondwaarde = prob.variablesDict()['Net_Grondwaarde'].value()
            print(f"Solving for second objective {optimize_for_2}, value {max_grondwaarde} for {optimize_for}")
            prob = build_lp(optimize_for_2, pre_optimized_for=optimize_for, pre_optimized_value=max_grondwaarde)

        elif optimize_for == "Woonprogramma":
            total_woningen = prob.variablesDict()['Total_Woningen'].value()
            print(f"Solving for second objective {optimize_for_2}, value {total_woningen} for {optimize_for}")
            prob = build_lp(optimize_for_2, pre_optimized_for=optimize_for, pre_optimized_value=total_woningen)

        elif optimize_for == "Uitgeefbaar":
            total_uitgeefbaar = prob.variablesDict()['Totaal_Uitgeefbaar'].value()
            print(f"Solving for second objective {optimize_for_2}, value {total_uitgeefbaar} for {optimize_for}")
            prob = build_lp(optimize_for_2, pre_optimized_for=optimize_for, pre_optimized_value=total_uitgeefbaar)

        elif optimize_for == "Betaalbaar en sociaal woonprogramma":
            total_sociaal_betaalbaar = prob.variablesDict()['Totaal_Sociaal_Betaalbaar'].value()
            print(f"Solving for second objective {optimize_for_2}, value {total_sociaal_betaalbaar} for {optimize_for}")
            prob = build_lp(optimize_for_2, pre_optimized_for=optimize_for, pre_optimized_value=total_sociaal_betaalbaar)
        else:
            raise ValueError(f"Second objective {optimize_for_2} not implemented.")

        prob.solve(PULP_CBC_CMD(msg=0))

        print("Did the second objective!")

    # Extract the solution. Could be the LP for the first objective, or it could be the one for the second objective.
    solution = None
    if prob.status == 1:
        solution = {v.name: v.varValue for v in prob.variables() if v.varValue is not None}
        parking_house_area = 0

        # Match solution variables with df_wonen 'Type' dynamically
        for index, row in df_wonen_parkeren.iterrows():
            for var_name, var_value in solution.items():
                if row['Type'].replace(' ', '_') in var_name:  # Match housing type in variable name
                    parking_house_area += var_value * row['Parkeren op eigen terrein']

        # Output the results
        print("Optimal solution found. Solution:")
        for v in prob.variables():
            print(v.name, "=", v.varValue)
    else:
        raise ValueError("No solution found for given input and constraints")


    # Extract final chosen solution and layers
    final_solution = solution

    # Prepare helper functions to safely get solution values
    def sol_val(var_prefix, t):
        var_name = f"{var_prefix}_{t.replace('-', '_').replace(' ', '_')}"
        if var_name not in final_solution:
            raise ValueError(f"Variable {var_name} not found in solution. Available variables are: {final_solution.keys()}")
        return final_solution[var_name]


    # Fill df_wonen_out
    wonen_rows = []
    for tp in housing_type_parents:
        num_used = 0
        for t in housing_type_children[tp]:
            aantal = sol_val("Aantal_woning", t)
            # Only show a row with this number of layers if it is actually used
            if aantal > 0:
                num_used += 1
                lagen = df_wonen.loc[df_wonen['Type'] == t, 'Lagen'].values[0]
                uitgeefbaar_per_woning = df_wonen.loc[df_wonen['Type'] == t, 'Uitgeefbaar'].values[0]
                grondwaarde_woning = grondwaarde_per_woning[t] * aantal
                wonen_rows.append({
                    'Type': tp.replace('_', ' '),
                    'Aantal': aantal,
                    'Lagen': lagen,
                    'Uitgeefbaar m2': aantal * uitgeefbaar_per_woning,
                    'Grondwaarde': grondwaarde_woning
                })
        # If none of this housing type are used at all, we still add a row indicating that none were used
        if num_used == 0:
            wonen_rows.append({
                'Type': tp.replace('_', ' '),
                'Aantal': 0,
                'Lagen': 0,
                'Uitgeefbaar m2': 0,
                'Grondwaarde': 0
            })

    wonen_rows.append({
        'Type': 'Totaal Sociaal',
        'Aantal': sum(sol_val("Aantal_woning", i) * is_sociaal[i] for i in housing_types),
        'Lagen': 'n.v.t',
        'Uitgeefbaar m2': sum(sol_val("Aantal_woning", i) * is_sociaal[i] * df_wonen.loc[df_wonen['Type'] == i, 'Uitgeefbaar woning'].values[0] for i in housing_types),
        'Grondwaarde': sum(sol_val("Aantal_woning", i) * grondwaarde_per_woning[i] * is_sociaal[i] for i in housing_types)
    })
    wonen_rows.append({
        'Type': 'Totaal Betaalbaar',
        'Aantal': sum(sol_val("Aantal_woning", i) * is_betaalbaar[i] for i in housing_types),	
        'Lagen': 'n.v.t',
        'Uitgeefbaar m2': sum(df_wonen.loc[df_wonen['Type'] == i, 'Uitgeefbaar woning'].values[0] * sol_val("Aantal_woning", i) * is_betaalbaar[i] for i in housing_types),
        'Grondwaarde': sum(sol_val("Aantal_woning", i) * grondwaarde_per_woning[i] * is_betaalbaar[i] for i in housing_types)
    })

    wonen_rows.append({
        'Type': 'Totaal',
        'Aantal': sum(sol_val("Aantal_woning", i) for i in housing_types),
        'Lagen': 'n.v.t',
        'Uitgeefbaar m2': sum(df_wonen.loc[df_wonen['Type'] == i, 'Uitgeefbaar woning'].values[0] * sol_val("Aantal_woning", i) for i in housing_types),
        'Grondwaarde': sum(sol_val("Aantal_woning", i) * grondwaarde_per_woning[i] for i in housing_types)
    })
    df_wonen_out = pd.DataFrame(wonen_rows, columns=['Type', 'Aantal', 'Lagen', 'Uitgeefbaar m2', 'Grondwaarde'])

    # Fill df_nwonen_comm_out and df_nwonen_maat_out
    comm_rows = []
    maat_rows = []
    for t in niet_wonen_types:
        aantal = sol_val("Aantal_niet_wonen", t)
        uitgeefbaar_per_type = df_niet_wonen.loc[df_niet_wonen['Type'] == t, 'Uitgeefbaar'].values[0]
        grondwaarde_nwonen = grondwaarde_per_nwoning[t] * aantal

        row = {
            'Type': t.replace('_', ' '),
            'Aantal': aantal,
            'Uitgeefbaar m2': aantal * uitgeefbaar_per_type,
            'Grondwaarde': grondwaarde_nwonen
        }

        if commercieel[t] == 1:
            comm_rows.append(row)
        else:
            maat_rows.append(row)

    comm_rows.append({
        'Type': 'Totaal Commercieel',
        'Aantal': sum(sol_val("Aantal_niet_wonen", j) * commercieel[j] for j in niet_wonen_types),
        'Uitgeefbaar m2': sum(df_niet_wonen.loc[df_niet_wonen['Type'] == j, 'Uitgeefbaar'].values[0] * sol_val("Aantal_niet_wonen", j) * commercieel[j] for j in niet_wonen_types),
        'Grondwaarde': sum(sol_val("Aantal_niet_wonen", j) * grondwaarde_per_nwoning[j] * commercieel[j] for j in niet_wonen_types)
    })

    maat_rows.append({
        'Type': 'Totaal Maatschappelijk',
        'Aantal': sum(sol_val("Aantal_niet_wonen", j) * maatschapppelijk[j] for j in niet_wonen_types),
        'Uitgeefbaar m2': sum(df_niet_wonen.loc[df_niet_wonen['Type'] == j, 'Uitgeefbaar'].values[0] * sol_val("Aantal_niet_wonen", j) * maatschapppelijk[j] for j in niet_wonen_types),
        'Grondwaarde': sum(sol_val("Aantal_niet_wonen", j) * grondwaarde_per_nwoning[j] * maatschapppelijk[j] for j in niet_wonen_types)
    })

    df_nwonen_comm_out = pd.DataFrame(comm_rows, columns=['Type', 'Aantal', 'Uitgeefbaar m2', 'Grondwaarde'])
    df_nwonen_maat_out = pd.DataFrame(maat_rows, columns=['Type', 'Aantal', 'Uitgeefbaar m2', 'Grondwaarde'])

    # Fill df_parkeren_out
    parkeren_rows = []
    for tp in parking_type_parents:
        num_used = 0
        for t in parking_type_children[tp]:
            aantal_pp = sol_val("Aantal_parkeren", t)
            # Only show if this number of parking layers is used
            if aantal_pp > 0:
                num_used += 1
                totaal_bvo = aantal_pp * bvo_pp[t]
                totaal_res_grondwaarde = aantal_pp * res_grondwaarde_pp[t] * (1 - (discount_factor_pp[t] * lagen_pp[t]))
                footprint = (aantal_pp * bvo_pp[t]) /  lagen_pp[t]
                parkeren_rows.append({
                    'Type': parking_type_parent[t].replace('_', ' '),
                    'Aantal P.P': aantal_pp,
                    'Aantal lagen': lagen_pp[t],
                    'Totaal BVO': totaal_bvo,
                    'Totaal res. grondwaarde': totaal_res_grondwaarde,
                    'Footprint': footprint
                })
        # If this type of parking is not used at all, still show one row
        if num_used == 0:
            parkeren_rows.append({
                'Type': parking_type_parent[t].replace('_', ' '),
                'Aantal P.P': 0,
                'Aantal lagen': 0,
                'Totaal BVO': 0,
                'Totaal res. grondwaarde': 0,
                'Footprint': 0
            })

    df_parkeren_out = pd.DataFrame(parkeren_rows, columns=['Type', 'Aantal P.P', 'Aantal lagen', 'Totaal BVO', 'Totaal res. grondwaarde', 'Footprint'])

    # Fill df_ruimte_out
    total_woningen_val = sum(sol_val("Aantal_woning", i) for i in housing_types)
    total_area_wonen_val = sum(uitgeefbaar_woning[i] * sol_val("Aantal_woning", i) for i in housing_types)
    total_area_niet_wonen_val = sum(area_per_nwoning[i] * sol_val("Aantal_niet_wonen", i) for i in niet_wonen_types)
    total_parking_footprint_val = sum((sol_val("Aantal_parkeren", i) * bvo_pp[i]) / lagen_pp[i] for i in parkeren_types)

    total_groen_allocated = final_solution['g']
    total_water_allocated = final_solution['w']

    total_infra = (sum(sol_val("Aantal_woning", i) * df_wonen.loc[df_wonen['Type'] == i, 'Infra niet PP'].values[0] for i in housing_types)
                   + sum(sol_val("Aantal_niet_wonen", j) * df_niet_wonen.loc[df_niet_wonen['Type'] == j, 'Benodigde infra'].values[0] for j in niet_wonen_types))
    
    total_opp_gebied = total_area_wonen_val + total_area_niet_wonen_val + total_parking_footprint_val + total_infra + total_groen_allocated + total_water_allocated
    # We can also compute uitgeefbaar and infra:
    total_uitgeefbaar = (sum(sol_val("Aantal_woning", i) * df_wonen.loc[df_wonen['Type'] == i, 'Uitgeefbaar woning'].values[0] for i in housing_types)
                         + sum(sol_val("Aantal_niet_wonen", j) * df_niet_wonen.loc[df_niet_wonen['Type'] == j, 'Uitgeefbaar'].values[0] for j in niet_wonen_types))
 
    grondwaarde_woningen = sum(sol_val("Aantal_woning", i) * grondwaarde_per_woning[i] for i in housing_types)
    grondwaarde_nwonen = sum(sol_val("Aantal_niet_wonen", j) * grondwaarde_per_nwoning[j] for j in niet_wonen_types)
    grondwaarde_parkeren = sum(sol_val("Aantal_parkeren", k) * res_grondwaarde_pp[k] * (1 - (discount_factor_pp[k] * lagen_pp[k]) ) for k in parkeren_types)

    cost_groen = total_groen_allocated * kosten_groen
    cost_water = total_water_allocated * kosten_water
    cost_infra = total_infra * kosten_infra
    cost_bouwrijp = total_opp_gebied * bouwrijp_maken

    total_grondwaarde = grondwaarde_woningen + grondwaarde_nwonen + grondwaarde_parkeren - cost_groen - cost_water - cost_infra - cost_bouwrijp
    
    df_ruimte_out = pd.DataFrame([
    ["Totaal opp. Gebied", "m2", int(total_opp_gebied)],
    ["Uitgeefbaar", "m2", int(total_uitgeefbaar)],
    ["Grondwaarde", "€", int(total_grondwaarde)],
    ["Woonprogramma Sociaal perc.", "percentage", 
     round((sum(sol_val("Aantal_woning", i) * is_sociaal[i] for i in housing_types) / total_woningen_val) * 100, 0) if total_woningen_val != 0 else 0],
    ["Woonprogramma Betaalbaar perc.", "percentage", 
     round((sum(sol_val("Aantal_woning", i) * is_betaalbaar[i] for i in housing_types) / total_woningen_val) * 100, 0) if total_woningen_val != 0 else 0],                                             
    ["Groen", "m2", int(total_groen_allocated)],
    ["Water", "m2", int(total_water_allocated)],
    ["Infra", "m2", int(total_infra)],
    ["Uitgeefbaar", "m2", int(total_uitgeefbaar)],
    ["Groennorm", "m2/woning", int(constraints['min_norm_groen'])],
    ["Waternorm", "m2/woning", int(constraints['min_norm_water'])]
    ], columns=['Output', 'Eenheid', 'Hoeveelheid'])


    return df_wonen_out, df_nwonen_comm_out, df_nwonen_maat_out, df_ruimte_out, df_parkeren_out 