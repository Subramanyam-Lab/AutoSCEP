import os
import random
import json
from scenario_random import generate_random_scenario
from yaml import safe_load

def create_scenarios(num_scenarios,output_dir):
    """
    Generate and save random scenarios to individual folders.

    Parameters:
        num_scenarios (int): Number of scenarios to generate.
        output_dir (str): Path to save scenarios.
        config (dict): Configuration for generating scenarios.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create base directory if not exists

    UserRunTimeConfig = safe_load(open("config_reducedrun.yaml"))

    # Extract all the configuration variables as in your original code
    USE_TEMP_DIR = UserRunTimeConfig["USE_TEMP_DIR"]
    temp_dir = UserRunTimeConfig["temp_dir"]
    version = UserRunTimeConfig["version"]
    Horizon = UserRunTimeConfig["Horizon"]
    # NoOfScenarios = UserRunTimeConfig["NoOfScenarios"]
    NoOfScenarios = 1
    lengthRegSeason = UserRunTimeConfig["lengthRegSeason"]
    discountrate = UserRunTimeConfig["discountrate"]
    WACC = UserRunTimeConfig["WACC"]
    solver = UserRunTimeConfig["solver"]
    scenariogeneration = UserRunTimeConfig["scenariogeneration"]
    fix_sample = UserRunTimeConfig["fix_sample"]
    LOADCHANGEMODULE = UserRunTimeConfig["LOADCHANGEMODULE"]
    filter_make = UserRunTimeConfig["filter_make"] 
    filter_use = UserRunTimeConfig["filter_use"]
    n_cluster = UserRunTimeConfig["n_cluster"]
    moment_matching = UserRunTimeConfig["moment_matching"]
    n_tree_compare = UserRunTimeConfig["n_tree_compare"]
    EMISSION_CAP = UserRunTimeConfig["EMISSION_CAP"]
    IAMC_PRINT = UserRunTimeConfig["IAMC_PRINT"]
    WRITE_LP = UserRunTimeConfig["WRITE_LP"]
    PICKLE_INSTANCE = UserRunTimeConfig["PICKLE_INSTANCE"] 

    # Non-configurable settings
    NoOfRegSeason = 4
    regular_seasons = ["winter", "spring", "summer", "fall"]
    NoOfPeakSeason = 2
    lengthPeakSeason = 24
    LeapYearsInvestment = 5
    time_format = "%d/%m/%Y %H:%M"
    if version in ["europe_v50"]:
        north_sea = False
    elif version in ["reduced"]:
        north_sea = False
    else:
        north_sea = True


    workbook_path = 'Data handler/' + version
    scenario_data_path = 'Data handler/' + version + '/ScenarioData' 

    FirstHoursOfRegSeason = [lengthRegSeason*i + 1 for i in range(NoOfRegSeason)]
    FirstHoursOfPeakSeason = [lengthRegSeason*NoOfRegSeason + lengthPeakSeason*i + 1 for i in range(NoOfPeakSeason)]
    Period = [i + 1 for i in range(int((Horizon-2020)/LeapYearsInvestment))]
    Scenario = ["scenario"+str(i + 1) for i in range(NoOfScenarios)]
    peak_seasons = ['peak'+str(i + 1) for i in range(NoOfPeakSeason)]
    Season = regular_seasons + peak_seasons
    Operationalhour = [i + 1 for i in range(FirstHoursOfPeakSeason[-1] + lengthPeakSeason - 1)]
    HoursOfRegSeason = [(s,h) for s in regular_seasons for h in Operationalhour \
                    if h in list(range(regular_seasons.index(s)*lengthRegSeason+1,
                                regular_seasons.index(s)*lengthRegSeason+lengthRegSeason+1))]
    HoursOfPeakSeason = [(s,h) for s in peak_seasons for h in Operationalhour \
                        if h in list(range(lengthRegSeason*len(regular_seasons)+ \
                                            peak_seasons.index(s)*lengthPeakSeason+1,
                                            lengthRegSeason*len(regular_seasons)+ \
                                                peak_seasons.index(s)*lengthPeakSeason+ \
                                                    lengthPeakSeason+1))]
    
    HoursOfSeason = HoursOfRegSeason + HoursOfPeakSeason
    # dict_countries = {"AT": "Austria", "BA": "BosniaH", "BE": "Belgium",
    #             "BG": "Bulgaria", "CH": "Switzerland", "CZ": "CzechR",
    #             "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
    #             "ES": "Spain", "FI": "Finland", "FR": "France",
    #             "GB": "GreatBrit.", "GR": "Greece", "HR": "Croatia",
    #             "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
    #             "LT": "Lithuania", "LU": "Luxemb.", "LV": "Latvia",
    #             "MK": "Macedonia", "NL": "Netherlands", "NO": "Norway",
    #             "PL": "Poland", "PT": "Portugal", "RO": "Romania",
    #             "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia",
    #             "SK": "Slovakia", "MF": "MorayFirth", "FF": "FirthofForth",
    #             "DB": "DoggerBank", "HS": "Hornsea", "OD": "OuterDowsing",
    #             "NF": "Norfolk", "EA": "EastAnglia", "BS": "Borssele",
    #             "HK": "HollandseeKust", "HB": "HelgolanderBucht", "NS": "Nordsoen",
    #             "UN": "UtsiraNord", "SN1": "SorligeNordsjoI", "SN2": "SorligeNordsjoII"}

    dict_countries = {"DE": "Germany", "DK": "Denmark", "FR": "France"}
        
            
        
    for scenario_num in range(1, num_scenarios + 1):
        scenario_folder = os.path.join(output_dir, f"{scenario_num}")
        os.makedirs(scenario_folder, exist_ok=True)

        generate_random_scenario(
                filepath=scenario_data_path,
                tab_file_path=scenario_folder,
                scenarios=NoOfScenarios,
                seasons=regular_seasons,
                Periods=len(Period),
                regularSeasonHours=lengthRegSeason,
                peakSeasonHours=lengthPeakSeason,
                dict_countries=dict_countries,
                time_format=time_format,
                filter_make=filter_make,
                filter_use=filter_use,
                n_cluster=n_cluster,
                moment_matching=moment_matching,
                n_tree_compare=n_tree_compare,
                fix_sample=fix_sample,
                north_sea=north_sea,
                LOADCHANGEMODULE=LOADCHANGEMODULE,
                seed=scenario_num
            )

        print(f"Scenario {scenario_num} generated and saved to {scenario_folder}")


if __name__ == "__main__":
    output_directory = "Data handler/random_sce_reduced/"
    num_scenarios = 1000
    create_scenarios(num_scenarios, output_directory)
