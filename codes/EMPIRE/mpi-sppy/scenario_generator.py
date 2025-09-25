from scenario_random_mpi import generate_random_scenario
from yaml import safe_load
from datetime import datetime

def scenario_generator(SEED,scenario_idx,num_sce):
    # UserRunTimeConfig = safe_load(open("config_reducedrun.yaml"))
    UserRunTimeConfig = safe_load(open("config_run.yaml"))

    version = UserRunTimeConfig["version"]
    Horizon = UserRunTimeConfig["Horizon"]
    # NoOfScenarios = UserRunTimeConfig["NoOfScenarios"]
    NoOfScenarios = num_sce
    lengthRegSeason = UserRunTimeConfig["lengthRegSeason"]
    fix_sample = UserRunTimeConfig["fix_sample"]
    LOADCHANGEMODULE = UserRunTimeConfig["LOADCHANGEMODULE"]
    filter_make = UserRunTimeConfig["filter_make"] 
    filter_use = UserRunTimeConfig["filter_use"]
    n_cluster = UserRunTimeConfig["n_cluster"]
    moment_matching = UserRunTimeConfig["moment_matching"]
    n_tree_compare = UserRunTimeConfig["n_tree_compare"]

    #############################
    ##Non configurable settings##
    #############################

    regular_seasons = ["winter", "spring", "summer", "fall"]
    lengthPeakSeason = 24
    LeapYearsInvestment = 5
    time_format = "%d/%m/%Y %H:%M"
    if version in ["europe_v51","europe_reduced_v51"]:
        north_sea = True
    else:
        north_sea = False

    #######
    ##RUN##
    #######

    name = version + '_reg' + str(lengthRegSeason) + \
        '_peak' + str(lengthPeakSeason) + \
        '_sce' + str(NoOfScenarios)
    if filter_use:
        name = name + "_filter" + str(n_cluster)
    if moment_matching:
        name = name + "_moment" + str(n_tree_compare)

    name = name + str(datetime.now().strftime("_%Y%m%d%H%M"))
    tab_file_path = 'Data handler/' + version + '/Tab_Files_' + name + '_seed' + str(SEED)
    scenario_data_path = 'Data handler/' + version + '/ScenarioData'

    Period = [i + 1 for i in range(int((Horizon-2020)/LeapYearsInvestment))]
    
    if version in ["europe_v51","europe_reduced_v51"]:
        dict_countries = {"AT": "Austria", "BA": "BosniaH", "BE": "Belgium",
                        "BG": "Bulgaria", "CH": "Switzerland", "CZ": "CzechR",
                        "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
                        "ES": "Spain", "FI": "Finland", "FR": "France",
                        "GB": "GreatBrit.", "GR": "Greece", "HR": "Croatia",
                        "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
                        "LT": "Lithuania", "LU": "Luxemb.", "LV": "Latvia",
                        "MK": "Macedonia", "NL": "Netherlands", "NO": "Norway",
                        "PL": "Poland", "PT": "Portugal", "RO": "Romania",
                        "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia",
                        "SK": "Slovakia", "MF": "MorayFirth", "FF": "FirthofForth",
                        "DB": "DoggerBank", "HS": "Hornsea", "OD": "OuterDowsing",
                        "NF": "Norfolk", "EA": "EastAnglia", "BS": "Borssele",
                        "HK": "HollandseeKust", "HB": "HelgolanderBucht", "NS": "Nordsoen",
                        "UN": "UtsiraNord", "SN1": "SorligeNordsjoI", "SN2": "SorligeNordsjoII"}
    elif version in ["reduced"]:
        dict_countries = {"DE": "Germany", "DK": "Denmark", "FR": "France"}
    elif version in ["europe_v50_mod"]:
        dict_countries = {"AT": "Austria", "BA": "BosniaH", "BE": "Belgium",
                  "BG": "Bulgaria", "CH": "Switzerland", "CZ": "CzechR",
                  "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
                  "ES": "Spain", "FI": "Finland", "FR": "France",
                  "GB": "GreatBrit.", "GR": "Greece", "HR": "Croatia",
                  "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
                  "LT": "Lithuania", "LU": "Luxemb."}
    else :
        dict_countries = {"AT": "Austria", "BA": "BosniaH", "BE": "Belgium",
                        "BG": "Bulgaria", "CH": "Switzerland", "CZ": "CzechR",
                        "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
                        "ES": "Spain", "FI": "Finland", "FR": "France",
                        "GB": "GreatBrit.", "GR": "Greece", "HR": "Croatia",
                        "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
                        "LT": "Lithuania", "LU": "Luxemb.", "LV": "Latvia",
                        "MK": "Macedonia", "NL": "Netherlands", "NO": "Norway",
                        "PL": "Poland", "PT": "Portugal", "RO": "Romania",
                        "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia",
                        "SK": "Slovakia"}

    generate_random_scenario(filepath = scenario_data_path,
                            tab_file_path = tab_file_path,
                            scenarios = NoOfScenarios,
                            seasons = regular_seasons,
                            Periods = len(Period),
                            regularSeasonHours = lengthRegSeason,
                            peakSeasonHours = lengthPeakSeason,
                            dict_countries = dict_countries,
                            time_format = time_format,
                            filter_make = filter_make,
                            filter_use = filter_use,
                            n_cluster = n_cluster,
                            moment_matching = moment_matching,
                            n_tree_compare = n_tree_compare,
                            fix_sample = fix_sample,
                            north_sea = north_sea,
                            LOADCHANGEMODULE = LOADCHANGEMODULE,
                            seed = SEED,
                            scenario_to_generate = scenario_idx)

    return tab_file_path