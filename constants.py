COMOV_STR = ' (Lyso NP +)'
COMOV_SUFFIX = '_np'

NOCOMOV_STR = ' (Lyso NP -)'
NOCOMOV_SUFFIX = '_nonp'

ALL_STR = ' (All lyso)'
ALL_SUFFIX = 'all'

comovement_names = {
    COMOV_SUFFIX: COMOV_STR,
    NOCOMOV_SUFFIX: NOCOMOV_STR,
    ALL_SUFFIX: ALL_STR,
}

# Used for active motion calculation
WV_SCALE = 20
WV_PAD = WV_SCALE // 2

FPS = 5.11

distr_names = {
    'lognormal': 'Log-normal',
    'power_law': 'Power law',
    'truncated_power_law': 'Truncated power law',
    'stretched_exponential': 'Stretched exponential',
    'exponential': 'Exponential',
}
distr_names_reverse = {v: k for k, v in distr_names.items()}

celltype_names = {
    'ht': 'HT1080',
    'mef': 'MEF',
    'mcf10a': 'MCF-10A',
    'mda231': 'MDA-MB-231',
    'mcf7': 'MCF-7',
}
celltype_names_reverse = {v: k for k, v in celltype_names.items()}

datatype_names = {
    'run_l': 'Run length',
    'run_t': 'Run time',
    'flight_l': 'Flight length',
    'flight_t': 'Flight time',
}

datatype_names_more = {
    **datatype_names,
    **{k + COMOV_SUFFIX: v + COMOV_STR
       for k, v in datatype_names.items()},
}

datatype_names_renaming = {
    'run_l': 'Run length',
    'run_t': 'Run time',
    'flight_l': 'Flight length',
    'flight_t': 'Flight time',
    COMOV_SUFFIX: COMOV_STR,
}


def _make_datatype_names_renaming_reverse():
    _datatype_names_renaming_reverse = {}

    for k_, v_ in datatype_names_renaming.items():
        if k_ != COMOV_SUFFIX:
            _datatype_names_renaming_reverse[v_] = k_
            _datatype_names_renaming_reverse[v_ + COMOV_STR] = \
                k_ + COMOV_SUFFIX

    return _datatype_names_renaming_reverse


datatype_names_renaming_reverse = _make_datatype_names_renaming_reverse()


treatment_names = {
    'nonp': 'Control/no NPs',
    '80-20': '80-20',
    '91-9': '91-9',
    'c9np': 'C9',
    'tmanp': 'TMA',
}
treatment_names_shorter = {
    'nonp': 'Control',
    '80-20': '80-20',
    '91-9': '91-9',
    'c9np': 'C9',
    'tmanp': 'TMA',
}
all_lyso_str = ' (All Lyso)'
treatment_names_more = {
    'nonp': treatment_names['nonp'],
    **{k: v + all_lyso_str for k, v in treatment_names.items() if k != 'nonp'},
    **{k + COMOV_SUFFIX: v + COMOV_STR
       for k, v in treatment_names.items() if k != 'nonp'},
}

treatment_names_reverse = {v: k for k, v in treatment_names.items()}

waic_index_renamings_dict = {
    'Data type': datatype_names_renaming,
    'Cell type': celltype_names,
    'Treatment': treatment_names,
}
waic_index_names_list = list(waic_index_renamings_dict.keys())

waic_index_renamings_reverse_dict = {
    'Data type': datatype_names_renaming_reverse,
    'Cell type': celltype_names_reverse,
    'Treatment': treatment_names_reverse,
}

active_t_ratio_names = {
    'active_t': 'Active time',
    'total_t_padded': 'Total time',
    'active_t_ratio': 'Active/total time ratio',
}

active_t_ratio_names_more = {
    **active_t_ratio_names,
    **{k + COMOV_SUFFIX: v + COMOV_STR
       for k, v in active_t_ratio_names.items()},
}

active_t_ratio_names_more_reverse = {
    v: k for k, v in active_t_ratio_names_more.items()
}

treatments_order = [
    'nonp', '80-20', '80-20_np',
    'tmanp', 'tmanp_np',
]

celltypes_order = [
    'mcf10a', 'mda231', 'mcf7', 'mef', 'ht',
]

symbols = {
    'mu': 'μ',
    'sigma': 'σ',
    'lambda': 'λ',
    'alpha': 'α',
    'beta': 'β',
}

msd_df_index_order = {
    'celltype': 0,
    'treatment': 1,
    'comov': 2,
}
